"""
研究：網路文章「MACD 三關」（柱狀體斜率／零軸位置／背離次數）是否真的有用。只做統計，不改規則。
輸出 research/macd.txt、research/macd_trades_*.csv。

測試對象：產業清單裡全部個股（不含 ETF），另外單獨列出 2059 川湖。
成本：手續費 0.1425%×2＋證交稅 0.3%，未含滑價。進場一律用「訊號隔天開盤價」，不偷看未來。

一、純 MACD 策略（出場也用 MACD）
    S0 金叉買、死叉賣（文章說大家都會、會被巴）
    S1 文章：柱狀體翻紅且連 2 天變長＋DIF、DEA 都在零軸上＋DIF 剛離零軸不遠（DIF/股價 ≤ 1.5%）
       出場：柱狀體連 3 天縮短（文章說「要準備跑」）或翻綠
    S2 同 S1 進場，但抱到翻綠才出場（檢查「縮短 3 天就跑」是否有幫助）
    對照：同期間買進持有

二、套用你的系統框架：訊號隔天開盤買，15 日內先到淨利 3% 算成功，先跌 3% 算停損，期滿用收盤價。
    比較各種 MACD 狀態，看「先到停利率」和平均淨報酬有沒有比「隨便哪一天進場」好。

三、背離次數：股價波段高點創新高、但 DIF 較前一個高點低 → 第 1／2／3 次背離。
    看背離確認後 10、20 日報酬，以及 20 日內跌超過 10% 的機率（文章說第三次「隨時會崩」）。

四、當作股票系統的濾網：跑現行引擎回測，把每筆交易依「訊號當天的 MACD 狀態」分組比較。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402
import scanner as S  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

FOCUS = "2059"
CUT = "2025-01-01"
NEAR_ZERO = 1.5  # DIF/股價 % ≤ 此值 → 「剛離零軸不遠」
OUT = Path(os.environ.get("MACD_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)

t0 = time.time()
stocks = {}
for sec in SECTORS:
    for s in stocks_of(sec):
        if not E.is_etf(s["code"]):
            stocks.setdefault(s["code"], s)

if os.environ.get("MACD_FAKE"):  # 離線測試用合成資料
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2016-01-01", "2026-10-16")
    prices, mkt = {}, None
    for code in list(stocks)[:8]:
        r = rng.normal(0.0004, 0.02, len(idx))
        c = 100 * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, 0.005, len(idx)))
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                           "close": c, "volume": rng.integers(500, 5000, len(idx)).astype(float)}, index=idx)
        prices[code] = (df, "fake", [])
    prices.setdefault(FOCUS, prices[next(iter(prices))])
else:
    prices = S.fetch_batch(list(stocks.values()))
    mkt = E.load_index(E.DATA_YEARS)
    if FOCUS not in prices or prices[FOCUS][0] is None:
        df, src, _ = E.load_prices(FOCUS)
        if df is not None:
            prices[FOCUS] = (df, src, [])

data = {k: v[0] for k, v in prices.items() if v and v[0] is not None and len(v[0]) >= 120}
print(f"個股 {len(data)} 檔有資料，抓資料 {time.time() - t0:.0f} 秒")


def add_macd(df):
    d = df.copy()
    c = d["close"]
    d["dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["dea"] = d["dif"].ewm(span=9, adjust=False).mean()
    d["hist"] = d["dif"] - d["dea"]
    d["difp"] = d["dif"] / c * 100
    d["ma60"] = c.rolling(60).mean()
    return d


def conditions(d):
    h, dif, dea, difp = d["hist"], d["dif"], d["dea"], d["difp"]
    up2 = (h > 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2))
    shrink3 = (h > 0) & (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3))
    above = (dif > 0) & (dea > 0)
    rising = dif > dif.shift(1)
    cond = {
        "任一天（對照）": pd.Series(True, index=d.index),
        "金叉當天": (h > 0) & (h.shift(1) <= 0),
        "第一關：柱體翻紅且連2天變長": up2,
        "第一關反面：柱體連3天縮短（文章說要跑）": shrink3,
        "第二關：零軸上、剛離零軸（DIF/價≤1.5%）": above & rising & (difp <= NEAR_ZERO),
        "第二關：零軸上、中段（1.5～4%）": above & rising & (difp > NEAR_ZERO) & (difp <= 4),
        "第二關：零軸上、遠離（>4%，文章說主升段重壓）": above & rising & (difp > 4),
        "第二關：零軸上但DIF下彎（高檔鈍化）": above & ~rising,
        "文章組合：剛離零軸＋柱體加速": up2 & above & (difp <= NEAR_ZERO),
    }
    return cond


def fwd_3pct(arr, i, buy_c, sell_c, hold=15, target=3.0, stop_pct=3.0):
    """隔天開盤買，15 日內先到淨利 target% 或先跌 stop_pct%。回傳 (結果, 淨報酬%)。"""
    O, H, L, C = arr
    e = i + 1
    if e + hold - 1 >= len(O):
        return None
    entry = O[e]
    tp = entry * (1 + target / 100 + buy_c) / (1 - sell_c)
    sl = entry * (1 - stop_pct / 100)
    for j in range(e, e + hold):
        if L[j] <= sl:  # 同一天兩邊都碰到 → 保守算停損
            x = min(O[j], sl) if j > e else sl
            return "停損", E.net_return(entry, x, buy_c, sell_c)
        if H[j] >= tp:
            return "停利", E.net_return(entry, max(O[j], tp) if j > e else tp, buy_c, sell_c)
    return "期滿", E.net_return(entry, C[e + hold - 1], buy_c, sell_c)


def run_strategy(d, buy_c, sell_c, kind):
    O, h = d["open"].to_numpy(), d["hist"].to_numpy()
    cond = conditions(d)
    entry_sig = (cond["金叉當天"] if kind == "S0" else cond["文章組合：剛離零軸＋柱體加速"]).to_numpy()
    dates = d.index
    trades, i, n = [], 60, len(d)
    while i < n - 2:
        if not entry_sig[i]:
            i += 1
            continue
        e = i + 1
        x = None
        for j in range(e, n - 1):
            exit_now = h[j] < 0
            if kind == "S1" and j >= 3:
                exit_now = exit_now or (h[j] < h[j - 1] < h[j - 2] < h[j - 3])
            if exit_now:
                x = j + 1
                break
        if x is None:
            break  # 尚未出場
        r = E.net_return(O[e], O[x], buy_c, sell_c)
        trades.append({"訊號日": dates[i].strftime("%Y-%m-%d"), "進場價": round(O[e], 2),
                       "出場日": dates[x].strftime("%Y-%m-%d"), "出場價": round(O[x], 2),
                       "持有天數": x - e, "淨報酬%": round(r, 2)})
        i = x
    return pd.DataFrame(trades)


def divergences(d, k=3, lookback=80):
    """回傳每個確認日的背離次數事件。用 high 找波段高點（左右各 k 根），確認日＝高點後第 k 天。"""
    H, dif = d["high"].to_numpy(), d["dif"].to_numpy()
    ev, prev, cnt = [], None, 0
    for p in range(k, len(d) - k):
        win = H[p - k:p + k + 1]
        if H[p] != win.max() or (win == H[p]).sum() > 1:
            continue
        if prev is not None and p - prev <= lookback and H[p] > H[prev]:
            cnt = cnt + 1 if dif[p] < dif[prev] else 0
        elif prev is not None and H[p] <= H[prev]:
            cnt = 0
        else:
            cnt = 0
        if prev is not None and H[p] > H[prev]:
            ev.append((p + k, cnt))  # 只記「創新高」的高點
        prev = p
    return ev


def stat_line(name, g, col="淨報酬%", res_col="結果"):
    if g is None or len(g) == 0:
        return f"{name}：0 筆"
    r = g[col]
    s = f"{name}：{len(g)} 筆・平均淨報酬 {r.mean():+.2f}%・中位數 {r.median():+.2f}%・賺錢比率 {(r > 0).mean() * 100:.1f}%"
    if res_col in g:
        s += f"・先到停利 {(g[res_col] == '停利').mean() * 100:.1f}%"
    return s


out = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・引擎 {E.ENGINE_VERSION}・個股 {len(data)} 檔（含 {FOCUS} 川湖："
       f"{'有' if FOCUS in data else '無資料'}）",
       f"資料期間約 {E.DATA_YEARS} 年。已扣手續費與證交稅，未含滑價。前段 ~2024／後段 2025~ 用來檢查是否只在某段有效。", ""]

# ---------------- 一、純 MACD 策略 ----------------
out.append("一、純 MACD 策略（進出場都只看 MACD）")
allS = {k: [] for k in ("S0", "S1", "S2")}
bh = []
for code, df in data.items():
    d = add_macd(df)
    buy_c, sell_c = E.cost_rates(code, E.DEFAULT_PARAMS)
    for kind in allS:
        t = run_strategy(d, buy_c, sell_c, kind)
        if len(t):
            t["代號"] = code
            allS[kind].append(t)
    o = d["open"].to_numpy()
    yrs = (d.index[-1] - d.index[60]).days / 365.25
    bh.append({"代號": code, "持有報酬%": E.net_return(o[61], d["close"].iloc[-1], buy_c, sell_c), "年數": yrs})
names = {"S0": "S0 金叉買死叉賣", "S1": "S1 文章規則（縮短3天就跑）", "S2": "S2 文章進場、翻綠才出"}
for kind, lst in allS.items():
    t = pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()
    allS[kind] = t
    if len(t):
        t.to_csv(OUT / f"macd_trades_{kind}.csv", index=False, encoding="utf-8-sig")
        per = t.groupby("代號")["淨報酬%"].apply(lambda r: (np.prod(1 + r / 100) - 1) * 100)
        out.append(stat_line(names[kind], t) + f"・平均持有 {t['持有天數'].mean():.1f} 天・每檔複利總報酬中位數 {per.median():+.0f}%")
        out.append("    " + stat_line("前段 ~2024", t[t["訊號日"] < CUT]))
        out.append("    " + stat_line("後段 2025~", t[t["訊號日"] >= CUT]))
        f = t[t["代號"] == FOCUS]
        if len(f):
            out.append("    " + stat_line(f"{FOCUS} 川湖", f) + f"・複利總報酬 {(np.prod(1 + f['淨報酬%'] / 100) - 1) * 100:+.0f}%")
    else:
        out.append(f"{names[kind]}：0 筆")
bh = pd.DataFrame(bh)
out.append(f"對照：同期間買進持有，每檔總報酬中位數 {bh['持有報酬%'].median():+.0f}%"
           + (f"・{FOCUS} 川湖 {bh.loc[bh['代號'] == FOCUS, '持有報酬%'].iloc[0]:+.0f}%" if FOCUS in set(bh["代號"]) else ""))
out.append("")

# ---------------- 二、你的框架：15 日內先到淨利 3% ----------------
out.append("二、套用股票系統框架：隔天開盤買、15 日內先到淨利 3%（成功）或先跌 3%（停損）")
rows = []
for code, df in data.items():
    d = add_macd(df)
    buy_c, sell_c = E.cost_rates(code, E.DEFAULT_PARAMS)
    cond = {k: v.to_numpy() for k, v in conditions(d).items()}
    arr = tuple(d[c].to_numpy() for c in ("open", "high", "low", "close"))
    for i in range(60, len(d) - 16):
        r = fwd_3pct(arr, i, buy_c, sell_c)
        if r is None:
            continue
        date = d.index[i].strftime("%Y-%m-%d")
        for name, m in cond.items():
            if m[i]:
                rows.append((name, code, date, r[0], r[1]))
ev = pd.DataFrame(rows, columns=["條件", "代號", "訊號日", "結果", "淨報酬%"])
for name in conditions(add_macd(next(iter(data.values())))).keys():
    g = ev[ev["條件"] == name]
    out.append(stat_line(name, g))
    a, b = g[g["訊號日"] < CUT], g[g["訊號日"] >= CUT]
    if len(a) and len(b):
        out.append(f"    前段 先到停利 {(a['結果'] == '停利').mean() * 100:.1f}%・平均 {a['淨報酬%'].mean():+.2f}%"
                   f"｜後段 先到停利 {(b['結果'] == '停利').mean() * 100:.1f}%・平均 {b['淨報酬%'].mean():+.2f}%")
    f = g[g["代號"] == FOCUS]
    if len(f):
        out.append(f"    {FOCUS} 川湖：{len(f)} 天・先到停利 {(f['結果'] == '停利').mean() * 100:.1f}%・平均 {f['淨報酬%'].mean():+.2f}%")
out.append("（同一段行情會連續好幾天符合條件，筆數是「天數」，彼此不獨立；看比率差距即可）")
out.append("")

# ---------------- 三、背離次數 ----------------
out.append("三、頂背離次數（股價創波段新高、DIF 卻比前一個高點低）→ 確認後的走勢")
drows = []
for code, df in data.items():
    d = add_macd(df)
    O, C, L = d["open"].to_numpy(), d["close"].to_numpy(), d["low"].to_numpy()
    for i, cnt in divergences(d):
        if i + 21 >= len(d):
            continue
        e = O[i + 1]
        drows.append({"代號": code, "訊號日": d.index[i].strftime("%Y-%m-%d"), "背離": min(cnt, 3),
                      "10日%": (C[i + 10] / e - 1) * 100, "20日%": (C[i + 20] / e - 1) * 100,
                      "20日內跌超10%": float(L[i + 1:i + 21].min() / e - 1 <= -0.10)})
dv = pd.DataFrame(drows)
lab = {0: "創新高、無背離（對照）", 1: "第 1 次背離（文章：警告、可續抱）",
       2: "第 2 次背離（文章：確認、減碼）", 3: "第 3 次以上（文章：最後煙火、隨時崩）"}
if len(dv):
    dv.to_csv(OUT / "macd_trades_divergence.csv", index=False, encoding="utf-8-sig")
    for k in range(4):
        g = dv[dv["背離"] == k]
        if len(g) == 0:
            out.append(f"{lab[k]}：0 次")
            continue
        out.append(f"{lab[k]}：{len(g)} 次・10 日平均 {g['10日%'].mean():+.2f}%・20 日平均 {g['20日%'].mean():+.2f}%"
                   f"・20 日後仍上漲 {(g['20日%'] > 0).mean() * 100:.0f}%・20 日內跌超 10% 機率 {g['20日內跌超10%'].mean() * 100:.0f}%")
        a, b = g[g["訊號日"] < CUT], g[g["訊號日"] >= CUT]
        if len(a) and len(b):
            out.append(f"    前段 20 日平均 {a['20日%'].mean():+.2f}%（{len(a)} 次）｜後段 {b['20日%'].mean():+.2f}%（{len(b)} 次）")
    f = dv[dv["代號"] == FOCUS]
    if len(f):
        out.append(f"{FOCUS} 川湖：" + "；".join(
            f"背離{k}：{len(f[f['背離'] == k])} 次、20 日平均 {f[f['背離'] == k]['20日%'].mean():+.1f}%"
            for k in range(4) if len(f[f["背離"] == k])))
out.append("（未扣成本，單純看價格走勢）")
out.append("")

# ---------------- 四、當作股票系統濾網 ----------------
out.append("四、當作股票系統濾網：現行引擎回測的每筆交易，依訊號當天 MACD 狀態分組")
BASE = {"f_stop_atr": True, "f_market": True, "f_sector": False}
bts = []
if True:
    for code, df in data.items():
        try:
            bt = E.Analyzer(df, code, BASE, mkt=mkt).backtest()
        except Exception as e:  # noqa: BLE001
            print(code, e)
            continue
        if not len(bt):
            continue
        d = add_macd(df)
        cond = conditions(d)
        idx = pd.to_datetime(bt["訊號日"])
        bt["代號"] = code
        bt["柱體連2天變長"] = cond["第一關：柱體翻紅且連2天變長"].reindex(idx).to_numpy()
        bt["柱體連3天縮短"] = cond["第一關反面：柱體連3天縮短（文章說要跑）"].reindex(idx).to_numpy()
        bt["DIF/價%"] = d["difp"].reindex(idx).round(2).to_numpy()
        bt["文章組合"] = cond["文章組合：剛離零軸＋柱體加速"].reindex(idx).to_numpy()
        bts.append(bt)
eng = pd.concat(bts, ignore_index=True) if bts else pd.DataFrame()
if len(eng):
    eng.to_csv(OUT / "macd_trades_engine.csv", index=False, encoding="utf-8-sig")
    out.append(stat_line("全部（現行預設）", eng))
    groups = {
        "柱體連2天變長 → 是": eng["柱體連2天變長"] == True,  # noqa: E712
        "柱體連2天變長 → 否": eng["柱體連2天變長"] != True,  # noqa: E712
        "柱體連3天縮短 → 是": eng["柱體連3天縮短"] == True,  # noqa: E712
        "DIF 在零軸下": eng["DIF/價%"] <= 0,
        "DIF 零軸上 ≤1.5%": (eng["DIF/價%"] > 0) & (eng["DIF/價%"] <= NEAR_ZERO),
        "DIF 零軸上 >1.5%": eng["DIF/價%"] > NEAR_ZERO,
        "符合文章組合": eng["文章組合"] == True,  # noqa: E712
    }
    for name, m in groups.items():
        g = eng[m]
        out.append("  " + stat_line(name, g))
        a, b = g[g["訊號日"] < CUT], g[g["訊號日"] >= CUT]
        if len(a) and len(b):
            out.append(f"      前段 先到停利 {(a['結果'] == '停利').mean() * 100:.1f}%・平均 {a['淨報酬%'].mean():+.2f}%（{len(a)}）"
                       f"｜後段 先到停利 {(b['結果'] == '停利').mean() * 100:.1f}%・平均 {b['淨報酬%'].mean():+.2f}%（{len(b)}）")
    f = eng[eng["代號"] == FOCUS]
    if len(f):
        out.append(f"  {FOCUS} 川湖：{len(f)} 筆・先到停利 {(f['結果'] == '停利').mean() * 100:.1f}%・平均 {f['淨報酬%'].mean():+.2f}%"
                   f"・其中柱體連2天變長 {int((f['柱體連2天變長'] == True).sum())} 筆")  # noqa: E712
else:
    out.append("引擎回測沒有交易")

out.append("")
out.append(f"總耗時 {time.time() - t0:.0f} 秒")
text = "\n".join(out)
(OUT / "macd.txt").write_text(text, encoding="utf-8")
print(text)
