"""
研究：有沒有「淨賺 3～5%、成功率 70～80%」而且長期不虧錢的短線方法？（2026-10-21）
只做統計，不改規則。輸出 research/winrate.txt、research/winrate_2409.txt。

做法（避免自己騙自己）：
1. 股票池：產業清單全部個股＋ETF，再加台灣 50 大型股與 2409 友達，約 200 檔、10 年日K。
2. 進場：訊號日收盤後判斷，「隔天開盤」買（不偷看未來）；同一檔同時只持有一筆（不重複計算同一段行情）。
3. 每種進場條件 × 每種出場方式都算：成功率（先到停利）、淨賺比例、平均淨報酬（期望值）、最差一筆。
4. 樣本內／樣本外：2016～2022 用來「挑」，2023～2026 用來「驗」。只有兩段都達標才算數。
5. 成本：手續費 0.1425%×2＋證交稅（股票 0.3%、ETF 0.1%），另外再扣滑價每邊 0.1%。
   ex-dividend 未還原（與 App 一致），會讓少數交易看起來比實際差。

達標定義：成功率 ≥ 70%、平均淨報酬 > 0、樣本內與樣本外各 ≥ 50 筆。
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

OUT = Path(os.environ.get("WR_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
SPLIT = pd.Timestamp("2023-01-01")
SLIP = 0.001  # 每邊滑價
HOLD = 15
FOCUS = "2409"

EXTRA = [  # 台灣 50 常見成分股＋友達（補齊非科技類，避免只測 AI 股）
    "2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", "2303", "2412", "2886", "2884",
    "1301", "1303", "2002", "2885", "2892", "2880", "5880", "2883", "2887", "3711", "2357", "2395",
    "3045", "4904", "2912", "1216", "2207", "2603", "2609", "2615", "1101", "1326", "6505", "2327",
    "3008", "2379", "3034", "2301", "5871", "2890", "9910", "2409", "3481", "2344", "2408", "2618",
    "1402", "2105", "2474", "4938", "2801", "2834", "1102", "2610",
]

t0 = time.time()
universe = {}
for sec in SECTORS:
    for s in stocks_of(sec):
        universe.setdefault(s["code"], s)
for c in EXTRA:
    universe.setdefault(c, {"code": c, "name": c, "market": "TW"})

if os.environ.get("WR_FAKE"):  # 離線測試用合成資料
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2016-01-01", "2026-10-20")
    prices = {}
    for code in list(universe)[:12] + [FOCUS, "0050"]:
        r = rng.normal(0.0004, 0.018, len(idx))
        c = 50 * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, 0.006, len(idx)))
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, .008, len(idx)))),
                           "low": np.minimum(o, c) * (1 - abs(rng.normal(0, .008, len(idx)))),
                           "close": c, "volume": rng.integers(500, 5000, len(idx)).astype(float)}, index=idx)
        prices[code] = (df, "fake", [])
    mkt = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx)))), index=idx)
else:
    prices = S.fetch_batch(list(universe.values()))
    mkt = E.load_index(E.DATA_YEARS)

data = {k: v[0] for k, v in prices.items() if v and v[0] is not None and len(v[0]) >= 260}
n_etf = sum(E.is_etf(k) for k in data)
print(f"有資料 {len(data)} 檔（個股 {len(data) - n_etf}、ETF {n_etf}），抓資料 {time.time() - t0:.0f} 秒，大盤 {'有' if mkt is not None else '無'}")


# ------------------------------------------------------------------ 指標
def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prep(df):
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    for n in (5, 10, 20, 60, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
    d["rsi2"] = rsi(c, 2)
    d["rsi14"] = rsi(c, 14)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["atrp"] = d["atr"] / c * 100
    sd = c.rolling(20).std()
    d["bbl"] = d["ma20"] - 2 * sd
    rng_ = (h - l).replace(0, np.nan)
    d["ibs"] = (c - l) / rng_
    d["vr"] = v / v.shift(1).rolling(5).mean()
    lo9, hi9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = ((c - lo9) / (hi9 - lo9).replace(0, np.nan) * 100).fillna(50)
    k = np.zeros(len(d)); dd = np.zeros(len(d)); k[0] = dd[0] = 50
    rv = rsv.to_numpy()
    for i in range(1, len(d)):
        k[i] = k[i - 1] * 2 / 3 + rv[i] / 3
        dd[i] = dd[i - 1] * 2 / 3 + k[i] / 3
    d["k"], d["d"] = k, dd
    d["hh10"] = h.rolling(10).max()
    d["hh240"] = h.shift(1).rolling(240).max()
    d["chg"] = c.pct_change() * 100
    return d


def setups(d, m):
    """回傳 {名稱: 布林 Series}。m = 大盤 DataFrame（對齊日期）。"""
    c, o = d["close"], d["open"]
    up200 = c > d["ma200"]
    up60 = (c > d["ma60"]) & (d["ma60"] > d["ma60"].shift(5))
    red = (c > o) & (d["chg"] > 0)
    down3 = (d["chg"] < 0) & (d["chg"].shift(1) < 0) & (d["chg"].shift(2) < 0)
    pull = (d["hh10"] - c) / d["hh10"] * 100
    risk_on = m["c"] > m["ma60"]
    s = {
        "對照：任一天": pd.Series(True, index=d.index),
        "Connors RSI2<10 且站上200日線": up200 & (d["rsi2"] < 10),
        "Connors RSI2<5 且站上200日線": up200 & (d["rsi2"] < 5),
        "連跌3天 且站上200日線": up200 & down3,
        "收在當日低檔 IBS<0.2 且站上200日線": up200 & (d["ibs"] < 0.2) & (c < d["ma5"]),
        "跌破布林下軌 且站上200日線": up200 & (c < d["bbl"]),
        "多頭回檔到月線（MA20±3%）": up60 & (d["ma20"] > d["ma60"]) & (c <= d["ma20"] * 1.01) & (c >= d["ma20"] * 0.97),
        "多頭急跌≥2ATR（離10日高）": up60 & ((d["hh10"] - c) >= 2 * d["atr"]),
        "多頭回檔3～12%後第一根帶量紅K（≈策略B）": up60 & (pull.shift(1).between(3, 12)) & red
            & (c > d["high"].shift(1)) & (d["vr"] >= 1.2),
        "KD 低檔金叉（K<25）且站上季線": (d["k"] > d["d"]) & (d["k"].shift(1) <= d["d"].shift(1))
            & (d["k"].shift(1) < 25) & (c > d["ma60"]),
        "創 52 週新高＋量比≥1.5": (c > d["hh240"]) & (d["vr"] >= 1.5) & red,
        "大盤單日跌≥2.5%（恐慌隔天買）": m["chg"] <= -2.5,
        "大盤5日跌≥6%": m["chg5"] <= -6,
    }
    for name in list(s):
        if name.startswith(("對照", "大盤")):
            continue
        s[name + "｜大盤多頭"] = s[name] & risk_on
    return {k: v.fillna(False).to_numpy() for k, v in s.items()}


EXITS = {  # 名稱: (淨利目標%, 停損種類, 參數)
    "停利3%／停損3%": (3, "pct", 3),
    "停利3%／停損5%": (3, "pct", 5),
    "停利3%／停損7%": (3, "pct", 7),
    "停利3%／停損10%": (3, "pct", 10),
    "停利3%／不停損（15日到期出）": (3, "none", None),
    "停利3%／停損2ATR": (3, "atr", 2),
    "停利4%／停損4%": (4, "pct", 4),
    "停利5%／停損5%": (5, "pct", 5),
    "停利5%／停損8%": (5, "pct", 8),
    "Connors出場：收盤站上MA5（最多15日、無停損）": (None, "ma5", None),
}


def simulate(d, sig, ex, buy_c, sell_c):
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    ATR, MA5 = d["atr"].to_numpy(), d["ma5"].to_numpy()
    tgt, kind, par = ex
    bc, sc = buy_c + SLIP, sell_c + SLIP
    out, i, n = [], 200, len(d)
    while i < n - 1 - HOLD:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        entry = O[e]
        tp = entry * (1 + bc) * (1 + tgt / 100) / (1 - sc) if tgt else np.inf
        if kind == "pct":
            sl = entry * (1 - par / 100)
        elif kind == "atr":
            sl = entry - par * ATR[i]
        else:
            sl = -np.inf
        res, x, xj = "期滿", C[e + HOLD - 1], e + HOLD - 1
        mae = 0.0
        for j in range(e, e + HOLD):
            mae = min(mae, L[j] / entry - 1)
            if L[j] <= sl:
                res, x, xj = "停損", (min(O[j], sl) if j > e else sl), j
                break
            if H[j] >= tp:
                res, x, xj = "停利", (max(O[j], tp) if j > e else tp), j
                break
            if kind == "ma5" and C[j] > MA5[j]:
                res, x, xj = "站上MA5", C[j], j
                break
        r = (x * (1 - sc)) / (entry * (1 + bc)) * 100 - 100
        out.append((d.index[i], res, r, xj - e + 1, mae * 100))
        i = xj + 1
    return out


# ------------------------------------------------------------------ 大盤
if mkt is not None:
    M = pd.DataFrame({"c": mkt})
    M["ma60"] = M["c"].rolling(60).mean()
    M["chg"] = M["c"].pct_change() * 100
    M["chg5"] = M["c"].pct_change(5) * 100
else:
    M = None

rows = []
for code, df in data.items():
    d = prep(df)
    if M is not None:
        m = M.reindex(d.index).ffill()
    else:
        m = pd.DataFrame({"c": np.nan, "ma60": np.nan, "chg": np.nan, "chg5": np.nan}, index=d.index)
    sg = setups(d, m)
    buy_c, sell_c = E.cost_rates(code, E.DEFAULT_PARAMS)
    typ = "ETF" if E.is_etf(code) else "個股"
    for sname, sig in sg.items():
        for ename, ex in EXITS.items():
            for (dt, res, r, days, mae) in simulate(d, sig, ex, buy_c, sell_c):
                rows.append((code, typ, sname, ename, dt, res, r, days, mae))
T = pd.DataFrame(rows, columns=["代號", "類型", "進場", "出場", "訊號日", "結果", "淨報酬%", "持有天數", "最大不利%"])
print(f"模擬完成 {len(T)} 筆，耗時 {time.time() - t0:.0f} 秒")


def stats(g):
    if len(g) == 0:
        return None
    r = g["淨報酬%"]
    hit = (g["結果"].isin(["停利", "站上MA5"]) & (r > 0)) if g["出場"].iloc[0].startswith("Connors") else (g["結果"] == "停利")
    return {"n": len(g), "win": hit.mean() * 100, "pos": (r > 0).mean() * 100, "avg": r.mean(),
            "med": r.median(), "worst": r.min(), "days": g["持有天數"].mean(),
            "loss_avg": r[r <= 0].mean() if (r <= 0).any() else 0.0,
            "gain_avg": r[r > 0].mean() if (r > 0).any() else 0.0}


def fmt(s):
    if s is None:
        return "0 筆"
    return (f"{s['n']} 筆・成功率 {s['win']:.0f}%・淨賺比例 {s['pos']:.0f}%・平均 {s['avg']:+.2f}%"
            f"（賺 {s['gain_avg']:+.1f}／賠 {s['loss_avg']:+.1f}）・最差 {s['worst']:+.1f}%・持有 {s['days']:.1f} 天")


lines = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・股票池 {len(data)} 檔（個股 {len(data) - n_etf}、ETF {n_etf}）・約 {E.DATA_YEARS} 年日K",
         "隔天開盤買、同檔不重疊、15 個交易日內；已扣手續費、證交稅，另扣滑價每邊 0.1%。",
         "成功率＝先到停利；樣本內 2016～2022（拿來挑）、樣本外 2023～（拿來驗）。",
         "達標＝成功率 ≥70%、平均淨報酬 >0、兩段各 ≥50 筆。", ""]

summary = []
for typ in ("個股", "ETF"):
    for (sname, ename), g in T[T["類型"] == typ].groupby(["進場", "出場"], sort=False):
        a, b = stats(g[g["訊號日"] < SPLIT]), stats(g[g["訊號日"] >= SPLIT])
        if a is None or b is None:
            continue
        ok = all(x["n"] >= 50 and x["win"] >= 70 and x["avg"] > 0 for x in (a, b))
        summary.append({"類型": typ, "進場": sname, "出場": ename, "IS": a, "OOS": b, "ok": ok, "ALL": stats(g)})

for typ in ("個股", "ETF"):
    sub = [x for x in summary if x["類型"] == typ]
    lines.append(f"==================== {typ} ====================")
    good = [x for x in sub if x["ok"]]
    lines.append(f"★ 兩段都達標（成功率≥70% 且賺錢）：{len(good)} 組")
    for x in sorted(good, key=lambda x: -x["OOS"]["avg"]):
        lines += [f"  {x['進場']} × {x['出場']}", f"    樣本內 {fmt(x['IS'])}", f"    樣本外 {fmt(x['OOS'])}"]
    hw = [x for x in sub if x["IS"]["win"] >= 70 and x["IS"]["n"] >= 50 and not x["ok"]]
    lines.append(f"✗ 成功率≥70% 但不賺錢或樣本外失效：{len(hw)} 組（前 10，依樣本外平均）")
    for x in sorted(hw, key=lambda x: -x["OOS"]["avg"])[:10]:
        lines += [f"  {x['進場']} × {x['出場']}", f"    樣本內 {fmt(x['IS'])}", f"    樣本外 {fmt(x['OOS'])}"]
    pos = [x for x in sub if x["IS"]["avg"] > 0 and x["OOS"]["avg"] > 0 and x["IS"]["n"] >= 50 and x["OOS"]["n"] >= 50]
    lines.append(f"◎ 兩段平均都賺錢（不論成功率）：{len(pos)} 組（前 15，依樣本外平均）")
    for x in sorted(pos, key=lambda x: -min(x["IS"]["avg"], x["OOS"]["avg"]))[:15]:
        lines += [f"  {x['進場']} × {x['出場']}", f"    樣本內 {fmt(x['IS'])}", f"    樣本外 {fmt(x['OOS'])}"]
    lines.append("— 停損寬度 vs 成功率（對照：任一天）—")
    for x in sub:
        if x["進場"] == "對照：任一天":
            lines.append(f"  {x['出場']}：{fmt(x['ALL'])}")
    lines.append("")

# 各進場條件在「停利3%／停損3%」下的表現（與股票系統框架對照）
lines.append("==================== 各條件 × 停利3%／停損3%（全期間）====================")
for typ in ("個股", "ETF"):
    for x in summary:
        if x["類型"] == typ and x["出場"] == "停利3%／停損3%":
            lines.append(f"  [{typ}] {x['進場']}：{fmt(x['ALL'])}")
(OUT / "winrate.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# ------------------------------------------------------------------ 2409 友達：股票系統完整報告＋各條件今日是否觸發
fl = [f"2409 友達｜{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}"]
try:
    df = data.get(FOCUS)
    if df is None:
        df, src, _ = E.load_prices(FOCUS)
    if df is not None:
        mk = E.market_env(mkt)
        rep = E.Analyzer(df, FOCUS, {}, mkt=mkt).report(mk)
        fl.append(f"資料至 {df.index[-1]:%Y-%m-%d}，收盤 {df['close'].iloc[-1]:.2f}，ATR% {rep.get('atrp')}")
        fl.append(f"決策：{rep['decision']}｜結構：{rep['structure']}｜階段：{rep['stage']}｜符合度 {rep['fit']:.0f}%")
        fl.append("原因：" + "；".join(rep["why"]))
        fl.append("過濾：" + "；".join(rep["filters"]["info"]))
        if rep.get("trigger"):
            fl.append(f"觸發：{rep['trigger']}")
        if rep.get("plan"):
            fl.append(f"計畫：{rep['plan']}")
        fl.append("支持：" + "；".join(map(str, rep["pros"])))
        fl.append("風險：" + "；".join(map(str, rep["cons"])))
        fl.append("衝突：" + "；".join(map(str, rep["conflicts"])))
        for mo in rep["modules"]:
            fl.append(f"  模組{mo['no']} {mo['name']}：{mo['score']}｜{mo['text']}")
        fl.append(f"回測：{rep['stats']}")
        fl.append(f"近期型態：{rep['recent_hits']}")
        d = prep(df)
        m = M.reindex(d.index).ffill() if M is not None else None
        if m is not None:
            sg = setups(d, m)
            fl.append("今日（最後一根）符合的研究條件：" + ("、".join(k for k, v in sg.items() if v[-1]) or "無"))
        last = d.iloc[-1]
        fl.append(f"MA5 {last['ma5']:.2f} MA10 {last['ma10']:.2f} MA20 {last['ma20']:.2f} MA60 {last['ma60']:.2f} "
                  f"MA200 {last['ma200']:.2f} RSI2 {last['rsi2']:.1f} RSI14 {last['rsi14']:.1f} K {last['k']:.1f} D {last['d']:.1f}")
        fl.append("最近 15 根：")
        fl.append(df.tail(15).round(2).to_string())
        g = T[T["代號"] == FOCUS]
        fl.append("\n2409 歷史（每種條件 × 停利3%／停損3%）：")
        for sname, gg in g[g["出場"] == "停利3%／停損3%"].groupby("進場", sort=False):
            fl.append(f"  {sname}：{fmt(stats(gg))}")
    else:
        fl.append("抓不到 2409 資料")
except Exception as ex:  # noqa: BLE001
    import traceback
    fl.append("錯誤：" + traceback.format_exc())
(OUT / "winrate_2409.txt").write_text("\n".join(fl), encoding="utf-8")
print("\n".join(fl))
