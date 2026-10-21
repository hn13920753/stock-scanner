"""
研究：強勢股大跌時買進、賺 3～10% 就跑、最多忍 −20%（使用者實戰方法，例：3653 健策 2026/5 從 5000 多跌到 3481 買、4980 賣）。
只做統計，不改規則。輸出 research/dip.txt。

進場：收盤距 120 日最高點跌了 X%（20／25／30／35%）的第一天，隔天開盤買；同一檔賣掉前不再買。
「長期看好」用資料定義（不同版本都測）：
  無濾網／大漲過的強勢股（跌前 120 日高 ≥ 一年前收盤 ×1.5）／年線仍上揚／再加「止跌紅K」確認
出場：先到停利（淨 3／5／10%）就賣；跌到 −20%（或 −15%、不停損）就認賠；最多抱 20／60／120 個交易日，到期收盤賣。
也測「分批」：先買一半，再跌 10% 補另一半，停利以平均成本計算、停損以第一筆價格 −20%；報酬以兩份資金計算（沒補到的一半視為閒置）。
成本：手續費＋證交稅＋滑價每邊 0.1%。還原權息價。樣本內 ～2022、樣本外 2023～。

重要偏差：股票池是「今天還在、成交值前 300」的股票，已下市或一蹶不振的股票不在裡面，
抄底類策略會因此被高估（存活者偏差）。結果請打折看。
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
from sectors import SECTORS, stocks_of  # noqa: E402

OUT = Path(os.environ.get("DIP_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
SPLIT = pd.Timestamp("2023-01-01")
SLIP = 0.001
FAKE = bool(os.environ.get("DIP_FAKE"))
CASES = ["3653", "2059", "2409"]
t0 = time.time()

EXTRA = ["2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", "2303", "2412", "2886", "2884",
         "1301", "1303", "2002", "2885", "2892", "2880", "5880", "2883", "2887", "3711", "2357", "2395",
         "3045", "4904", "2912", "1216", "2207", "2603", "2609", "2615", "1101", "1326", "6505", "2327",
         "3008", "2379", "3034", "2301", "5871", "2890", "9910", "2409", "3481", "2344", "2408", "2618", "3653"]


def universe():
    u = {}
    for sec in SECTORS:
        for s in stocks_of(sec):
            if not E.is_etf(s["code"]):
                u.setdefault(s["code"], s["market"])
    for c in EXTRA:
        u.setdefault(c, "TW")
    if FAKE:
        return u
    try:
        import requests
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30)
        df = pd.DataFrame(r.json())
        df = df[df["Code"].str.fullmatch(r"[1-9]\d{3}")]
        df["tv"] = pd.to_numeric(df["TradeValue"], errors="coerce")
        for c in df.nlargest(300, "tv")["Code"]:
            u.setdefault(c, "TW")
    except Exception as ex:  # noqa: BLE001
        print("證交所清單失敗：", ex)
    return u


def fetch(u):
    import yfinance as yf

    def grab(tks):
        got = {}
        for k in range(0, len(tks), 80):
            part = tks[k:k + 80]
            try:
                d = yf.download(part, period="11y", interval="1d", group_by="ticker", auto_adjust=True,
                                actions=False, threads=True, progress=False)
            except Exception:  # noqa: BLE001
                continue
            for tk in part:
                try:
                    sub = d[tk] if isinstance(d.columns, pd.MultiIndex) else d
                    sub = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                    sub = sub[(sub["volume"] > 0) & (sub["high"] >= sub["low"])]
                    if len(sub) >= 300:
                        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                        got[tk.split(".")[0]] = sub.astype(float)
                except Exception:  # noqa: BLE001
                    continue
        return got

    out = grab([f"{c}.{m}" for c, m in u.items()])
    miss = [c for c in u if c not in out]
    out.update(grab([f"{c}.{'TWO' if u[c] == 'TW' else 'TW'}" for c in miss]))
    return out


U = universe()
if FAKE:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-10-01", "2026-10-20")
    data = {}
    for code in list(U)[:30] + CASES:
        c = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.025, len(idx))))
        o = c * (1 + rng.normal(0, 0.008, len(idx)))
        data[code] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                                   "close": c, "volume": 1000.0}, index=idx)
else:
    data = fetch(U)
print(f"個股 {len(data)} 檔有資料，{time.time() - t0:.0f} 秒")


def prep(df):
    d = df.copy()
    c, h = d["close"], d["high"]
    d["hh120"] = h.rolling(120).max()
    d["dd"] = c / d["hh120"] - 1
    d["ma200"] = c.rolling(200).mean()
    d["strong"] = d["hh120"] >= c.shift(250) * 1.5
    d["up200"] = d["ma200"] > d["ma200"].shift(20)
    d["red_turn"] = (c > d["open"]) & (c > h.shift(1))
    return d


FILTERS = {
    "對照：任一天買（不管有沒有跌）": lambda d: pd.Series(True, index=d.index),
    "無濾網": lambda d: pd.Series(True, index=d.index),
    "大漲過的強勢股（跌前高點≥一年前×1.5）": lambda d: d["strong"],
    "年線仍上揚": lambda d: d["up200"],
    "強勢股＋止跌紅K（收紅且過昨高）": lambda d: d["strong"] & d["red_turn"],
}
DDS = [0.20, 0.25, 0.30, 0.35]
EXITS = [  # (停利%, 停損%, 最多天數)
    (3, 20, 60), (5, 20, 60), (10, 20, 60), (10, 20, 120), (3, 20, 20),
    (3, 15, 60), (10, 15, 120), (3, None, 120), (10, None, 120),
]


def simulate(d, sig, tgt, stop, hold, bc, sc, scale_in=False):
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    out, i, n = [], 260, len(d)
    while i < n - 2:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        p1 = O[e]
        cost, qty = p1 * (1 + bc), 1.0
        add_px = p1 * 0.9
        added = False
        sl = p1 * (1 - stop / 100) if stop else -np.inf
        res, xj, x = None, None, None
        last = min(e + hold - 1, n - 1)
        for j in range(e, last + 1):
            if scale_in and not added and L[j] <= add_px and j > e:
                fill = min(O[j], add_px)
                cost += fill * (1 + bc)
                qty += 1
                added = True
            avg = cost / qty
            tp = avg * (1 + tgt / 100) / (1 - sc)
            if L[j] <= sl:
                res, x, xj = "停損", (min(O[j], sl) if j > e else sl), j
                break
            if H[j] >= tp and not (j == e and False):
                res, x, xj = "停利", (max(O[j], tp) if j > e else tp), j
                break
        if res is None:
            if last >= n - 1 and last - e + 1 < hold:
                break  # 尚未結束的交易不計
            res, x, xj = "到期", C[last], last
        if scale_in:  # 以「預計兩份資金」為分母，沒補到的那一半當作閒置現金，才能和單筆公平比較
            r = (x * (1 - sc) * qty - cost) / (2 * p1 * (1 + bc)) * 100
        else:
            r = (x * (1 - sc) * qty) / cost * 100 - 100
        mae = L[e:xj + 1].min() / p1 * 100 - 100
        out.append((d.index[i], res, r, xj - e + 1, mae, qty))
        i = xj + 1
    return out


rows = []
case_rows = []
for code, df in data.items():
    d = prep(df)
    bc0, sc0 = E.cost_rates(code, E.DEFAULT_PARAMS)
    bc, sc = bc0 + SLIP, sc0 + SLIP
    for fname, f in FILTERS.items():
        base = f(d).fillna(False).to_numpy()
        for dd in DDS:
            hit = (d["dd"] <= -dd).to_numpy()
            first = hit & ~np.r_[False, hit[:-1]]  # 剛跌到門檻的那一天
            if fname.startswith("對照"):
                if dd != DDS[0]:
                    continue
                sig = base
            elif "止跌" in fname:  # 止跌紅K：跌到門檻後 10 天內第一根止跌紅K
                zone = pd.Series(hit).rolling(10, min_periods=1).max().to_numpy() > 0
                sig = base & zone
            else:
                sig = base & first
            for (tgt, stop, hold) in EXITS:
                for scale in (False, True):
                    if scale and (stop != 20 or tgt not in (3, 10)):
                        continue
                    for (dt, res, r, days, mae, qty) in simulate(d, sig, tgt, stop, hold, bc, sc, scale):
                        rows.append((code, fname, dd, tgt, stop, hold, scale, dt, res, r, days, mae))
T = pd.DataFrame(rows, columns=["代號", "濾網", "跌幅", "停利", "停損", "天數上限", "分批", "訊號日", "結果", "淨報酬%", "持有天數", "最大浮虧%"])
print(f"模擬 {len(T)} 筆，{time.time() - t0:.0f} 秒")


def st(g):
    r = g["淨報酬%"]
    return {"n": len(g), "win": (g["結果"] == "停利").mean() * 100, "pos": (r > 0).mean() * 100, "avg": r.mean(),
            "stop": (g["結果"] == "停損").mean() * 100, "days": g["持有天數"].mean(), "worst": r.min(),
            "mae50": g["最大浮虧%"].median(), "mae10": g["最大浮虧%"].quantile(0.1)}


def fs(s):
    return (f"{s['n']} 筆・成功率 {s['win']:.0f}%（淨賺 {s['pos']:.0f}%）・平均 {s['avg']:+.2f}%・停損 {s['stop']:.0f}%"
            f"・平均抱 {s['days']:.0f} 天・過程浮虧中位 {s['mae50']:.0f}%／最糟一成 {s['mae10']:.0f}%")


def ex_name(k):
    tgt, stop, hold, scale = k
    return f"停利{tgt}%／{'停損' + str(stop) + '%' if stop else '不停損'}／最多{hold}天{'／分批(再跌10%補)' if scale else ''}"


L_ = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・個股 {len(data)} 檔・還原權息・約 10 年",
      "強勢股大跌買進：距 120 日高跌 X% 的第一天隔天開盤買；扣手續費、證交稅、滑價。成功率＝先碰到停利。",
      "⚠ 股票池只含今天還在的股票（存活者偏差），抄底結果偏樂觀。", ""]
keys = ["濾網", "跌幅", "停利", "停損", "天數上限", "分批"]
summ = []
for k, g in T.groupby(keys, dropna=False, sort=False):
    a, b = g[g["訊號日"] < SPLIT], g[g["訊號日"] >= SPLIT]
    if len(a) < 30 or len(b) < 30:
        continue
    yr = g.groupby(g["訊號日"].dt.year)["淨報酬%"].mean()
    g22 = g[g["訊號日"].dt.year == 2022]
    summ.append({"k": k, "all": st(g), "a": st(a), "b": st(b), "yrs": (yr > 0).sum(), "ny": len(yr),
                 "dates": g["訊號日"].nunique(), "y22": st(g22) if len(g22) else None})

bench = {(x["k"][2], x["k"][3], x["k"][4], x["k"][5]): x for x in summ if x["k"][0].startswith("對照")}
ok = [x for x in summ if not x["k"][0].startswith("對照") and x["a"]["win"] >= 70 and x["b"]["win"] >= 70 and x["a"]["avg"] > 0 and x["b"]["avg"] > 0]
L_.append(f"★ 樣本內外成功率都 ≥70% 且平均都賺錢：{len(ok)} 組（依「兩段較差的平均」排序，列前 25）")
for x in sorted(ok, key=lambda x: -min(x["a"]["avg"], x["b"]["avg"]))[:25]:
    f, dd, tgt, stop, hold, sc = x["k"]
    L_ += [f"  【{f}・跌 {dd * 100:.0f}%】{ex_name((tgt, stop if stop == stop else None, hold, sc))}",
           f"    全期 {fs(x['all'])}・{x['ny']} 年中 {x['yrs']} 年為正・最差一筆 {x['all']['worst']:+.0f}%",
           f"    樣本內 成功率 {x['a']['win']:.0f}% 平均 {x['a']['avg']:+.2f}%（{x['a']['n']} 筆）｜樣本外 成功率 {x['b']['win']:.0f}% 平均 {x['b']['avg']:+.2f}%（{x['b']['n']} 筆）"]
    bm = bench.get((tgt, stop, hold, sc))
    if bm:
        L_.append(f"    對照（任一天買、同出場）：成功率 {bm['all']['win']:.0f}%・平均 {bm['all']['avg']:+.2f}%")
    if x["y22"]:
        L_.append(f"    2022 空頭年：{x['y22']['n']} 筆・成功率 {x['y22']['win']:.0f}%・平均 {x['y22']['avg']:+.2f}%・停損 {x['y22']['stop']:.0f}%")
L_.append("")
L_.append("— 對照表：無濾網 vs 強勢股，跌 30%，各種出場 —")
for x in summ:
    f, dd, tgt, stop, hold, sc = x["k"]
    if (dd == 0.30 and f in ("無濾網", "大漲過的強勢股（跌前高點≥一年前×1.5）")) or f.startswith("對照"):
        L_.append(f"  {f}｜{ex_name((tgt, stop if stop == stop else None, hold, sc))}：{fs(x['all'])}")
L_.append("")

# 個案：3653、2059、2409 近兩年在「強勢股・跌 30%・停利10%／停損20%／120天」下的訊號
L_.append("— 個案：強勢股・跌 25%／30%，停利 10%／停損 20%／最多 120 天 —")
for c in CASES:
    g = T[(T["代號"] == c) & (T["濾網"] == "大漲過的強勢股（跌前高點≥一年前×1.5）") & (T["停利"] == 10)
          & (T["停損"] == 20) & (T["天數上限"] == 120) & (~T["分批"]) & (T["跌幅"].isin([0.25, 0.30]))]
    for _, r in g.sort_values("訊號日").tail(8).iterrows():
        L_.append(f"  {c} 跌{r['跌幅'] * 100:.0f}% 訊號 {r['訊號日']:%Y-%m-%d}：{r['結果']} {r['淨報酬%']:+.1f}%・抱 {r['持有天數']} 天・過程最多浮虧 {r['最大浮虧%']:.0f}%")
    if c in data:
        d = prep(data[c])
        L_.append(f"  {c} 目前：收 {d['close'].iloc[-1]:.2f}、距 120 日高 {d['dd'].iloc[-1] * 100:+.0f}%、強勢股 {bool(d['strong'].iloc[-1])}")

txt = "\n".join(L_)
(OUT / "dip.txt").write_text(txt, encoding="utf-8")
print(txt)
print(f"總耗時 {time.time() - t0:.0f} 秒")
