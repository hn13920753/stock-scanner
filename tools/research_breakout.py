"""
研究：能不能「看出起漲點」，做到淨賺 3～5%、成功率 70～80%？（2026-10-21 第二輪）
只做統計，不改規則。輸出 research/breakout.txt。

第一輪（research_winrate.py）測的是回檔／超跌買點；這一輪專測「起漲」類方法：
  Minervini 趨勢樣板＋VCP 收斂突破、O'Neil 底部突破、Darvas 箱型突破、Wyckoff 彈簧（假跌破收回）、
  布林擠壓突破、量縮後爆量長紅、NR7 窄幅內包突破、強勢相對強度股突破 20 日高、跳空起漲缺口，
  以及一個機器學習模型（梯度提升樹）。

防止自欺：
- 用 Yahoo「還原權息、分割」的價格（auto_adjust），避免除權息、分割造成假的大跌。
- 訊號日收盤後判斷、隔天開盤買；同一檔同時只持有一筆。
- 樣本內 2016～2022（挑）、樣本外 2023～（驗）；機器學習 2016～2021 訓練、2022 定門檻、2023～ 驗。
- 成本：手續費 0.1425%×2＋證交稅 0.3%（ETF 0.1%）＋滑價每邊 0.1%。
- 另外列「起漲準確度」：15 日內最高曾漲到 +3%／+5% 的比例（不管中途有沒有先跌），
  這是「有沒有起漲」本身；但能不能賺到要看先碰停利還是停損。
股票池：產業清單＋台灣 50＋證交所當日成交值前 300 檔普通股（存活者偏差：今天還在的股票）。
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

OUT = Path(os.environ.get("BO_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
SPLIT = pd.Timestamp("2023-01-01")
SLIP = 0.001
HOLD = 15
FOCUS = "2409"
FAKE = bool(os.environ.get("BO_FAKE"))
t0 = time.time()

EXTRA = ["2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", "2303", "2412", "2886", "2884",
         "1301", "1303", "2002", "2885", "2892", "2880", "5880", "2883", "2887", "3711", "2357", "2395",
         "3045", "4904", "2912", "1216", "2207", "2603", "2609", "2615", "1101", "1326", "6505", "2327",
         "3008", "2379", "3034", "2301", "5871", "2890", "9910", "2409", "3481", "2344", "2408", "2618"]


def universe():
    u = {}
    for sec in SECTORS:
        for s in stocks_of(sec):
            u.setdefault(s["code"], s["market"])
    for c in EXTRA:
        u.setdefault(c, "TW")
    if FAKE:
        return u
    try:  # 證交所當日全部上市股票成交資訊 → 成交值前 300 的普通股
        import requests
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30)
        df = pd.DataFrame(r.json())
        df = df[df["Code"].str.fullmatch(r"[1-9]\d{3}")]
        df["tv"] = pd.to_numeric(df["TradeValue"], errors="coerce")
        for c in df.nlargest(300, "tv")["Code"]:
            u.setdefault(c, "TW")
        print(f"證交所清單：{len(df)} 檔普通股，取成交值前 300")
    except Exception as ex:  # noqa: BLE001
        print("證交所清單失敗：", ex)
    return u


def fetch(u):
    import yfinance as yf
    out = {}

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

    out.update(grab([f"{c}.{m}" for c, m in u.items()]))
    miss = [c for c in u if c not in out]
    out.update(grab([f"{c}.{'TWO' if u[c] == 'TW' else 'TW'}" for c in miss]))
    return out


def fake_data(u):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-10-01", "2026-10-20")
    out = {}
    for code in list(u)[:25] + [FOCUS]:
        r = rng.normal(0.0004, 0.02, len(idx))
        c = 50 * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, 0.006, len(idx)))
        out[code] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, .008, len(idx)))),
                                  "low": np.minimum(o, c) * (1 - abs(rng.normal(0, .008, len(idx)))),
                                  "close": c, "volume": rng.integers(500, 9000, len(idx)).astype(float)}, index=idx)
    mk = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx)))), index=idx)
    return out, mk


U = universe()
if FAKE:
    data, mkt = fake_data(U)
else:
    data = fetch(U)
    mkt = E.load_index(11)
n_etf = sum(E.is_etf(k) for k in data)
print(f"有資料 {len(data)} 檔（個股 {len(data) - n_etf}、ETF {n_etf}），抓資料 {time.time() - t0:.0f} 秒")

M = pd.DataFrame({"c": mkt})
M["ma60"] = M["c"].rolling(60).mean()
M["r20"] = M["c"].pct_change(20) * 100
M["r60"] = M["c"].pct_change(60) * 100


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prep(df):
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    for n in (5, 10, 20, 50, 60, 150, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["atrp"] = d["atr"] / c * 100
    d["vol50"] = v.shift(1).rolling(50).mean()
    d["vr50"] = v / d["vol50"]
    d["vr5"] = v / v.shift(1).rolling(5).mean()
    d["chg"] = c.pct_change() * 100
    d["hh252"] = h.shift(1).rolling(252).max()
    d["ll252"] = l.shift(1).rolling(252).min()
    d["hh20"] = h.shift(1).rolling(20).max()
    d["hh10"] = h.shift(1).rolling(10).max()
    d["hh30"] = h.shift(1).rolling(30).max()
    d["ll30"] = l.shift(1).rolling(30).min()
    d["rng10"] = (h.rolling(10).max() - l.rolling(10).min()) / c
    d["rng60"] = (h.rolling(60).max() - l.rolling(60).min()) / c
    sd = c.rolling(20).std()
    d["bbu"] = d["ma20"] + 2 * sd
    d["bbw"] = 4 * sd / d["ma20"]
    d["bbw_q"] = d["bbw"].rolling(120).quantile(0.1)
    d["ibs"] = (c - l) / (h - l).replace(0, np.nan)
    d["rsi14"] = rsi(c, 14)
    d["rsi2"] = rsi(c, 2)
    d["r5"] = c.pct_change(5) * 100
    d["r20"] = c.pct_change(20) * 100
    d["r60"] = c.pct_change(60) * 100
    d["red"] = (c > o) & (d["chg"] > 0)
    rngd = h - l
    d["nr7"] = rngd <= rngd.rolling(7).min()
    d["inside"] = (h <= h.shift(1)) & (l >= l.shift(1))
    return d


t1 = time.time()
P = {k: prep(v) for k, v in data.items()}
# 相對強度：個股 60 日報酬在當天全部股票中的百分位
r60 = pd.DataFrame({k: v["r60"] for k, v in P.items() if not E.is_etf(k)})
rs_rank = r60.rank(axis=1, pct=True)
print(f"指標計算 {time.time() - t1:.0f} 秒")


def setups(code, d):
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    m = M.reindex(d.index).ffill()
    risk_on = m["c"] > m["ma60"]
    rs = rs_rank[code].reindex(d.index) if code in rs_rank else pd.Series(np.nan, index=d.index)
    tmpl = ((c > d["ma50"]) & (d["ma50"] > d["ma150"]) & (d["ma150"] > d["ma200"])
            & (d["ma200"] > d["ma200"].shift(20)) & (c >= 0.75 * d["hh252"]) & (c >= 1.3 * d["ll252"]))
    vcp = tmpl & ((d["rng10"] / d["rng60"]).shift(1) < 0.4) & (c > d["hh10"]) & (d["vr50"] >= 1.5) & d["red"]
    depth = (d["hh30"] - d["ll30"]) / d["hh30"]
    oneil = depth.between(0.08, 0.25) & (c > d["hh30"]) & (d["vr50"] >= 1.4) & (c > d["ma50"]) & (d["ibs"] > 0.75)
    no_new = h.shift(1).rolling(4).max() < d["hh10"]
    darvas = (d["hh10"] >= 0.95 * d["hh252"]) & no_new & (c > d["hh10"]) & (d["vr50"] >= 1.5) & d["red"]
    spring_day = l < l.shift(1).rolling(60).min()
    spring = (spring_day.shift(1).rolling(3).max().fillna(0) > 0) & (c > l.shift(4).rolling(60).min()) \
        & d["red"] & (c > h.shift(1)) & (d["vr5"] >= 1.2)
    squeeze = (d["bbw"].shift(1) <= d["bbw_q"].shift(1)) & (c > d["bbu"]) & (d["vr50"] >= 1.5)
    dry = (v.shift(1).rolling(5).mean() < 0.6 * d["vol50"]) & (d["vr50"] >= 2) & d["red"] & (d["chg"] >= 3) & (c > d["ma20"])
    nr7 = d["nr7"].shift(1).fillna(False).astype(bool) & d["inside"].shift(1).fillna(False).astype(bool) \
        & (c > h.shift(1)) & d["red"] & (c > d["ma50"])
    rslead = (rs >= 0.8) & (c > d["hh20"]) & (d["vr50"] >= 1.5) & d["red"]
    gap = (o > h.shift(1) * 1.01) & (c >= o) & (d["vr50"] >= 2) & (c > d["ma20"])
    s = {
        "對照：任一天": pd.Series(True, index=d.index),
        "Minervini 趨勢樣板＋VCP 收斂突破": vcp,
        "O'Neil 底部突破（30日區間深8～25%）": oneil,
        "Darvas 箱型突破（近52週高）": darvas,
        "Wyckoff 彈簧（跌破60日低後收回）": spring,
        "布林擠壓突破": squeeze,
        "量縮後爆量長紅（量≥2倍、漲≥3%）": dry,
        "NR7 窄幅內包後突破": nr7,
        "強勢股（RS前20%）突破20日高": rslead,
        "跳空起漲缺口（量≥2倍）": gap,
        "強勢組合：趨勢樣板＋RS前20%＋底部突破": tmpl & (rs >= 0.8) & oneil,
    }
    for name in list(s):
        if not name.startswith("對照"):
            s[name + "｜大盤多頭"] = s[name] & risk_on
    return {k: v.fillna(False).astype(bool).to_numpy() for k, v in s.items()}


EXITS = {
    "停利3%／停損3%": (3, "pct", 3),
    "停利3%／停損5%": (3, "pct", 5),
    "停利3%／停損7%": (3, "pct", 7),
    "停利3%／不停損（15日到期）": (3, "none", None),
    "停利3%／結構停損（訊號日低點）": (3, "low", None),
    "停利5%／停損5%": (5, "pct", 5),
    "停利5%／停損8%": (5, "pct", 8),
    "停利5%／結構停損（訊號日低點）": (5, "low", None),
}


def sim_one(O, H, L, C, i, ex, bc, sc):
    tgt, kind, par = ex
    e = i + 1
    entry = O[e]
    tp = entry * (1 + bc) * (1 + tgt / 100) / (1 - sc)
    sl = entry * (1 - par / 100) if kind == "pct" else (min(L[i], entry * 0.97) if kind == "low" else -np.inf)
    for j in range(e, e + HOLD):
        if L[j] <= sl:
            x = min(O[j], sl) if j > e else sl
            return "停損", (x * (1 - sc)) / (entry * (1 + bc)) * 100 - 100, j
        if H[j] >= tp:
            x = max(O[j], tp) if j > e else tp
            return "停利", (x * (1 - sc)) / (entry * (1 + bc)) * 100 - 100, j
    j = e + HOLD - 1
    return "期滿", (C[j] * (1 - sc)) / (entry * (1 + bc)) * 100 - 100, j


rows, mfe_rows = [], []
t2 = time.time()
for code, d in P.items():
    sg = setups(code, d)
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    bc0, sc0 = E.cost_rates(code, E.DEFAULT_PARAMS)
    bc, sc = bc0 + SLIP, sc0 + SLIP
    typ = "ETF" if E.is_etf(code) else "個股"
    n = len(d)
    for sname, sig in sg.items():
        exits = EXITS if not sname.startswith("對照") else {k: EXITS[k] for k in ("停利3%／停損3%", "停利3%／停損7%", "停利3%／不停損（15日到期）", "停利5%／停損5%")}
        for ename, ex in exits.items():
            i = 260
            while i < n - 1 - HOLD:
                if not sig[i]:
                    i += 1
                    continue
                res, r, xj = sim_one(O, H, L, C, i, ex, bc, sc)
                rows.append((code, typ, sname, ename, d.index[i], res, r, xj - i))
                i = xj + 1
        # 起漲準確度（每 15 日最多算一次）
        i = 260
        while i < n - 1 - HOLD:
            if not sig[i]:
                i += 1
                continue
            e = i + 1
            entry = O[e] * (1 + bc) / (1 - sc)  # 含成本的損益兩平價
            mx = H[e:e + HOLD].max() / entry * 100 - 100
            mn = L[e:e + HOLD].min() / O[e] * 100 - 100
            mfe_rows.append((typ, sname, d.index[i], mx, mn))
            i += HOLD
T = pd.DataFrame(rows, columns=["代號", "類型", "進場", "出場", "訊號日", "結果", "淨報酬%", "天數"])
MF = pd.DataFrame(mfe_rows, columns=["類型", "進場", "訊號日", "最高淨漲%", "最大跌%"])
print(f"模擬 {len(T)} 筆，{time.time() - t2:.0f} 秒")


def st(g):
    if len(g) == 0:
        return None
    r = g["淨報酬%"]
    return {"n": len(g), "win": (g["結果"] == "停利").mean() * 100, "avg": r.mean(),
            "worst": r.min(), "g": r[r > 0].mean() if (r > 0).any() else 0, "b": r[r <= 0].mean() if (r <= 0).any() else 0}


def fs(s):
    return "0 筆" if s is None else (f"{s['n']} 筆・成功率 {s['win']:.0f}%・平均 {s['avg']:+.2f}%"
                                     f"（賺 {s['g']:+.1f}／賠 {s['b']:+.1f}）・最差 {s['worst']:+.1f}%")


N_COMBOS = T.groupby(["類型", "進場", "出場"]).ngroups
L_ = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・{len(data)} 檔（個股 {len(data) - n_etf}、ETF {n_etf}）・還原權息價・約 10 年",
      "隔天開盤買、同檔不重疊、15 個交易日；扣手續費、證交稅、滑價每邊 0.1%。成功率＝先碰到停利。",
      "樣本內 2016～2022、樣本外 2023～。達標＝兩段成功率都 ≥70%、平均都 >0、各 ≥30 筆。",
      f"注意：共測 {N_COMBOS} 種組合，組合越多越容易「碰巧」過關（用隨機亂數資料測試時也會有 1～2 組過關），要看各年是否穩定。", ""]

for typ in ("個股", "ETF"):
    L_.append(f"==================== {typ}：起漲準確度（15 日內最高曾到 / 最大曾跌）====================")
    for sname, g in MF[MF["類型"] == typ].groupby("進場", sort=False):
        if len(g) < 30:
            continue
        L_.append(f"  {sname}：{len(g)} 次・曾淨漲≥3% {(g['最高淨漲%'] >= 3).mean() * 100:.0f}%・≥5% {(g['最高淨漲%'] >= 5).mean() * 100:.0f}%"
                  f"・同期曾跌≥3% {(g['最大跌%'] <= -3).mean() * 100:.0f}%・≥5% {(g['最大跌%'] <= -5).mean() * 100:.0f}%")
    L_.append("")

for typ in ("個股", "ETF"):
    summ = []
    for (sname, ename), g in T[T["類型"] == typ].groupby(["進場", "出場"], sort=False):
        a, b = st(g[g["訊號日"] < SPLIT]), st(g[g["訊號日"] >= SPLIT])
        if a is None or b is None:
            continue
        ok = all(x["n"] >= 30 and x["win"] >= 70 and x["avg"] > 0 for x in (a, b))
        summ.append((sname, ename, a, b, ok))
    L_.append(f"==================== {typ}：交易結果 ====================")
    good = [x for x in summ if x[4]]
    L_.append(f"★ 兩段都達標：{len(good)} 組")
    for s_, e_, a, b, _ in good:
        L_ += [f"  {s_} × {e_}", f"    樣本內 {fs(a)}", f"    樣本外 {fs(b)}"]
        g = T[(T["類型"] == typ) & (T["進場"] == s_) & (T["出場"] == e_)]
        yr = g.groupby(g["訊號日"].dt.year)["淨報酬%"].agg(["count", "mean"])
        L_.append("    各年：" + "、".join(f"{y} {int(r['count'])}筆{r['mean']:+.1f}%" for y, r in yr.iterrows())
                  + f"｜不同訊號日 {g['訊號日'].nunique()} 天")
    near = sorted([x for x in summ if x[2]["n"] >= 30 and x[3]["n"] >= 30 and not x[4]],
                  key=lambda x: -min(x[2]["win"], x[3]["win"]))[:8]
    L_.append("◇ 成功率最高的 8 組（未達標）：")
    for s_, e_, a, b, _ in near:
        L_ += [f"  {s_} × {e_}", f"    樣本內 {fs(a)}", f"    樣本外 {fs(b)}"]
    pos = sorted([x for x in summ if x[2]["n"] >= 30 and x[3]["n"] >= 30 and x[2]["avg"] > 0 and x[3]["avg"] > 0],
                 key=lambda x: -min(x[2]["avg"], x[3]["avg"]))[:12]
    L_.append(f"◎ 兩段平均都賺錢：{len(pos)} 組（列前 12）")
    for s_, e_, a, b, _ in pos:
        L_ += [f"  {s_} × {e_}", f"    樣本內 {fs(a)}", f"    樣本外 {fs(b)}"]
    L_.append("— 停利3%／停損3% 各方法（全期間）—")
    for s_, e_, a, b, _ in summ:
        if e_ == "停利3%／停損3%":
            L_.append(f"  {s_}：{fs(st(T[(T['類型'] == typ) & (T['進場'] == s_) & (T['出場'] == e_)]))}")
    L_.append("")

# ------------------------------------------------------------------ 機器學習
L_.append("==================== 機器學習（梯度提升樹，只用個股）====================")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    FEATS = ["r5", "r20", "r60", "atrp", "vr5", "vr50", "rsi14", "rsi2", "ibs", "chg", "bbw", "rng10",
             "d_ma20", "d_ma60", "d_ma200", "d_hh252", "d_ll252", "rs", "m_r20", "m_on"]
    LAB = {"停利3%／停損5%": (3, "pct", 5), "停利3%／停損3%": (3, "pct", 3), "停利5%／停損5%": (5, "pct", 5)}
    frames = []
    for code, d in P.items():
        if E.is_etf(code):
            continue
        m = M.reindex(d.index).ffill()
        f = pd.DataFrame(index=d.index)
        for k in ["r5", "r20", "r60", "atrp", "vr5", "vr50", "rsi14", "rsi2", "ibs", "chg", "bbw", "rng10"]:
            f[k] = d[k]
        for k in ("ma20", "ma60", "ma200"):
            f["d_" + k] = d["close"] / d[k] - 1
        f["d_hh252"] = d["close"] / d["hh252"] - 1
        f["d_ll252"] = d["close"] / d["ll252"] - 1
        f["rs"] = rs_rank[code].reindex(d.index) if code in rs_rank else np.nan
        f["m_r20"] = m["r20"]
        f["m_on"] = (m["c"] > m["ma60"]).astype(float)
        O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        bc0, sc0 = E.cost_rates(code, E.DEFAULT_PARAMS)
        n = len(d)
        for lname, ex in LAB.items():
            y, rr = np.full(n, np.nan), np.full(n, np.nan)
            for i in range(260, n - 1 - HOLD):
                res, r, _ = sim_one(O, H, L, C, i, ex, bc0 + SLIP, sc0 + SLIP)
                y[i], rr[i] = res == "停利", r
            f["y_" + lname], f["r_" + lname] = y, rr
        f["code"] = code
        frames.append(f.iloc[260:n - 1 - HOLD])
    D = pd.concat(frames).dropna(subset=FEATS[:-3])
    D["date"] = D.index
    print(f"機器學習資料 {len(D)} 列")
    tr, va, te = D[D.date < "2022-01-01"], D[(D.date >= "2022-01-01") & (D.date < "2023-01-01")], D[D.date >= SPLIT]
    for lname in LAB:
        yk, rk = "y_" + lname, "r_" + lname
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=300, l2_regularization=1.0, random_state=0)
        clf.fit(tr[FEATS], tr[yk].astype(int))
        pv, pt = clf.predict_proba(va[FEATS])[:, 1], clf.predict_proba(te[FEATS])[:, 1]
        L_.append(f"【{lname}】訓練 {len(tr)} 列（基準成功率 {tr[yk].mean() * 100:.0f}%），驗證 2023～ {len(te)} 列（基準 {te[yk].mean() * 100:.0f}%）")
        for q in (0.5, 0.9, 0.95, 0.99, 0.999):
            thr = np.quantile(pv, q)  # 門檻用 2022 決定
            sel = te[pt >= thr]
            if len(sel) == 0:
                continue
            L_.append(f"  模型分數前 {(1 - q) * 100:.1f}%（門檻 {thr:.2f}，2022 定）：2023～ {len(sel)} 天次・成功率 {sel[yk].mean() * 100:.0f}%"
                      f"・平均 {sel[rk].mean():+.2f}%・{sel['code'].nunique()} 檔・{sel['date'].nunique()} 個交易日")
        # 非重疊實際交易：每檔分數超過前 1% 門檻才買，持有期間不再買
        thr = np.quantile(pv, 0.99)
        te2 = te.assign(p=pt)
        tr_rows = []
        for code, g in te2.groupby("code"):
            g = g.sort_index()
            busy = None
            for dt, row in g.iterrows():
                if busy is not None and dt <= busy:
                    continue
                if row["p"] >= thr:
                    tr_rows.append(row[rk])
                    pos = g.index.get_loc(dt)
                    busy = g.index[min(pos + HOLD, len(g) - 1)]
        if tr_rows:
            a = np.array(tr_rows)
            L_.append(f"  實際可交易（前 1%、同檔不重疊）：{len(a)} 筆・平均 {a.mean():+.2f}%・淨賺比例 {(a > 0).mean() * 100:.0f}%")
except Exception:  # noqa: BLE001
    import traceback
    L_.append("機器學習失敗：" + traceback.format_exc()[-800:])
L_.append("")

# ------------------------------------------------------------------ 2409 今天
if FOCUS in P:
    d = P[FOCUS]
    sg = setups(FOCUS, d)
    hit = [k for k, v in sg.items() if v[-1] and not k.startswith("對照")]
    L_.append(f"2409 友達：資料至 {d.index[-1]:%Y-%m-%d}，收 {d['close'].iloc[-1]:.2f}，ATR% {d['atrp'].iloc[-1]:.1f}，"
              f"最後一根符合：{'、'.join(hit) or '無'}")
    g = T[(T["代號"] == FOCUS) & (T["出場"] == "停利3%／停損3%")]
    for sname, gg in g.groupby("進場", sort=False):
        L_.append(f"  2409 歷史 {sname}：{fs(st(gg))}")

txt = "\n".join(L_)
(OUT / "breakout.txt").write_text(txt, encoding="utf-8")
print(txt)
print(f"總耗時 {time.time() - t0:.0f} 秒")
