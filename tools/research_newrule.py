"""
新規則研究（2026-10-27）：回應使用者三項要求
  ① 高波動個股不再一律「不適合」→ 回檔布局買法（離近 60 日高點跌 X%）不看股性，對所有個股與 ETF 適用。
  ② 缺點 1 訊號少、缺點 2 帳戶報酬輸 0050 → 比較「區間內都可買」與「第一次跌到才買」、閒置資金放 0050。
  ③ 舊的「淨利 3%、最多 15 天」換成數據支持的停利方式 → 用引擎現有五種買點，比較 3%/15 天、5%/60 天、10%/60 天。
只做統計，不改規則。輸出 research/newrule.txt、research/newrule_trades.csv。

方法
- 股票池：產業清單＋台灣 50 等權值股＋證交所成交值前 300；Yahoo 還原權息價；約 10 年。
- 回檔布局：收盤距近 60 日最高點跌 ≥ X%（X＝20、25、30、35、40）。
    first＝第一次跌破 X% 的那天才買；zone＝只要還在 X% 以下、手上沒有這檔就可買（訊號較多）。
  隔天開盤買；淨賺 5% 或 10% 賣；買價 −20% 停損；最多 60 個交易日，到期收盤賣。
- 驗證：2016～2022 樣本內、2023～ 樣本外；分年度。
- 帳戶：100 萬、每筆 20 萬（最多 5 筆）、同一天多檔訊號時隨機挑；三種閒置資金處理：
    現金不動／閒置全放 0050（要買股時賣 0050 換現金）／0050 買進放著（對照）。
- 成本：手續費 0.1425%×2、證交稅（股票 0.3%、ETF 0.1%）、滑價每邊 0.1%。
- 偏差：股票池為今天仍存在的股票（存活者偏差），回檔買進會被高估；2016～2026 為台股大多頭。
"""
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

OUT = Path(os.environ.get("NR_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
FAKE = bool(os.environ.get("NR_FAKE"))
SPLIT = pd.Timestamp("2023-01-01")
SLIP = 0.001
STOP, HOLD = 0.20, 60
DROPS = (0.20, 0.25, 0.30, 0.35, 0.40)
t0 = time.time()

EXTRA = ["2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", "2303", "2412", "2886", "2884",
         "1301", "1303", "2002", "2885", "2892", "2880", "5880", "2883", "2887", "3711", "2357", "2395",
         "3045", "4904", "2912", "1216", "2207", "2603", "2609", "2615", "1101", "1326", "6505", "2327",
         "3008", "2379", "3034", "2301", "5871", "2890", "9910", "2409", "3481", "2344", "2408", "2618",
         "3653", "6187", "3324", "0050"]


def universe():
    u = {}
    for sec in SECTORS:
        for s in stocks_of(sec):
            u.setdefault(s["code"], s["market"])
    for c in EXTRA:
        u.setdefault(c, "TWO" if c in ("6187", "3324") else "TW")
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


def fetch(u, adjust=True):
    import yfinance as yf

    def grab(tks):
        got = {}
        for k in range(0, len(tks), 80):
            part = tks[k:k + 80]
            try:
                d = yf.download(part, period="11y", interval="1d", group_by="ticker", auto_adjust=adjust,
                                actions=False, threads=True, progress=False)
            except Exception:  # noqa: BLE001
                continue
            for tk in part:
                try:
                    sub = d[tk] if isinstance(d.columns, pd.MultiIndex) else d
                    sub = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                    sub = sub[(sub["volume"] > 0) & (sub["high"] >= sub["low"]) & (sub["low"] > 0)]
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


def fake_data(codes):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-10-01", "2026-10-26")
    out = {}
    for code in codes:
        c = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.024, len(idx))))
        o = c * (1 + rng.normal(0, 0.007, len(idx)))
        out[code] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, .008, len(idx)))),
                                  "low": np.minimum(o, c) * (1 - abs(rng.normal(0, .008, len(idx)))),
                                  "close": c, "volume": rng.integers(500, 9000, len(idx)).astype(float)}, index=idx)
    return out


U = universe()
if FAKE:
    keep = list(U)[:int(os.environ.get("NR_N", 30))] + ["0050"]
    U = {k: U.get(k, "TW") for k in keep}
    P = fake_data(U)
else:
    P = fetch(U)
n_etf = sum(E.is_etf(k) for k in P)
print(f"有資料 {len(P)} 檔（個股 {len(P) - n_etf}、ETF {n_etf}），{time.time() - t0:.0f} 秒", flush=True)


# ------------------------------------------------------------------ 一、回檔布局逐筆
def dip_trades(code, df, drop, tgt, mode):
    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    hh = df["high"].rolling(60).max().to_numpy()
    dd = C / hh - 1
    buy, sell = E.cost_rates(code, E.DEFAULT_PARAMS)
    bc, sc = buy + SLIP, sell + SLIP
    out, i, n = [], 60, len(df)
    while i < n - 1:
        hit = dd[i] <= -drop and (mode == "zone" or dd[i - 1] > -drop)
        if not hit:
            i += 1
            continue
        e = i + 1
        px = O[e]
        tp = px * (1 + bc) * (1 + tgt) / (1 - sc)
        sl = px * (1 - STOP)
        last = min(e + HOLD - 1, n - 1)
        res = x = xj = None
        for j in range(e, last + 1):
            if L[j] <= sl:
                res, x, xj = "停損", (min(O[j], sl) if j > e else sl), j
                break
            if H[j] >= tp:
                res, x, xj = "停利", (max(O[j], tp) if j > e else tp), j
                break
        if res is None:
            if last - e + 1 < HOLD:
                break  # 尚未結束
            res, x, xj = "到期", C[last], last
        r = (x * (1 - sc)) / (px * (1 + bc)) - 1
        out.append({"代號": code, "類": "ETF" if E.is_etf(code) else "個股", "跌幅": drop, "停利": tgt, "模式": mode,
                    "訊號日": df.index[i], "進場日": df.index[e], "出場日": df.index[xj], "結果": res,
                    "報酬%": r * 100, "天數": xj - e + 1, "浮虧%": (L[e:xj + 1].min() / px - 1) * 100,
                    "ATR%": np.nan})
        i = xj + 1
    return out


def atr_pct(df):
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    return (tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100)


rows = []
ATRS = {c: atr_pct(df) for c, df in P.items()}
for c, df in P.items():
    for drop in DROPS:
        for tgt in (0.05, 0.10):
            for mode in ("first", "zone"):
                for t in dip_trades(c, df, drop, tgt, mode):
                    t["ATR%"] = float(ATRS[c].get(t["訊號日"], np.nan))
                    rows.append(t)
T = pd.DataFrame(rows)
print(f"回檔布局逐筆 {len(T)} 筆，{time.time() - t0:.0f} 秒", flush=True)


def st(d):
    if d is None or not len(d):
        return None
    return {"n": len(d), "win": (d["結果"] == "停利").mean() * 100, "avg": d["報酬%"].mean(),
            "sl": (d["結果"] == "停損").mean() * 100, "days": d["天數"].mean(), "worst": d["報酬%"].min(),
            "mae15": (d["浮虧%"] <= -15).mean() * 100}


def fmt(s):
    if not s:
        return "0 筆"
    return (f"{s['n']:>5} 筆・成功 {s['win']:>3.0f}%・停損 {s['sl']:>3.0f}%・每筆 {s['avg']:+5.2f}%・抱 {s['days']:>4.1f} 天"
            f"・最差 {s['worst']:+.0f}%・曾浮虧≥15% {s['mae15']:.0f}%")


L_ = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・股票池 {len(P)} 檔（個股 {len(P) - n_etf}、ETF {n_etf}）・Yahoo 還原權息價",
      "回檔布局：距近 60 日高點跌 ≥X%、隔天開盤買、淨賺 5%／10% 賣、−20% 停損、最多 60 天；扣手續費、證交稅、滑價。",
      "樣本內 2016～2022、樣本外 2023～。存活者偏差：股票池是今天還在的股票，結果偏樂觀。", ""]

L_.append("==================== 一、回檔布局：跌幅 × 停利 × 模式（全部／樣本內／樣本外）====================")
for typ in ("個股", "ETF"):
    for tgt in (0.10, 0.05):
        for mode in ("first", "zone"):
            L_.append(f"--- {typ}・停利 {tgt * 100:.0f}%・{'第一次跌到才買' if mode == 'first' else '區間內都可買'} ---")
            for drop in DROPS:
                d = T[(T["類"] == typ) & (T["停利"] == tgt) & (T["模式"] == mode) & (T["跌幅"] == drop)]
                L_.append(f"  跌 {drop * 100:.0f}%｜全部 {fmt(st(d))}")
                L_.append(f"        ｜樣本內 {fmt(st(d[d['進場日'] < SPLIT]))}")
                L_.append(f"        ｜樣本外 {fmt(st(d[d['進場日'] >= SPLIT]))}")
L_.append("")

# 高波動個股（原本判「不適合」）
L_.append("==================== 二、原本「不適合本策略」的高波動個股（訊號日 ATR > 4.5%）====================")
for drop in DROPS:
    d = T[(T["類"] == "個股") & (T["停利"] == 0.10) & (T["模式"] == "first") & (T["跌幅"] == drop)]
    hi, mid, lo = d[d["ATR%"] > 4.5], d[(d["ATR%"] <= 4.5) & (d["ATR%"] >= 1.2)], d[d["ATR%"] < 1.2]
    L_.append(f"  跌 {drop * 100:.0f}%｜ATR>4.5% {fmt(st(hi))}")
    L_.append(f"        ｜1.2～4.5% {fmt(st(mid))}")
    L_.append(f"        ｜樣本外 ATR>4.5% {fmt(st(hi[hi['進場日'] >= SPLIT]))}")
L_.append("")

# 分年
L_.append("==================== 三、分年（停利 10%・第一次跌到）====================")
for typ, drop in (("個股", 0.25), ("個股", 0.30), ("個股", 0.40), ("ETF", 0.20), ("ETF", 0.25)):
    d = T[(T["類"] == typ) & (T["停利"] == 0.10) & (T["模式"] == "first") & (T["跌幅"] == drop)]
    if len(d):
        g = d.groupby(d["進場日"].dt.year)["報酬%"].agg(["count", "mean"])
        L_.append(f"  {typ} 跌 {drop * 100:.0f}%：" + "  ".join(f"{y} {int(r['count'])}筆{r['mean']:+.1f}%" for y, r in g.iterrows()))
L_.append("")

# 每月平均訊號數（缺點 1）
L_.append("==================== 四、訊號多寡（缺點 1）：每月平均有幾檔出現訊號、有訊號的月份比例 ====================")
months = pd.period_range("2016-01", "2026-09", freq="M")
for typ in ("個股", "ETF"):
    for drop in (0.25, 0.30, 0.40) if typ == "個股" else (0.15, 0.20, 0.25):
        for mode in ("first", "zone"):
            d = T[(T["類"] == typ) & (T["停利"] == 0.10) & (T["模式"] == mode) & (T["跌幅"] == drop)]
            if not len(d):
                continue
            cnt = d.groupby(d["訊號日"].dt.to_period("M")).size().reindex(months, fill_value=0)
            L_.append(f"  {typ} 跌 {drop * 100:.0f}%・{mode:<5}：每月平均 {cnt.mean():.1f} 筆・有訊號的月份 {(cnt > 0).mean() * 100:.0f}%"
                      f"・最多一個月 {cnt.max()} 筆")
L_.append("（ETF 跌 15% 未在逐筆跑，若顯示空白屬正常）")
L_.append("")


# ------------------------------------------------------------------ 五、帳戶模擬（缺點 2）
def account(sig_sel, idle, capital=1_000_000, per=200_000, seed=0):
    """sig_sel：DataFrame 逐筆（已含進出場日與報酬）。用逐筆報酬近似逐日淨值：持有期間以收盤價計值。"""
    c50 = P["0050"]["close"]
    cal = c50.index[c50.index >= pd.Timestamp("2016-06-01")]
    rng = np.random.default_rng(seed)
    by_day = {}
    for r in sig_sel.itertuples():
        by_day.setdefault(r.進場日, []).append(r)
    cash, units50 = float(capital), 0.0
    if idle == "0050":
        px0 = c50.loc[cal[0]]
        units50 = cash * (1 - 0.001425 - SLIP) / px0
        cash = 0.0
    open_, eq, n_tr = [], [], 0
    held = set()
    for day in cal:
        p50 = c50.loc[day]
        # 出場
        for pos in list(open_):
            if day >= pos["exit"]:
                proceeds = pos["amt"] * (1 + pos["ret"] / 100)
                open_.remove(pos)
                held.discard(pos["code"])
                if idle == "0050":
                    units50 += proceeds * (1 - 0.001425 - SLIP) / p50
                else:
                    cash += proceeds
        # 進場
        cands = [r for r in by_day.get(day, []) if r.代號 not in held]
        rng.shuffle(cands)
        for r in cands:
            if len(open_) >= capital // per:
                break
            if idle == "0050":
                need = per / (1 - 0.001425 - 0.001 - SLIP)
                if units50 * p50 < need:
                    break
                units50 -= need / p50
            else:
                if cash < per:
                    break
                cash -= per
            open_.append({"code": r.代號, "amt": per, "ret": r.報酬, "exit": r.出場日,
                          "entry": day, "c": P[r.代號]["close"], "e0": P[r.代號]["close"].get(day, np.nan)})
            held.add(r.代號)
            n_tr += 1
        mark = 0.0
        for pos in open_:
            cc = pos["c"]
            cur = cc.get(day, np.nan)
            e0 = cc.asof(pos["entry"]) if np.isnan(pos["e0"]) else pos["e0"]
            mark += pos["amt"] * (cur / e0) if np.isfinite(cur) and e0 > 0 else pos["amt"]
        eq.append(cash + units50 * p50 + mark)
    eq = pd.Series(eq, index=cal)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    yr = eq.groupby(eq.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    return cagr * 100, mdd * 100, n_tr, yr.min() * 100, eq


L_.append("==================== 五、100 萬帳戶模擬（缺點 2）：每筆 20 萬、最多同時 5 筆 ====================")
try:
    T = T.rename(columns={"報酬%": "報酬"})
    c50 = P["0050"]["close"]
    c50 = c50[c50.index >= pd.Timestamp("2016-06-01")]
    y = (c50.index[-1] - c50.index[0]).days / 365.25
    L_.append(f"  對照 0050 買進放著：年化 {((c50.iloc[-1] / c50.iloc[0]) ** (1 / y) - 1) * 100:+.1f}%・最大回撤 {(c50 / c50.cummax() - 1).min() * 100:.0f}%"
              f"・最差年度 {(c50.groupby(c50.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1).min()) * 100:+.0f}%")
    configs = [("個股", 0.25, "first"), ("個股", 0.30, "first"), ("個股", 0.40, "first"), ("個股", 0.30, "zone"),
               ("ETF", 0.20, "first"), ("ETF", 0.25, "first"), ("全部", 0.30, "first")]
    for typ, drop, mode in configs:
        d = T[(T["停利"] == 0.10) & (T["模式"] == mode) & (T["跌幅"] == drop)]
        if typ != "全部":
            d = d[d["類"] == typ]
        d = d[d["代號"] != "0050"]
        if not len(d):
            continue
        for idle in ("cash", "0050"):
            res = [account(d, idle, seed=s) for s in range(3)]
            cagr = np.mean([r[0] for r in res]); mdd = np.mean([r[1] for r in res])
            ntr = np.mean([r[2] for r in res]); wy = np.mean([r[3] for r in res])
            L_.append(f"  {typ} 跌 {drop * 100:.0f}%・{mode}・閒置{'放 0050' if idle == '0050' else '現金'}："
                      f"年化 {cagr:+.1f}%・最大回撤 {mdd:.0f}%・最差年度 {wy:+.0f}%・成交 {ntr:.0f} 筆（3 次隨機平均）")
except Exception:  # noqa: BLE001
    L_.append("帳戶模擬失敗：" + traceback.format_exc()[-1500:])
L_.append("")


# ------------------------------------------------------------------ 六、引擎現有五種買點：舊 3%/15 天 vs 新停利
L_.append("==================== 六、App 現有五種買點：停利與持有天數比較（引擎回測，原始價）====================")
try:
    mkt = None
    try:
        if not FAKE:
            mkt = E.load_index()
    except Exception:  # noqa: BLE001
        mkt = None
    codes = []
    for sec in SECTORS:
        for s in stocks_of(sec):
            if s["code"] not in codes:
                codes.append(s["code"])
    codes = codes[:int(os.environ.get("NR_ENGINE_N", 400))]
    raw = P if FAKE else fetch({c: U.get(c, "TW") for c in codes}, adjust=False)
    VARIANTS = [("舊 3%・15 天", {"target_net": 3.0, "hold_days": 15}),
                ("5%・60 天", {"target_net": 5.0, "hold_days": 60}),
                ("10%・60 天", {"target_net": 10.0, "hold_days": 60}),
                ("10%・60 天・不看股性", {"target_net": 10.0, "hold_days": 60, "_nosuit": True})]
    allt = {k: [] for k, _ in VARIANTS}
    for c in codes:
        if c not in raw:
            continue
        df = raw[c].copy()
        df["volume"] = df["volume"] / 1000
        try:
            a = E.Analyzer(df, c, {}, mkt=mkt)
        except Exception:  # noqa: BLE001
            continue
        base = dict(a.p)
        for k, pv in VARIANTS:
            a.p = {**base, **{x: v for x, v in pv.items() if not x.startswith("_")}}
            if pv.get("_nosuit"):
                a.suitability = lambda i: None
                a.p["min_rr"] = 0.3
            else:
                a.suitability = E.Analyzer.suitability.__get__(a)
            bt = a.backtest()
            if len(bt):
                bt["代號"] = c
                bt["類"] = "ETF" if a.etf else "個股"
                allt[k].append(bt)
    for k, _ in VARIANTS:
        if not allt[k]:
            L_.append(f"  {k}：0 筆")
            continue
        b = pd.concat(allt[k])
        b["進場日"] = pd.to_datetime(b["進場日"])
        for typ in ("個股", "ETF"):
            x = b[b["類"] == typ]
            if not len(x):
                continue
            pre, post = x[x["進場日"] < SPLIT], x[x["進場日"] >= SPLIT]
            L_.append(f"  {k}｜{typ}：{len(x)} 筆・先到停利 {(x['結果'] == '停利').mean() * 100:.0f}%・每筆 {x['淨報酬%'].mean():+.2f}%"
                      f"・抱 {x['持有天數'].mean():.1f} 天｜前段 {len(pre)} 筆 {pre['淨報酬%'].mean() if len(pre) else float('nan'):+.2f}%"
                      f"｜後段 {len(post)} 筆 {post['淨報酬%'].mean() if len(post) else float('nan'):+.2f}%")
except Exception:  # noqa: BLE001
    L_.append("引擎比較失敗：" + traceback.format_exc()[-1500:])

try:
    T.to_csv(OUT / "newrule_trades.csv", index=False, encoding="utf-8-sig")
except Exception:  # noqa: BLE001
    pass
txt = "\n".join(L_)
(OUT / "newrule.txt").write_text(txt, encoding="utf-8")
print(txt)
print(f"總耗時 {time.time() - t0:.0f} 秒")
