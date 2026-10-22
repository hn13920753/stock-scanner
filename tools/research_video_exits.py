"""
影片方法・出場參數比較（2026-10-22 第二次）：停利 3/4/5%、停損 紅K低/回檔低/固定 −7%/−10%、持有 15/30 天。
只做統計，不改規則。輸出 research/video_exits.txt。

（以下為原影片方法說明）
影片方法回測（2026-10-22）：朱家泓「多頭回檔後買上漲」——價漲、量增、線實。
只做統計，不改規則。輸出 research/video_pullback.txt。

影片原話（字幕）整理成可計算的規則：
  先決條件（兩個都要成立）
    1. 多頭股票拉回整理（回檔）
    2. 回檔「不可以把前面的低點跌破」、「也不可以跌破月線」
  當天確認訊號（三個同時符合）
    價漲：收盤價突破前一天的高點
    量增：成交量大於昨天的 1.3 倍
    線實：紅K實體、不要長的上影線、漲幅大於 2%
    另：站上 5 日均線
  進場後一定要設停損。

本腳本的數字定義（影片沒有講到的部分由本研究補上，事前固定、不看結果調整）：
  多頭：訊號前一天 MA20 > MA60，且 MA20 比 5 天前高（月線上揚）。
  回檔：近 15 天內的最高點出現在訊號日之前 2～15 天，之後最低點比該高點低 ≥ 3%。
  不破前低：回檔最低點 > 該高點之前 20 天內的最低點。
  不破月線：回檔期間每天收盤 ≥ 當天 MA20。
  線實：收 > 開、上影線 ≤ 全日振幅 25%、漲幅 > 2%。
  進場：A 訊號當天收盤買（盤中最後確認）；B 隔天開盤買。
  停損：訊號紅K最低點（影片示範的轉折低點）。
  停利：扣費淨賺 3%（股票系統標準）；另比較 2%、5%。最多抱 15 個交易日，到期收盤賣。
  同一檔賣出前不重複進場。手續費 0.1425%×2、證交稅 0.3%、滑價每邊 0.1%。
  價格：Yahoo 還原權息價計算（避免除息缺口誤觸停損），列表另附當時原始價。
對照組：同樣是「多頭＋不破月線」的任一天收盤買、同樣出場（停損改用當天最低點），
用來看這三個條件到底有沒有比隨便一天買更好。
指定標的（看結果之前決定）：2330 台積電；另跑全部自選股。
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sectors import SECTORS  # noqa: E402

OUT = Path(os.environ.get("VP_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
FAKE = bool(os.environ.get("VP_FAKE"))
PICK = "2330"
BUDGET = 200_000
FEE, TAX, SLIP = 0.001425, 0.003, 0.001
HOLD = 15

WL = [(c, n, m) for items in SECTORS["⭐ 自選股"].values() for (c, n, m) in items]
NAMES = {c: n for c, n, _ in WL}


def fetch(code, mkt):
    if FAKE:
        rng = np.random.default_rng(int(code))
        idx = pd.bdate_range("2015-10-01", "2026-10-21")
        c = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, len(idx))))
        o = c * (1 + rng.normal(0, 0.008, len(idx)))
        v = rng.lognormal(10, 0.5, len(idx))
        return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                             "close": c, "volume": v, "raw_close": c}, index=idx)
    import yfinance as yf
    for suf in (f".{mkt}", ".TWO" if mkt == "TW" else ".TW"):
        try:
            h = yf.Ticker(code + suf).history(period="11y", interval="1d", auto_adjust=False, actions=False)
        except Exception:  # noqa: BLE001
            continue
        if h is None or len(h) < 300:
            continue
        h = h.rename(columns=str.lower)
        h = h[(h["volume"] > 0) & (h["low"] > 0)].copy()
        r = h["adj close"] / h["close"]
        df = pd.DataFrame({"open": h["open"] * r, "high": h["high"] * r, "low": h["low"] * r,
                           "close": h["adj close"], "volume": h["volume"], "raw_close": h["close"]})
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df.astype(float)
    return None


def features(df):
    d = df.copy()
    C, H, L, O, V = d["close"], d["high"], d["low"], d["open"], d["volume"]
    d["ma5"], d["ma20"], d["ma60"] = C.rolling(5).mean(), C.rolling(20).mean(), C.rolling(60).mean()
    bull = (d["ma20"] > d["ma60"]) & (d["ma20"] > d["ma20"].shift(5))
    d["bull_prev"] = bull.shift(1).fillna(False).astype(bool)
    rng = (H - L).replace(0, np.nan)
    d["price_up"] = C > H.shift(1)
    d["vol_up"] = V >= 1.3 * V.shift(1)
    d["solid"] = (C > O) & ((H - C) / rng <= 0.25) & (C / C.shift(1) - 1 > 0.02)
    d["above5"] = C > d["ma5"]
    # 回檔條件（逐日計算，只用訊號日以前的資料）
    Hn, Ln, Cn, M20 = H.to_numpy(), L.to_numpy(), C.to_numpy(), d["ma20"].to_numpy()
    n = len(d)
    pull = np.zeros(n, bool)
    no_break_low = np.zeros(n, bool)
    no_break_ma = np.zeros(n, bool)
    depth = np.full(n, np.nan)
    pbl = np.full(n, np.nan)
    for i in range(80, n):
        w0 = i - 15
        k = w0 + int(np.argmax(Hn[w0:i]))          # 近 15 天（不含今天）最高點
        if not (2 <= i - k <= 15):
            continue
        hi = Hn[k]
        lo = Ln[k + 1:i].min() if i - k > 1 else np.inf
        dp = lo / hi - 1
        depth[i] = dp
        pbl[i] = lo
        pull[i] = dp <= -0.03
        prev_low = Ln[max(0, k - 20):k].min()
        no_break_low[i] = lo > prev_low
        seg = slice(k + 1, i)
        no_break_ma[i] = bool(np.all(Cn[seg] >= M20[seg]))
    d["pullback"], d["no_break_low"], d["no_break_ma"], d["pb_depth"], d["pb_low"] = pull, no_break_low, no_break_ma, depth, pbl
    d["setup"] = d["bull_prev"] & d["pullback"] & d["no_break_low"] & d["no_break_ma"]
    d["signal"] = d["setup"] & d["price_up"] & d["vol_up"] & d["solid"] & d["above5"]
    d["base"] = d["bull_prev"] & (C >= d["ma20"])
    return d


def run(d, mask, entry, net, stop_kind="signal_low", hold=HOLD):
    HOLD_ = hold
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    RC = d["raw_close"].to_numpy()
    sig = d[mask].to_numpy() if isinstance(mask, str) else mask
    n = len(d)
    bc, sc = FEE + SLIP, FEE + TAX + SLIP
    out, i = [], 80
    while i < n:
        if not sig[i]:
            i += 1
            continue
        if entry == "close":
            e, px, first = i, C[i], i + 1
        else:
            if i + 1 >= n:
                break
            e, px, first = i + 1, O[i + 1], i + 1
        if stop_kind == "signal_low":
            sl = L[i]
        elif stop_kind == "pb_low":
            sl = min(L[i], d["pb_low"].iat[i])
        else:
            sl = (C[i] if entry == "close" else O[i + 1]) * (1 - float(stop_kind))
        if px <= sl * 1.001:            # 開盤已在停損下 → 不進場
            i += 1
            continue
        tp = px * (1 + bc) * (1 + net) / (1 - sc)
        last = min(e + HOLD_ - (0 if entry == "close" else 1), n - 1)
        res = x = xj = None
        for j in range(first, last + 1):
            if O[j] <= sl:
                res, x, xj = "停損", O[j], j; break
            if L[j] <= sl:
                res, x, xj = "停損", sl, j; break
            if H[j] >= tp:
                res, x, xj = "停利", max(O[j], tp), j; break
        if res is None:
            if first > n - 1:
                break
            res, x, xj = ("持有中" if last == n - 1 and (last - e) < HOLD_ - 1 else "到期"), C[last], last
        sh = math.floor(BUDGET / (px * (1 + bc)))
        cost = sh * px * (1 + bc)
        pnl = sh * x * (1 - sc) - cost
        out.append({"訊號日": d.index[i].date(), "買進日": d.index[e].date(), "買價(原始)": px * RC[e] / C[e],
                    "停損距%": (sl / px - 1) * 100, "賣出日": d.index[xj].date(), "結果": res,
                    "報酬%": pnl / cost * 100, "損益": pnl, "天數": xj - e + (1 if entry == "close" else 1),
                    "最大浮虧%": (L[first:xj + 1].min() / px - 1) * 100 if xj >= first else 0.0,
                    "年": d.index[i].year})
        i = xj + 1
    return pd.DataFrame(out)


def summ(t, label):
    t = t[t["結果"] != "持有中"] if len(t) else t
    if not len(t):
        return f"  {label:<34} 0 筆"
    w = (t["報酬%"] > 0).mean() * 100
    tp = (t["結果"] == "停利").mean() * 100
    sl = (t["結果"] == "停損").mean() * 100
    return (f"  {label:<34} {len(t):>4} 筆  成功率(賺錢) {w:4.0f}%  先停利 {tp:3.0f}%  先停損 {sl:3.0f}%  "
            f"平均 {t['報酬%'].mean():+.2f}%  合計 {t['損益'].sum():+,.0f} 元  平均抱 {t['天數'].mean():.1f} 天")



def stats(t):
    if not len(t):
        return None
    t = t[t["結果"] != "持有中"]
    if not len(t):
        return None
    win = t[t["報酬%"] > 0]["報酬%"]; los = t[t["報酬%"] <= 0]["報酬%"]
    yr = t.groupby("年")["報酬%"].mean()
    return dict(n=len(t), w=(t["報酬%"] > 0).mean() * 100, avg=t["報酬%"].mean(), tot=t["損益"].sum(),
                aw=win.mean() if len(win) else 0, al=los.mean() if len(los) else 0, worst=t["報酬%"].min(),
                mae=t["最大浮虧%"].quantile(0.1), days=t["天數"].mean(), exp=(t["結果"] == "到期").mean() * 100,
                yp=int((yr > 0).sum()), yn=len(yr),
                a=t[pd.to_datetime(t["訊號日"]) < "2023-01-01"]["報酬%"].mean(),
                b=t[pd.to_datetime(t["訊號日"]) >= "2023-01-01"]["報酬%"].mean())


def line(lbl, s):
    if s is None:
        return f"  {lbl:<26} 0 筆"
    return (f"  {lbl:<26} {s['n']:>4}筆 成功率{s['w']:4.0f}% 平均{s['avg']:+.2f}% 合計{s['tot']:+,.0f} "
            f"｜賺時{s['aw']:+.2f}% 賠時{s['al']:+.2f}% 最差{s['worst']:+.1f}% 一成曾浮虧≥{-s['mae']:.1f}% "
            f"到期{s['exp']:.0f}% 抱{s['days']:.1f}天｜正年{s['yp']}/{s['yn']}｜16-22 {s['a']:+.2f}% 23- {s['b']:+.2f}%")


t0 = time.time()
data = {}
for c, nme, m in WL:
    df = fetch(c, m)
    if df is not None:
        data[c] = features(df)
print(f"抓到 {len(data)} 檔，{time.time() - t0:.0f} 秒")
STOPS = [("signal_low", "紅K低點"), ("pb_low", "回檔低點"), ("0.07", "固定−7%"), ("0.10", "固定−10%")]
L_ = ["影片方法・出場參數比較（訊號當天收盤買、每筆 20 萬、扣費滑價、還原權息）", ""]
for hold in (15, 30):
    L_.append(f"==================== 全部自選股｜最多持有 {hold} 天 ====================")
    for net in (0.03, 0.04, 0.05):
        for sk, sn in STOPS:
            ts = [run(dd, "signal", "close", net, sk, hold) for dd in data.values()]
            ts = pd.concat([x for x in ts if len(x)], ignore_index=True)
            L_.append(line(f"停利{net*100:.0f}%｜停損{sn}", stats(ts)))
        L_.append("")
    L_.append(f"  對照（多頭任一天買，同出場）｜最多 {hold} 天")
    for net in (0.04, 0.05):
        for sk, sn in STOPS[2:]:
            ts = [run(dd, "base", "close", net, sk, hold) for dd in data.values()]
            ts = pd.concat([x for x in ts if len(x)], ignore_index=True)
            L_.append(line(f"任一天 停利{net*100:.0f}%｜停損{sn}", stats(ts)))
    L_.append("")
d = data.get(PICK)
if d is not None:
    L_.append(f"==================== 2330 台積電（最多 15 天）====================")
    for net in (0.04, 0.05):
        for sk, sn in STOPS:
            L_.append(line(f"停利{net*100:.0f}%｜停損{sn}", stats(run(d, "signal", "close", net, sk, 15))))
# 每檔：停利 5%／停損 −10%／15 天
L_.append("")
L_.append("每檔（停利 5%｜停損 −10%｜15 天）：")
for c, dd in data.items():
    s = stats(run(dd, "signal", "close", 0.05, "0.10", 15))
    if s:
        L_.append(f"  {c} {NAMES[c]:<6} {s['n']:>3}筆 成功率{s['w']:4.0f}% 平均{s['avg']:+.2f}% 最差{s['worst']:+.1f}%")
txt = "\n".join(L_)
print(txt)
(OUT / "video_exits.txt").write_text(txt, encoding="utf-8")
