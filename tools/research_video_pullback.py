"""
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


def run(d, mask, entry, net, stop_kind="signal_low"):
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
        sl = L[i] if stop_kind == "signal_low" else min(L[i], d["pb_low"].iat[i])
        if px <= sl * 1.001:            # 開盤已在停損下 → 不進場
            i += 1
            continue
        tp = px * (1 + bc) * (1 + net) / (1 - sc)
        last = min(e + HOLD - (0 if entry == "close" else 1), n - 1)
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
            res, x, xj = ("持有中" if last == n - 1 and (last - e) < HOLD - 1 else "到期"), C[last], last
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


t0 = time.time()
data = {}
for c, nme, m in WL:
    df = fetch(c, m)
    if df is not None:
        data[c] = features(df)
print(f"自選股 {len(WL)} 檔，抓到 {len(data)} 檔，{time.time() - t0:.0f} 秒")

L_ = ["影片方法回測：朱家泓「多頭回檔後買上漲」（價漲・量增・線實）",
      f"執行 {pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・每筆 20 萬・最多 15 交易日・扣手續費/證交稅/滑價",
      ""]

d = data.get(PICK)
if d is not None:
    L_.append(f"==================== 指定標的 {PICK} {NAMES[PICK]}（資料 {d.index[0].date()} ～ {d.index[-1].date()}）====================")
    L_.append(f"  條件逐層過濾的天數：多頭 {int(d['bull_prev'].sum())}・+回檔≥3% {int((d['bull_prev'] & d['pullback']).sum())}"
              f"・+不破前低/月線 {int(d['setup'].sum())}・+價漲量增線實站5均（訊號）{int(d['signal'].sum())}")
    L_.append("  ── 主版本：訊號當天收盤買、停損=紅K低點 ──")
    for net in (0.02, 0.03, 0.05):
        L_.append(summ(run(d, "signal", "close", net), f"影片訊號｜停利淨 {net * 100:.0f}%"))
    for net in (0.03, 0.05):
        L_.append(summ(run(d, "signal", "close", net, "pb_low"), f"影片訊號｜停損=回檔低點｜停利 {net * 100:.0f}%"))
    L_.append("  ── 隔天開盤買 ──")
    for net in (0.03,):
        L_.append(summ(run(d, "signal", "open", net), f"影片訊號｜隔天開盤｜停利 3%"))
    L_.append("  ── 對照組：多頭＋站上月線的任一天收盤買（停損=當天低點）──")
    for net in (0.02, 0.03, 0.05):
        L_.append(summ(run(d, "base", "close", net), f"任一天｜停利淨 {net * 100:.0f}%"))
    L_.append("  ── 拆解：少一個條件會怎樣（收盤買、停利 3%）──")
    core = d["price_up"] & d["vol_up"] & d["solid"] & d["above5"]
    L_.append(summ(run(d, (d["bull_prev"] & core).to_numpy(), "close", 0.03), "只有三條件（不管回檔/前低/月線）"))
    L_.append(summ(run(d, (d["setup"] & d["price_up"] & d["above5"]).to_numpy(), "close", 0.03), "回檔＋價漲（不要求量增線實）"))
    t = run(d, "signal", "close", 0.03)
    if len(t):
        L_.append("")
        L_.append("  逐筆（主版本，停利 3%）：")
        tt = t.copy()
        tt["買價(原始)"] = tt["買價(原始)"].round(1)
        for k in ("停損距%", "報酬%", "最大浮虧%"):
            tt[k] = tt[k].round(2)
        tt["損益"] = tt["損益"].round(0).astype(int)
        L_.append(tt.drop(columns=["年"]).to_string(index=False))
        L_.append("")
        L_.append("  分年：")
        for y, g in t.groupby("年"):
            L_.append(f"    {y}: {len(g)} 筆・賺錢 {(g['報酬%'] > 0).mean() * 100:.0f}%・平均 {g['報酬%'].mean():+.2f}%")
    last = d.iloc[-1]
    L_.append("")
    L_.append(f"  最新一天 {d.index[-1].date()}：收盤 {last['raw_close']:.1f}｜多頭 {bool(last['bull_prev'])}"
              f"｜回檔≥3% {bool(last['pullback'])}（幅度 {last['pb_depth'] * 100 if pd.notna(last['pb_depth']) else float('nan'):.1f}%）"
              f"｜不破前低 {bool(last['no_break_low'])}｜不破月線 {bool(last['no_break_ma'])}"
              f"｜價漲 {bool(last['price_up'])}｜量增 {bool(last['vol_up'])}｜線實 {bool(last['solid'])}｜站5均 {bool(last['above5'])}"
              f"｜→ 訊號 {bool(last['signal'])}；先決條件 {bool(last['setup'])}")
    L_.append(f"  MA5 {last['ma5'] * last['raw_close'] / last['close']:.1f}｜MA20 {last['ma20'] * last['raw_close'] / last['close']:.1f}"
              f"｜MA60 {last['ma60'] * last['raw_close'] / last['close']:.1f}（原始價換算）")

L_.append("")
L_.append("==================== 全部自選股合計（收盤買、停損=紅K低點、最多 15 天）====================")
for net in (0.02, 0.03, 0.05):
    allt = [run(dd, "signal", "close", net).assign(代號=c) for c, dd in data.items()]
    allt = pd.concat([x for x in allt if len(x)], ignore_index=True) if any(len(x) for x in allt) else pd.DataFrame()
    L_.append(summ(allt, f"影片訊號｜停利 {net * 100:.0f}%"))
    allb = [run(dd, "base", "close", net) for dd in data.values()]
    allb = pd.concat([x for x in allb if len(x)], ignore_index=True)
    L_.append(summ(allb, f"對照任一天｜停利 {net * 100:.0f}%"))
    allp = [run(dd, "signal", "close", net, "pb_low") for dd in data.values()]
    allp = [x for x in allp if len(x)]
    if allp:
        L_.append(summ(pd.concat(allp, ignore_index=True), f"影片訊號｜停損=回檔低點｜停利 {net * 100:.0f}%"))
    if net == 0.03 and len(allt):
        allt["樣本"] = np.where(pd.to_datetime(allt["訊號日"]) < "2023-01-01", "2016-2022", "2023-")
        for s, g in allt.groupby("樣本"):
            L_.append(summ(g, f"   └ 期間 {s}"))
        L_.append("   每檔（停利 3%）：")
        for c, g in allt.groupby("代號"):
            g2 = g[g["結果"] != "持有中"]
            if len(g2):
                L_.append(f"     {c} {NAMES[c]:<6} {len(g2):>3} 筆・賺錢 {(g2['報酬%'] > 0).mean() * 100:3.0f}%・平均 {g2['報酬%'].mean():+.2f}%・合計 {g2['損益'].sum():+,.0f}")
L_.append("")
L_.append("最新一天自選股狀態（先決條件成立＝回檔中、等確認紅K）：")
for c, dd in data.items():
    r = dd.iloc[-1]
    if r["signal"] or r["setup"]:
        L_.append(f"  {c} {NAMES[c]}：{'★今天出現訊號' if r['signal'] else '先決條件成立，等價漲量增線實'}｜收 {r['raw_close']:.1f}"
                  f"｜回檔 {r['pb_depth'] * 100:.1f}%")

txt = "\n".join(L_)
print(txt)
(OUT / "video_pullback.txt").write_text(txt, encoding="utf-8")
