"""
六檔回測（2026-10-27）：3037 欣興、6187 萬潤、2308 台達電、3711 日月光投控、3231 緯創、3324 雙鴻。
只做統計，不改規則。輸出 research/six.txt。

買法（每筆 20 萬、隔天開盤買、同一檔賣掉前不再買）：
  A 跌 X%：收盤距近 60 日最高點第一次跌超過 20／25／30%
  B 跌破月線：收盤第一次跌破 MA20（前一天還在上面）
  C 跌破 5 日線：收盤第一次跌破 MA5
  D 任一天買（對照組）：每天都買，看「不挑時機」的結果
賣法：淨賺 10%（另列 5%）賣；跌到買價 −20% 停損；最多 60 個交易日，到期收盤賣。
成本：手續費 0.1425%×2、證交稅 0.3%、滑價每邊 0.1%。價格：Yahoo 日K原始價，股利未計入（偏保守）。
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SIX_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
FAKE = bool(os.environ.get("SIX_FAKE"))
STOCKS = [("3037", "欣興", "TW"), ("6187", "萬潤", "TWO"), ("2308", "台達電", "TW"),
          ("3711", "日月光投控", "TW"), ("3231", "緯創", "TW"), ("3324", "雙鴻", "TWO")]
BUDGET = 200_000
FEE, TAX, SLIP = 0.001425, 0.003, 0.001
HOLD, STOP = 60, 0.20


def fetch_raw(code, mkt):
    if FAKE:
        rng = np.random.default_rng(int(code))
        idx = pd.bdate_range("2015-10-01", "2026-10-26")
        c = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.025, len(idx))))
        o = c * (1 + rng.normal(0, 0.008, len(idx)))
        return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                             "close": c, "volume": 1000.0}, index=idx)
    import yfinance as yf
    for suf in (f".{mkt}", ".TWO" if mkt == "TW" else ".TW"):
        for _ in range(3):
            try:
                h = yf.Ticker(code + suf).history(period="11y", interval="1d", auto_adjust=False, actions=False)
            except Exception:  # noqa: BLE001
                h = None
            if h is not None and len(h) >= 300:
                h = h.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                h = h[(h["volume"] > 0) & (h["low"] > 0)]
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                return h.astype(float)
            time.sleep(2)
    return None


def signals(df, kind):
    C = df["close"]
    if kind[1:3].isdigit():
        x = int(kind[1:3]) / 100
        dd = C / df["high"].rolling(60).max() - 1
        return ((dd <= -x) & (dd.shift(1) > -x)).to_numpy()
    if kind == "跌破月線":
        m = C.rolling(20).mean()
        return ((C < m) & (C.shift(1) >= m.shift(1))).to_numpy()
    if kind == "跌破5日線":
        m = C.rolling(5).mean()
        return ((C < m) & (C.shift(1) >= m.shift(1))).to_numpy()
    return np.ones(len(df), bool)  # 任一天


def backtest(df, kind, net_target):
    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sig = signals(df, kind)
    trades, i, n = [], 60, len(df)
    bc, sc = FEE + SLIP, FEE + TAX + SLIP
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        px = O[e]
        sh = math.floor(BUDGET / (px * (1 + bc)))
        if sh <= 0:
            i += 1
            continue
        cost = sh * px * (1 + bc)
        tp = px * (1 + bc) * (1 + net_target) / (1 - sc)
        sl = px * (1 - STOP)
        last = min(e + HOLD - 1, n - 1)
        res, x, xj = None, None, None
        for j in range(e, last + 1):
            if L[j] <= sl:
                res, x, xj = "停損", (min(O[j], sl) if j > e else sl), j
                break
            if H[j] >= tp:
                res, x, xj = "停利", (max(O[j], tp) if j > e else tp), j
                break
        if res is None:
            res = "持有中" if (last == n - 1 and last - e + 1 < HOLD) else "到期"
            x, xj = C[last], last
        pnl = sh * x * (1 - sc) - cost
        trades.append({"訊號日": df.index[i], "買進日": df.index[e], "買價": px, "賣出日": df.index[xj], "賣價": x,
                       "結果": res, "損益": pnl, "報酬%": pnl / cost * 100, "天數": xj - e + 1,
                       "浮虧%": (L[e:xj + 1].min() / px - 1) * 100, "年": df.index[e].year})
        i = xj + 1
    return pd.DataFrame(trades)


def summ(t):
    d = t[t["結果"] != "持有中"] if len(t) else t
    if not len(d):
        return "0 筆", d
    s = (f"{len(d):>3} 筆・成功 {(d['結果'] == '停利').mean() * 100:>3.0f}%・停損 {(d['結果'] == '停損').sum():>2}"
         f"・每筆 {d['報酬%'].mean():+5.1f}%・總損益 {d['損益'].sum():+10,.0f}・平均抱 {d['天數'].mean():.0f} 天"
         f"・最差 {d['報酬%'].min():+.0f}%・曾浮虧≥15% 占 {(d['浮虧%'] <= -15).mean() * 100:.0f}%")
    return s, d


KINDS = ["跌20%", "跌25%", "跌30%", "跌破月線", "跌破5日線", "任一天"]
t0 = time.time()
data = {}
for c, n, m in STOCKS:
    df = fetch_raw(c, m)
    if df is not None:
        data[c] = df
print(f"6 檔，抓到 {len(data)} 檔，{time.time() - t0:.0f} 秒")

L_ = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・資料：Yahoo Finance 日K原始價（盤後）",
      "賣法：淨賺 10% 賣／跌 20% 停損／最多 60 個交易日；每筆 20 萬；扣手續費、證交稅、滑價；股利未計。",
      "成功＝先到停利。『任一天』是對照組：每天都買的結果，挑時機的方法要比它好才算有用。", ""]

for c, n, m in STOCKS:
    if c not in data:
        L_.append(f"==================== {c} {n}：抓不到資料 ====================\n")
        continue
    df = data[c]
    L_.append(f"==================== {c} {n}（{df.index[0]:%Y-%m-%d}～{df.index[-1]:%Y-%m-%d}，{len(df)} 天）====================")
    big = (df["close"].pct_change().abs() > 0.15).sum()
    if big:
        L_.append(f"  注意：有 {big} 天單日漲跌超過 15%（除權、分割或資料問題），結果請保留看待。")
    for k in KINDS:
        s, _ = summ(backtest(df, k, 0.10))
        L_.append(f"  {k:<7}｜停利10%：{s}")
    for k in ("跌20%", "跌破月線", "任一天"):
        s, _ = summ(backtest(df, k, 0.05))
        L_.append(f"  {k:<7}｜停利 5%：{s}")
    t = backtest(df, "跌20%", 0.10)
    if len(t):
        L_.append("  跌20% 逐筆：")
        for _, r in t.iterrows():
            L_.append(f"    {r['買進日']:%Y-%m-%d} 買 {r['買價']:,.2f} → {r['賣出日']:%Y-%m-%d} {r['結果']} {r['賣價']:,.2f}"
                      f"｜{r['報酬%']:+.1f}%・抱 {r['天數']} 天・最大浮虧 {r['浮虧%']:.0f}%")
    # 現況
    C = df["close"]
    hh = df["high"].iloc[-60:].max()
    ma5, ma20, ma60 = C.iloc[-5:].mean(), C.iloc[-20:].mean(), C.iloc[-60:].mean()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - C.shift()).abs(), (df["low"] - C.shift()).abs()], axis=1).max(axis=1)
    atr = tr.iloc[-14:].mean() / C.iloc[-1] * 100
    L_.append(f"  現況（{df.index[-1]:%m/%d} 收盤）：收 {C.iloc[-1]:,.2f}・60 日高 {hh:,.2f}（{(C.iloc[-1] / hh - 1) * 100:+.0f}%）"
              f"・MA5 {ma5:,.1f}・MA20 {ma20:,.1f}・MA60 {ma60:,.1f}・近 60 日低 {df['low'].iloc[-60:].min():,.2f}・ATR {atr:.1f}%")
    last5 = df.iloc[-5:]
    L_.append("  最近 5 天：" + "；".join(f"{d:%m/%d} 開{r.open:,.1f} 高{r.high:,.1f} 低{r.low:,.1f} 收{r.close:,.1f} 量{r.volume / 1000:,.0f}張"
                                        for d, r in last5.iterrows()))
    L_.append("")

# 六檔合計
L_.append("==================== 六檔合計（停利 10%）====================")
for k in KINDS:
    ts = [backtest(df, k, 0.10) for df in data.values()]
    ts = [t for t in ts if len(t)]
    if not ts:
        continue
    s, d = summ(pd.concat(ts))
    L_.append(f"  {k:<7}：{s}")
    if len(d) and k in ("跌20%", "跌25%", "任一天"):
        yr = d.groupby("年")["報酬%"].agg(["count", "mean"])
        L_.append("      分年：" + "  ".join(f"{y} {int(r['count'])}筆{r['mean']:+.1f}%" for y, r in yr.iterrows()))

txt = "\n".join(L_)
(OUT / "six.txt").write_text(txt, encoding="utf-8")
print(txt)
