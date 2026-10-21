"""
自選股實測（2026-10-21）：「從近 60 日高點跌 ≥20% 就買」每筆投入 20 萬，逐筆列出真實日期與價格。
只做統計，不改規則。輸出 research/watchlist.txt。

規則（第四輪研究的方法）：
  訊號：收盤距近 60 日最高點第一次跌超過 X%（主版本 20%，另比較 25／30／40%）→ 隔天開盤買。
  買進：20 萬元能買的整數股（零股也算），扣手續費 0.1425%。
  賣出：淨賺 10%（扣手續費、證交稅 0.3%）就賣；跌到買價 −20% 停損；最多抱 60 個交易日，到期以收盤賣。
  另列「淨賺 5%」版本比較。同一檔賣掉前不再買。
價格：Yahoo Finance 日K「原始價」（就是當天看盤看到的價格，未還原權息），所以持有期間領到的股利沒算進去（偏保守）。
三檔事前指定（看結果之前就決定）：2330 台積電、2059 川湖、2383 台光電。
資料準確度檢查：同期間 Yahoo 收盤價 vs 證交所官方 STOCK_DAY 收盤價逐日比對。
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
import engine as E  # noqa: E402
from sectors import SECTORS  # noqa: E402

OUT = Path(os.environ.get("WL_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
FAKE = bool(os.environ.get("WL_FAKE"))
PICK = ["2330", "2059", "2383"]
BUDGET = 200_000
FEE, TAX, SLIP = 0.001425, 0.003, 0.001
HOLD, STOP = 60, 0.20

WL = [(c, n, m) for items in SECTORS["⭐ 自選股"].values() for (c, n, m) in items]
NAMES = {c: n for c, n, _ in WL}


def fetch_raw(code, mkt):
    if FAKE:
        rng = np.random.default_rng(int(code))
        idx = pd.bdate_range("2015-10-01", "2026-10-21")
        c = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.025, len(idx))))
        o = c * (1 + rng.normal(0, 0.008, len(idx)))
        return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                             "close": c}, index=idx)
    import yfinance as yf
    for suf in (f".{mkt}", ".TWO" if mkt == "TW" else ".TW"):
        try:
            h = yf.Ticker(code + suf).history(period="11y", interval="1d", auto_adjust=False, actions=False)
        except Exception:  # noqa: BLE001
            continue
        if h is None or len(h) < 300:
            continue
        h = h.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        h = h[(h["volume"] > 0) & (h["low"] > 0)]
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return h.astype(float)
    return None


def backtest(df, drop, net_target):
    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    hh = df["high"].rolling(60).max().to_numpy()
    dd = C / hh - 1
    trades, i, n = [], 60, len(df)
    bc, sc = FEE + SLIP, FEE + TAX + SLIP
    while i < n - 1:
        if not (dd[i] <= -drop and dd[i - 1] > -drop):
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
            if last == n - 1 and last - e + 1 < HOLD:
                res, x, xj = "持有中", C[last], last
            else:
                res, x, xj = "到期", C[last], last
        pnl = sh * x * (1 - sc) - cost
        mae = L[e:xj + 1].min() / px - 1
        trades.append({"訊號日": df.index[i], "距高點": dd[i], "買進日": df.index[e], "買價": px, "股數": sh,
                       "投入": cost, "賣出日": df.index[xj], "賣價": x, "結果": res, "損益": pnl,
                       "報酬%": pnl / cost * 100, "天數": xj - e + 1, "期間最大浮虧%": mae * 100})
        i = xj + 1
    return pd.DataFrame(trades)


def money(v):
    return f"{v:+,.0f}"


t0 = time.time()
data = {}
for c, n, m in WL:
    df = fetch_raw(c, m)
    if df is not None:
        data[c] = df
print(f"自選股 {len(WL)} 檔，抓到 {len(data)} 檔，{time.time() - t0:.0f} 秒")

L_ = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・資料：Yahoo Finance 日K原始價（盤後）",
      f"規則：距近 60 日高點跌 ≥20% 的第一天 → 隔天開盤買 20 萬；淨賺 10% 賣／跌 20% 停損／最多 60 個交易日。",
      "已扣手續費 0.1425%×2、證交稅 0.3%、滑價每邊 0.1%；持有期間股利未計入。", ""]

# 一、三檔逐筆明細
for c in PICK:
    if c not in data:
        L_.append(f"{c} 抓不到資料")
        continue
    df = data[c]
    t = backtest(df, 0.20, 0.10)
    L_.append(f"==================== {c} {NAMES.get(c, '')}（資料 {df.index[0]:%Y-%m-%d}～{df.index[-1]:%Y-%m-%d}）====================")
    big = (df["close"].pct_change().abs() > 0.15).sum()
    if big:
        L_.append(f"  注意：有 {big} 天單日漲跌超過 15%（可能是除權、分割或資料問題），該期間結果請保留看待。")
    if len(t) == 0:
        L_.append("  10 年內沒有出現訊號")
    for _, r in t.iterrows():
        L_.append(f"  {r['買進日']:%Y-%m-%d} 買 {r['買價']:,.2f} × {r['股數']:,} 股（{r['投入']:,.0f} 元，訊號日距高點 {r['距高點'] * 100:.0f}%）"
                  f" → {r['賣出日']:%Y-%m-%d} {r['結果']} {r['賣價']:,.2f}｜損益 {money(r['損益'])} 元（{r['報酬%']:+.1f}%）"
                  f"・抱 {r['天數']} 天・過程最大浮虧 {r['期間最大浮虧%']:.0f}%")
    done = t[t["結果"] != "持有中"]
    if len(done):
        L_.append(f"  合計：{len(done)} 筆・停利 {(done['結果'] == '停利').sum()}・停損 {(done['結果'] == '停損').sum()}"
                  f"・到期 {(done['結果'] == '到期').sum()}・成功率 {(done['結果'] == '停利').mean() * 100:.0f}%"
                  f"・總損益 {money(done['損益'].sum())} 元・每筆平均 {done['報酬%'].mean():+.1f}%")
    t5 = backtest(df, 0.20, 0.05)
    d5 = t5[t5["結果"] != "持有中"]
    if len(d5):
        L_.append(f"  （若改成淨賺 5% 就賣：{len(d5)} 筆・成功率 {(d5['結果'] == '停利').mean() * 100:.0f}%・總損益 {money(d5['損益'].sum())} 元）")
    first = df.index[60]
    shares = math.floor(BUDGET / (df["open"].iloc[61] * (1 + FEE)))
    bh = shares * df["close"].iloc[-1] * (1 - FEE - TAX) - shares * df["open"].iloc[61] * (1 + FEE)
    L_.append(f"  對照：{first:%Y-%m-%d} 起 20 萬買進一直放著（不含股利）：{money(bh)} 元")
    L_.append("")

# 二、全部自選股
L_.append("==================== 全部自選股（跌 20%／淨賺 10%／停損 20%／60 天，每筆 20 萬）====================")
rows = []
for c, df in data.items():
    for drop in (0.20, 0.25, 0.30, 0.40):
        t = backtest(df, drop, 0.10)
        d = t[t["結果"] != "持有中"] if len(t) else t
        rows.append({"代號": c, "跌幅": drop, "筆數": len(d), "停利": int((d["結果"] == "停利").sum()) if len(d) else 0,
                     "停損": int((d["結果"] == "停損").sum()) if len(d) else 0,
                     "總損益": float(d["損益"].sum()) if len(d) else 0.0,
                     "持有中": t.iloc[-1] if len(t) and t.iloc[-1]["結果"] == "持有中" else None})
R = pd.DataFrame(rows)
for c in data:
    r = R[(R["代號"] == c) & (R["跌幅"] == 0.20)].iloc[0]
    wr = f"{r['停利'] / r['筆數'] * 100:.0f}%" if r["筆數"] else "—"
    L_.append(f"  {c} {NAMES[c]:<6}：{r['筆數']:>2} 筆・停利 {r['停利']}・停損 {r['停損']}・成功率 {wr}・總損益 {money(r['總損益'])} 元")
for drop in (0.20, 0.25, 0.30, 0.40):
    g = R[R["跌幅"] == drop]
    n, w, s_ = g["筆數"].sum(), g["停利"].sum(), g["停損"].sum()
    L_.append(f"  ▶ 跌 {drop * 100:.0f}% 就買，全部自選股合計：{n} 筆・停利 {w}・停損 {s_}・成功率 {w / n * 100 if n else 0:.0f}%"
              f"・總損益 {money(g['總損益'].sum())} 元・每筆平均 {g['總損益'].sum() / n if n else 0:+,.0f} 元")
L_.append("")

# 三、現在的位置
L_.append("==================== 現在（最新收盤）距近 60 日高點 ====================")
for c, df in data.items():
    hh = df["high"].iloc[-60:].max()
    dd = df["close"].iloc[-1] / hh - 1
    flag = "◎ 已達 20%" if dd <= -0.20 else ""
    L_.append(f"  {c} {NAMES[c]:<6}：收 {df['close'].iloc[-1]:,.2f}（{df.index[-1]:%m/%d}）・60 日高 {hh:,.2f}・距高點 {dd * 100:+.0f}% {flag}")
L_.append("")

# 四、資料準確度：Yahoo vs 證交所
L_.append("==================== 資料準確度：Yahoo 收盤 vs 證交所官方收盤（逐日比對）====================")
if not FAKE:
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    for c in PICK:
        if c not in data:
            continue
        tw = []
        for ym in ("20260901", "20261001", "20250401", "20240801"):
            try:
                j = s.get(f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={ym}&stockNo={c}", timeout=15).json()
                for a in j.get("data", []):
                    y, m, d = a[0].split("/")
                    tw.append((pd.Timestamp(int(y) + 1911, int(m), int(d)), float(a[6].replace(",", ""))))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        if not tw:
            L_.append(f"  {c}：證交所資料抓取失敗，無法比對")
            continue
        T = pd.Series(dict(tw)).sort_index()
        Y = data[c]["close"].reindex(T.index)
        ok = Y.notna()
        diff = (Y[ok] / T[ok] - 1).abs()
        L_.append(f"  {c}：比對 {ok.sum()} 天（2024/8、2025/4、2026/9～10）・完全相同 {(diff < 0.0005).sum()} 天・"
                  f"差距 >0.5% 有 {(diff > 0.005).sum()} 天・Yahoo 缺的日子 {(~ok).sum()} 天")
        for dt_, v in diff[diff > 0.005].head(3).items():
            L_.append(f"    {dt_:%Y-%m-%d} Yahoo {Y[dt_]:,.2f} vs 證交所 {T[dt_]:,.2f}")
else:
    L_.append("  （離線測試略過）")

txt = "\n".join(L_)
(OUT / "watchlist.txt").write_text(txt, encoding="utf-8")
print(txt)
