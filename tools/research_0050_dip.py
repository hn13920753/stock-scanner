"""只做 0050：比較「0050 放著」與「平常放現金、0050 跌深才買、賺 10% 賣」等做法。
0050 還原權息價（Yahoo auto_adjust）；成本：手續費 0.1425%×2、ETF 證交稅 0.1%、滑價每邊 0.1%。
訊號：收盤距近 60 日最高點跌 ≥ X% → 隔天開盤買；賣：淨賺 10%（或 5%）、停損（無／−20%）。
A 全部現金等訊號，訊號出現全押 0050；B 平常 0050 放 50%，訊號時把另外 50% 現金也買進，賣出後回到 50%。"""
import numpy as np
import pandas as pd
import yfinance as yf

FEE, TAX, SLIP = 0.001425, 0.001, 0.001
BC, SC = FEE + SLIP, FEE + TAX + SLIP
d = yf.Ticker("0050.TW").history(period="11y", auto_adjust=True, actions=False)
d = d[d.Volume > 0].dropna(subset=["Open", "High", "Low", "Close"]).rename(columns=str.lower)
d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
dd = (d.close / d.high.rolling(60).max() - 1).to_numpy()
idx = d.index


def sim(start, drop, tgt, stop, base):
    """回傳 (期末/期初, 最大回撤, 交易筆數, 在場時間比例)。base＝平常持有 0050 的比例（0 或 0.5）。"""
    k0 = int(np.searchsorted(idx, pd.Timestamp(start)))
    cash = 1.0
    core = 0.0
    if base:
        core = base * (1 - BC) / O[k0]; cash = 1 - base
    pos = None; eq = []; n = 0; inside = 0; pend = False
    for k in range(k0, len(d)):
        if pend and pos is None:
            px = O[k]; amt = cash
            pos = dict(u=amt * (1 - BC) / px, px=px, tp=px * (1 + BC) * (1 + tgt) / (1 - SC), sp=px * (1 - stop) if stop else -1)
            cash = 0.0; n += 1
        pend = False
        if pos is not None:
            inside += 1
            ex = None
            if O[k] <= pos["sp"] and k > 0: ex = O[k]
            elif L[k] <= pos["sp"]: ex = pos["sp"]
            elif O[k] >= pos["tp"]: ex = O[k]
            elif H[k] >= pos["tp"]: ex = pos["tp"]
            if ex is not None:
                cash += pos["u"] * ex * (1 - SC); pos = None
        eq.append(cash + core * C[k] + (pos["u"] * C[k] if pos else 0))
        if pos is None and dd[k] <= -drop and cash > 0:
            pend = True
    e = pd.Series(eq, index=idx[k0:])
    return e.iloc[-1] / e.iloc[0], (e / e.cummax() - 1).min(), n, inside / len(e)


L_ = ["# 只做 0050：跌深買、賺就賣 vs 放著（50 萬為例）"]
for start, lab in (("2016-06-01", "2016/6～今"), ("2025-01-01", "2025/1～今")):
    yrs = (idx[-1] - pd.Timestamp(start)).days / 365.25
    k0 = int(np.searchsorted(idx, pd.Timestamp(start)))
    hold = C[-1] / O[k0] * (1 - BC)
    hs = pd.Series(C[k0:], index=idx[k0:])
    L_.append(f"\n## {lab}\n- 0050 放著：50 萬 → {50*hold:.1f} 萬｜年化 {hold**(1/yrs)-1:+.1%}｜回撤 {(hs/hs.cummax()-1).min():.0%}")
    for base in (0, 0.5):
        for drop in (0.05, 0.10, 0.15, 0.20):
            for tgt, stop in ((0.10, None), (0.10, 0.20), (0.05, None)):
                m, mdd, n, ins = sim(start, drop, tgt, stop, base)
                L_.append(f"- {'平常全現金' if not base else '平常 50% 0050'}｜跌 {drop:.0%} 買｜賺 {tgt:.0%} 賣｜停損 {'無' if not stop else f'−{stop:.0%}'}："
                          f"50 萬 → {50*m:.1f} 萬｜年化 {m**(1/yrs)-1:+.1%}｜回撤 {mdd:.0%}｜交易 {n} 次｜在場時間 {ins:.0%}")
txt = "\n".join(L_)
open("research/only_0050.txt", "w").write(txt)
print(txt)
