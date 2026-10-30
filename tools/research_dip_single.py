"""單一 AI 股＋0050 模擬（2024-01～今）：
平常 50 萬全放 0050；個股收盤距近 60 日最高點跌 ≥35% 且手上沒有 → 隔天開盤賣 10 萬 0050、買個股；
淨賺 10% 賣／買價 −20% 停損／最多 60 個交易日；賣掉的錢隔天開盤買回 0050。
對照：50 萬 0050 放著。還原權息價計算，交易明細另附原始價。成本：手續費 0.1425%、股票證交稅 0.3%、0050 證交稅 0.1%、滑價每邊 0.1%。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sectors import SECTORS  # noqa: E402

START, CAP, PER = "2024-01-01", 500_000, 100_000
FEE, SLIP = 0.001425, 0.001
MAIN = "3037"
pool = {c: (n, m) for items in SECTORS["AI 產業"].values() for c, n, m in items}


def load(tk, adj=True):
    h = yf.Ticker(tk).history(period="4y", auto_adjust=adj, actions=False)
    h = h[h["Volume"] > 0].rename(columns=str.lower)
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h


E50 = load("0050.TW")


def run(code, mkt):
    try:
        s = load(f"{code}.{mkt}")
        raw = load(f"{code}.{mkt}", adj=False)
        if len(s) < 300:
            raise ValueError
    except Exception:  # noqa: BLE001
        alt = "TWO" if mkt == "TW" else "TW"
        s = load(f"{code}.{alt}"); raw = load(f"{code}.{alt}", adj=False)
    cal = E50.index[E50.index >= START]
    s = s.reindex(E50.index).ffill()
    raw = raw.reindex(E50.index).ffill()
    h60 = s.high.rolling(60).max(); dd = s.close / h60 - 1
    u50 = CAP * (1 - FEE - SLIP) / E50.open.loc[cal[0]]
    cash = 0.0; pos = None; pend_buy = False; pend_sell = None; eq = []; log = []
    for k, day in enumerate(cal):
        o50, c50 = E50.open.loc[day], E50.close.loc[day]
        if pend_sell is not None:          # 賣個股的錢今天開盤買回 0050
            u50 += pend_sell * (1 - FEE - SLIP) / o50; pend_sell = None
        if pend_buy and pos is None:
            px = s.open.loc[day]
            u50 -= PER / (1 - FEE - 0.001 - SLIP) / o50
            pos = dict(d0=day, px=px, raw=raw.open.loc[day], k0=k, lo=px,
                       tp=px * (1 + FEE + SLIP) * 1.10 / (1 - FEE - 0.003 - SLIP), sp=px * 0.80)
        pend_buy = False
        if pos is not None:
            o, hh, ll, cc = s.open.loc[day], s.high.loc[day], s.low.loc[day], s.close.loc[day]
            pos["lo"] = min(pos["lo"], ll); ex = None
            if day > pos["d0"] and o <= pos["sp"]: ex = (o, "停損(跳空)")
            elif ll <= pos["sp"]: ex = (pos["sp"], "停損 −20%")
            elif day > pos["d0"] and o >= pos["tp"]: ex = (o, "停利(跳空)")
            elif hh >= pos["tp"]: ex = (pos["tp"], "停利 +10%")
            elif k - pos["k0"] + 1 >= 60: ex = (cc, "滿 60 天")
            if ex:
                shares = PER / (pos["px"] * (1 + FEE + SLIP))
                val = shares * ex[0] * (1 - FEE - 0.003 - SLIP)
                log.append(dict(買進日=pos["d0"].strftime("%Y-%m-%d"), 買價_原始=round(pos["raw"], 1),
                                賣出日=day.strftime("%Y-%m-%d"), 賣價_原始=round(ex[0] * pos["raw"] / pos["px"], 1),
                                原因=ex[1], 抱幾天=k - pos["k0"] + 1, 損益元=round(val - PER),
                                中間最多虧=f"{pos['lo']/pos['px']-1:+.1%}"))
                pend_sell = val; pos = None
        mark = 0.0 if pos is None else PER / (pos["px"] * (1 + FEE + SLIP)) * s.close.loc[day]
        eq.append(u50 * c50 + mark + (pend_sell or 0))
        if pos is None and pend_sell is None and dd.loc[day] <= -0.35:
            pend_buy = True
    e = pd.Series(eq, index=cal)
    hold = CAP * E50.close.loc[cal] / E50.open.loc[cal[0]]
    return e, hold, pd.DataFrame(log)


def stats(e):
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return e.iloc[-1], (e.iloc[-1] / CAP) ** (1 / yrs) - 1, (e / e.cummax() - 1).min()


L = [f"# 回檔布局＋閒置放 0050｜單一 AI 股模擬｜{START}～今｜50 萬、每次 10 萬"]
e, hold, log = run(MAIN, pool[MAIN][1])
a, b = stats(e), stats(hold)
L.append(f"\n## 指定股票：{MAIN} {pool[MAIN][0]}（事先選定）")
L.append(f"- 本方法：期末 {a[0]:,.0f}｜總報酬 {a[0]/CAP-1:+.1%}｜年化 {a[1]:+.1%}｜最大回撤 {a[2]:.1%}")
L.append(f"- 0050 放著：期末 {b[0]:,.0f}｜總報酬 {b[0]/CAP-1:+.1%}｜年化 {b[1]:+.1%}｜最大回撤 {b[2]:.1%}")
L.append(f"- 差額：{a[0]-b[0]:+,.0f} 元；個股交易 {len(log)} 筆，合計損益 {log['損益元'].sum() if len(log) else 0:+,.0f} 元")
L.append(log.to_string(index=False) if len(log) else "（期間沒有觸發）")
m1 = e.resample("ME").last(); m2 = hold.resample("ME").last()
L.append("每月月底帳戶（本方法／0050 放著）：" + "、".join(f"{d.strftime('%y/%m')} {x/10000:.1f}萬/{y/10000:.1f}萬" for d, x, y in zip(m1.index, m1, m2)))

L.append("\n## AI 產業 40 檔各自跑一次（同樣規則）")
rows = []
for c, (n, m) in pool.items():
    try:
        e2, h2, lg = run(c, m)
        rows.append(dict(股票=f"{c} {n}", 交易筆數=len(lg), 個股損益=int(lg["損益元"].sum()) if len(lg) else 0,
                         贏0050=int(e2.iloc[-1] - h2.iloc[-1]), 回撤=f"{stats(e2)[2]:.1%}",
                         停利=int((lg["原因"].str.startswith("停利")).sum()) if len(lg) else 0,
                         停損=int((lg["原因"].str.startswith("停損")).sum()) if len(lg) else 0))
    except Exception as ex:  # noqa: BLE001
        rows.append(dict(股票=f"{c} {n}", 交易筆數=-1, 個股損益=0, 贏0050=0, 回撤=str(ex)[:30], 停利=0, 停損=0))
t = pd.DataFrame(rows).sort_values("贏0050")
ok = t[t.交易筆數 >= 0]
L.append(f"- 有觸發的 {(ok.交易筆數>0).sum()} 檔、沒觸發 {(ok.交易筆數==0).sum()} 檔；贏 0050 {(ok.贏0050>0).sum()} 檔、輸 {(ok.贏0050<0).sum()} 檔"
         f"｜平均差額 {ok.贏0050.mean():+,.0f} 元｜個股交易共 {ok.交易筆數.sum()} 筆（停利 {ok.停利.sum()}、停損 {ok.停損.sum()}）")
L.append(t.to_string(index=False))
txt = "\n".join(L)
open("research/dip_single.txt", "w").write(txt)
log.to_csv("research/dip_single_3037.csv", index=False, encoding="utf-8-sig")
print(txt)
