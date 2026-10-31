"""0050 每月提領 5 萬模擬（2014 起；Yahoo 更早資料有錯）：本金全部放 0050（還原權息、扣賣出成本），每月第一個交易日開盤賣出 5 萬。
起始月份：2004～2024 每年 1 月；看 3 年後、5 年後剩多少，以及會不會領光。"""
import numpy as np
import pandas as pd
import yfinance as yf

d = yf.Ticker("0050.TW").history(period="max", auto_adjust=True, actions=False)
d = d[(d.Volume > 0) & (d.Open > 0)].dropna(subset=["Open", "Close"])
d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
d = d[d.index >= "2014-01-03"]  # Yahoo 2014-01-02 以前資料有 −75% 錯誤與開盤 0
first_days = d.groupby(d.index.to_period("M")).head(1)
SC = 0.001425 + 0.001 + 0.001
W = 50_000
out = [f"# 0050 每月領 5 萬（資料 {d.index[0]:%Y-%m}～{d.index[-1]:%Y-%m}，還原權息）"]
for cap in (1_000_000, 2_000_000, 3_000_000, 5_000_000, 8_000_000, 12_000_000):
    rows = []
    for y in range(2014, 2025):
        st = pd.Timestamp(f"{y}-01-01")
        fd = first_days[first_days.index >= st]
        units = cap * (1 - 0.002425) / fd.Open.iloc[0]
        res, low, dead = {}, cap, None
        for k, (day, r) in enumerate(fd.iterrows()):
            if k > 0:
                need = W / (1 - SC) / r.Open
                if units < need:
                    dead = k; units = 0; break
                units -= need
            val = units * r.Close
            low = min(low, val)
            if k == 36: res[3] = val
            if k == 60: res[5] = val
        rows.append((y, res.get(3), res.get(5), dead, low))
    t = pd.DataFrame(rows, columns=["起始年", "3年後", "5年後", "第幾個月領光", "最低"])
    dead = t[t["第幾個月領光"].notna()]
    def w(x): return "—" if x is None or pd.isna(x) else f"{x/10000:,.0f}萬"
    out.append(f"\n## 本金 {cap//10000} 萬（每年領 60 萬＝本金 {600000/cap:.0%}）")
    out.append(f"- 11 個起點中 {len(dead)} 次在途中領光" + (f"（最快第 {int(dead['第幾個月領光'].min())} 個月）" if len(dead) else ""))
    t3 = t["3年後"].dropna(); t5 = t["5年後"].dropna()
    out.append(f"- 3 年後剩：最差 {w(t3.min())}、中位數 {w(t3.median())}、最好 {w(t3.max())}（{len(t3)} 個起點）")
    out.append(f"- 5 年後剩：最差 {w(t5.min())}、中位數 {w(t5.median())}（{len(t5)} 個起點）")
    out.append("- 逐年：" + "、".join(f"{r.起始年}起 3年後{w(r['3年後'])}" + (f"(第{int(r['第幾個月領光'])}月光)" if pd.notna(r['第幾個月領光']) else "") for _, r in t.iterrows()))
txt = "\n".join(out)
open("research/withdraw.txt", "w").write(txt)
print(txt)
