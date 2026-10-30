"""單一股票、指定期間：綠K碰五日線→帶量紅K→第三天開盤買（不停損）。
賣法 A：10 個交易日內淨賺 3% 就賣，沒到第 10 天收盤賣。
賣法 B：不限天數，抱到淨賺 3%（或 5%）才賣。
對照：同期間任何一天開盤買、同樣賣法。原始價；扣手續費 0.1425%×2、證交稅 0.3%。"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

code, start = sys.argv[1], sys.argv[2]
mkt = sys.argv[3] if len(sys.argv) > 3 else "TW"
h = yf.Ticker(f"{code}.{mkt}").history(period="3y", auto_adjust=False, actions=False)
h = h[h["Volume"] > 0]
d = pd.DataFrame({"O": h["Open"], "H": h["High"], "L": h["Low"], "C": h["Close"], "V": h["Volume"] / 1000})
d.index = pd.to_datetime(d.index).tz_localize(None).strftime("%Y-%m-%d")
d["MA5"] = d.C.rolling(5).mean()
d["VOL5"] = d.V.rolling(5).mean()
low9, high9 = d.L.rolling(9).min(), d.H.rolling(9).max()
rsv = ((d.C - low9) / (high9 - low9) * 100).fillna(50).to_numpy()
K = np.zeros(len(d)); D = np.zeros(len(d)); k = dd = 50.0
for i, r in enumerate(rsv):
    k = k * 2 / 3 + r / 3; dd = dd * 2 / 3 + k / 3; K[i], D[i] = k, dd
FEE, TAX = 0.001425, 0.003
O, H, L, C = (d[c].to_numpy() for c in "OHLC")
idx, n = list(d.index), len(d)
green = ((d.C < d.O) & (d.L <= d.MA5) & (d.MA5 <= d.H)).to_numpy()
red = ((d.C > d.O) & (d.V > d.VOL5)).to_numpy()
first = next(i for i in range(n) if idx[i] >= start)


def trade(e, tgt, limit):
    px = O[e]; cost = px * (1 + FEE); tp = cost * (1 + tgt) / (1 - FEE - TAX); lo = px
    last = n - 1 if limit is None else min(e + limit - 1, n - 1)
    for j in range(e, last + 1):
        lo = min(lo, L[j])
        if H[j] >= tp:
            ret = tgt if (O[j] < tp or j == e) else O[j] * (1 - FEE - TAX) / cost - 1
            return "成功", j - e + 1, ret * 100, (lo / px - 1) * 100, idx[j]
    ret = (C[last] * (1 - FEE - TAX) / cost - 1) * 100
    if limit is not None and last == e + limit - 1:
        return "失敗(第10天賣)", limit, ret, (lo / px - 1) * 100, idx[last]
    return "還在抱", last - e + 1, ret, (lo / px - 1) * 100, ""


sigs = [i + 1 for i in range(max(first, 6), n - 1) if green[i - 1] and red[i]]
alld = list(range(max(first, 6), n))
out = [f"# {code}｜{idx[first]}～{idx[-1]}｜綠K碰五日線→帶量紅K→第三天開盤買｜不停損",
       f"期間股價：{C[first]:.1f} → {C[-1]:.1f}（{C[-1]/C[first]-1:+.0%}），期間最高 {H[first:].max():.1f}、最低 {L[first:].min():.1f}",
       f"訊號 {len(sigs)} 次；同期間交易日 {len(alld)} 天"]
rows = []
for e in sigs:
    i = e - 1
    a = trade(e, 0.03, 10); b3 = trade(e, 0.03, None); b5 = trade(e, 0.05, None)
    rows.append({"綠K日": idx[i - 1], "紅K日": idx[i], "進場日": idx[e], "進場價": round(O[e], 2),
                 "A_10天內3%": a[0], "A_天數": a[1], "A_淨報酬%": round(a[2], 2), "A_最大浮虧%": round(a[3], 1),
                 "B_抱到3%": b3[0], "B3_天數": b3[1], "B3_出場日": b3[4], "B3_最大浮虧%": round(b3[3], 1),
                 "B_抱到5%": b5[0], "B5_天數": b5[1], "B5_最大浮虧%": round(b5[3], 1), "B5_目前報酬%": round(b5[2], 2),
                 "紅K日K": round(K[i], 1), "紅K日D": round(D[i], 1)})
t = pd.DataFrame(rows)
t.to_csv(f"research/period_{code}.csv", index=False, encoding="utf-8-sig")


def summ(es, label):
    A = [trade(e, 0.03, 10) for e in es]; A = [x for x in A if x[0] != "還在抱"]
    B3 = [trade(e, 0.03, None) for e in es]; B5 = [trade(e, 0.05, None) for e in es]
    def s(lst, tag):
        ok = [x for x in lst if x[0] == "成功"]; op = [x for x in lst if x[0] == "還在抱"]
        days = np.array([x[1] for x in ok]) if ok else np.array([0])
        dd = np.array([x[3] for x in lst])
        r = f"  - {tag}：{len(lst)} 次，成功 {len(ok)}（{len(ok)/max(len(lst),1):.0%}）"
        if tag.startswith("A"):
            r += f"｜每筆平均 {np.mean([x[2] for x in lst]):+.2f}%｜失敗平均 {np.mean([x[2] for x in lst if x[0]!='成功'] or [0]):+.2f}%"
        else:
            r += f"｜還在抱 {len(op)} 次（目前平均 {np.mean([x[2] for x in op] or [0]):+.1f}%）｜成功的抱 中位數 {np.median(days):.0f} 天、最久 {days.max()} 天"
        r += f"｜中間曾虧 >10% {np.mean(dd<=-10):.0%}、>20% {np.mean(dd<=-20):.0%}、最慘 {dd.min():.1f}%"
        return r
    return [f"- {label}", s(A, "A 10天內賺3%（沒到第10天賣）"), s(B3, "B 抱到賺3%"), s(B5, "B 抱到賺5%")]

out += summ(sigs, "你的型態") + summ(alld, "對照：任何一天買")
out.append("## 加 KD")
for lab, m in (("紅K當天 K>D", t["紅K日K"] > t["紅K日D"]), ("紅K當天 K≤D", t["紅K日K"] <= t["紅K日D"])):
    s = t[m & (t["A_10天內3%"] != "還在抱")]
    if len(s):
        out.append(f"- {lab}：{len(s)} 次，10天內賺3% 成功 {(s['A_10天內3%']=='成功').mean():.0%}，每筆平均 {s['A_淨報酬%'].mean():+.2f}%")
out.append("## 逐筆")
out.append(t.to_string(index=False))
txt = "\n".join(out)
open(f"research/period_{code}.txt", "w").write(txt)
print(txt)
