"""比較進場規則（不停損、第三天開盤買）：
  舊：綠K被五日線穿過＋紅K量>五日均量
  新：前一天綠K（位置不管）＋紅K被五日線穿過且收盤站上五日線（量不管／量>五日均量）
賣法 A：10 個交易日內淨賺 3% 賣，沒到第 10 天收盤賣。賣法 B：抱到淨賺 3% 才賣。
對照：同期間任何一天開盤買。原始價；扣手續費 0.1425%×2、證交稅 0.3%。"""
import numpy as np
import pandas as pd
import yfinance as yf

FEE, TAX = 0.001425, 0.003
CASES = [("3037", "欣興", "2025-01-01"), ("2344", "華邦電", "2025-01-01"), ("2344", "華邦電", "2016-01-01")]


def load(code):
    h = yf.Ticker(code + ".TW").history(period="11y", auto_adjust=False, actions=False)
    h = h[h["Volume"] > 0]
    d = pd.DataFrame({"O": h["Open"], "H": h["High"], "L": h["Low"], "C": h["Close"], "V": h["Volume"] / 1000})
    d.index = pd.to_datetime(d.index).tz_localize(None).strftime("%Y-%m-%d")
    d["MA5"] = d.C.rolling(5).mean(); d["VOL5"] = d.V.rolling(5).mean()
    return d


def trade(d, e, limit):
    O, H, L, C = (d[c].to_numpy() for c in "OHLC"); n = len(d)
    px = O[e]; cost = px * (1 + FEE); tp = cost * 1.03 / (1 - FEE - TAX); lo = px
    last = n - 1 if limit is None else min(e + limit - 1, n - 1)
    for j in range(e, last + 1):
        lo = min(lo, L[j])
        if H[j] >= tp:
            ret = 3.0 if (O[j] < tp or j == e) else (O[j] * (1 - FEE - TAX) / cost - 1) * 100
            return "成功", j - e + 1, ret, (lo / px - 1) * 100
    ret = (C[last] * (1 - FEE - TAX) / cost - 1) * 100
    if limit is not None and last == e + limit - 1:
        return "失敗", limit, ret, (lo / px - 1) * 100
    return "未完", last - e + 1, ret, (lo / px - 1) * 100


out = ["# 規則比較（不停損、第三天開盤買）"]
allrows = []
for code, name, start in CASES:
    d = load(code); idx = list(d.index); n = len(d)
    g = (d.C < d.O).to_numpy()
    g_touch = ((d.C < d.O) & (d.L <= d.MA5) & (d.MA5 <= d.H)).to_numpy()
    r = (d.C > d.O).to_numpy()
    r_touch_above = ((d.C > d.O) & (d.L <= d.MA5) & (d.C >= d.MA5)).to_numpy()
    vol = (d.V > d.VOL5).to_numpy()
    first = max(6, next(i for i in range(n) if idx[i] >= start))
    rules = {
        "舊：綠K碰五日線＋紅K帶量": lambda i: g_touch[i - 1] and r[i] and vol[i],
        "新：綠K（不管位置）＋紅K碰五日線站上": lambda i: g[i - 1] and r_touch_above[i],
        "新＋紅K帶量": lambda i: g[i - 1] and r_touch_above[i] and vol[i],
        "對照：任何一天買": None,
    }
    out.append(f"\n## {name} {code}｜{idx[first]}～{idx[-1]}｜股價 {d.C.iloc[first]:.1f} → {d.C.iloc[-1]:.1f}")
    for lab, f in rules.items():
        es = list(range(first, n)) if f is None else [i + 1 for i in range(first, n - 1) if f(i)]
        A = [trade(d, e, 10) for e in es]; A = [x for x in A if x[0] != "未完"]
        B = [trade(d, e, None) for e in es]
        if not A:
            out.append(f"- {lab}：0 次"); continue
        a_ok = np.mean([x[0] == "成功" for x in A]); a_pos = np.mean([x[2] > 0 for x in A]); a_avg = np.mean([x[2] for x in A])
        a_fail = [x[2] for x in A if x[0] != "成功"]
        bo = [x for x in B if x[0] == "成功"]; bdays = np.array([x[1] for x in bo]); bdd = np.array([x[3] for x in B])
        out.append(f"- {lab}：{len(A)} 次｜A 10天內賺3% {a_ok:.0%}｜A 賺錢機率 {a_pos:.0%}｜A 每筆平均 {a_avg:+.2f}%"
                   f"｜A 失敗平均 {np.mean(a_fail) if a_fail else 0:+.1f}%、最慘 {min(x[2] for x in A):+.1f}%"
                   f"｜B 抱到3% 達成 {len(bo)}/{len(B)}、中位數 {np.median(bdays):.0f} 天、最久 {bdays.max()} 天、中間曾虧>10% {np.mean(bdd<=-10):.0%}、最慘浮虧 {bdd.min():.0f}%")
        if f is not None and lab.startswith("新："):
            for e in es:
                x = trade(d, e, 10); y = trade(d, e, None)
                allrows.append({"股票": name, "期間起": start, "紅K日": idx[e - 1], "進場日": idx[e], "進場價": round(d.O.iloc[e], 2),
                                "10天內3%": x[0], "天數": x[1], "淨報酬%": round(x[2], 2), "10天內最大浮虧%": round(x[3], 1),
                                "抱到3%天數": y[1] if y[0] == "成功" else "未達成", "抱到3%最大浮虧%": round(y[3], 1),
                                "紅K量/五日均量": round(d.V.iloc[e - 1] / d.VOL5.iloc[e - 1], 2)})
pd.DataFrame(allrows).to_csv("research/rule_compare_trades.csv", index=False, encoding="utf-8-sig")
txt = "\n".join(out)
open("research/rule_compare.txt", "w").write(txt)
print(txt)
