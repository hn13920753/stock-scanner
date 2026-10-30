"""單一股票：綠K碰五日線→帶量紅K→第三天開盤買，不停損、不限天數，一直抱到淨賺 3% 或 5% 才賣。
對照：同一檔股票「任何一天開盤買」用同樣賣法。價格用 Yahoo 原始價。"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

code = sys.argv[1] if len(sys.argv) > 1 else "2344"
h = yf.Ticker(code + ".TW").history(period="11y", auto_adjust=False, actions=False)
h = h[h["Volume"] > 0]
d = pd.DataFrame({"O": h["Open"], "H": h["High"], "L": h["Low"], "C": h["Close"], "V": h["Volume"] / 1000})
d.index = pd.to_datetime(d.index).tz_localize(None).strftime("%Y-%m-%d")
d["MA5"] = d.C.rolling(5).mean()
d["VOL5"] = d.V.rolling(5).mean()
FEE, TAX = 0.001425, 0.003
O, H, L, C = (d[k].to_numpy() for k in "OHLC")
idx, n = list(d.index), len(d)

green = (d.C < d.O) & (d.L <= d.MA5) & (d.MA5 <= d.H)
red = (d.C > d.O) & (d.V > d.VOL5)
sig = (green.shift(1, fill_value=False) & red).to_numpy()


def run(e, tgt):
    px = O[e]
    cost = px * (1 + FEE)
    tp = cost * (1 + tgt) / (1 - FEE - TAX)
    lo = px
    for j in range(e, n):
        lo = min(lo, L[j])
        if H[j] >= tp:
            ret = tgt if O[j] < tp or j == e else O[j] * (1 - FEE - TAX) / cost - 1
            return dict(進場日=idx[e], 進場價=round(px, 2), 停利價=round(tp, 2), 出場日=idx[j], 結果="達成",
                        抱幾天=j - e + 1, 淨報酬=round(ret * 100, 2), 最大浮虧=round((lo / px - 1) * 100, 1))
    return dict(進場日=idx[e], 進場價=round(px, 2), 停利價=round(tp, 2), 出場日="", 結果="還沒達成(仍持有)",
                抱幾天=n - e, 淨報酬=round((C[-1] * (1 - FEE - TAX) / cost - 1) * 100, 2), 最大浮虧=round((lo / px - 1) * 100, 1))


out = [f"# {code}｜不停損、不限天數，抱到淨賺才賣（資料 {idx[0]}～{idx[-1]}，原始價，扣手續費 0.1425%×2、證交稅 0.3%）"]
for tgt in (0.03, 0.05):
    rows = [run(i + 1, tgt) for i in range(6, n - 1) if sig[i]]
    t = pd.DataFrame(rows)
    t.to_csv(f"research/hold_{code}_{int(tgt*100)}.csv", index=False, encoding="utf-8-sig")
    base = pd.DataFrame([run(e, tgt) for e in range(6, n)])
    for name, s in (("你的型態", t), ("對照：任何一天買", base)):
        ok = s[s["結果"] == "達成"]
        op = s[s["結果"] != "達成"]
        dd = s["最大浮虧"]
        out.append(f"## 淨賺 {tgt:.0%} 才賣｜{name}：{len(s)} 次")
        out.append(f"- 達成 {len(ok)} 次（{len(ok)/len(s):.0%}）；到今天還沒達成 {len(op)} 次")
        out.append(f"- 達成要抱：平均 {ok['抱幾天'].mean():.1f} 天、中位數 {ok['抱幾天'].median():.0f} 天、"
                   f"超過 20 天 {(ok['抱幾天']>20).mean():.0%}、超過 60 天 {(ok['抱幾天']>60).mean():.0%}、最久 {ok['抱幾天'].max()} 天")
        out.append(f"- 期間最大浮虧：中位數 {dd.median():.1f}%｜曾虧超過 10% 的 {(dd<=-10).mean():.0%}｜超過 20% 的 {(dd<=-20).mean():.0%}｜最慘 {dd.min():.1f}%")
        if len(op):
            out.append(f"- 還沒達成的：{'、'.join(f'{r.進場日} 買 {r.進場價}（已抱 {r.抱幾天} 天，現在 {r.淨報酬:+.1f}%）' for r in op.itertuples())[:600]}")
        if name == "你的型態":
            out.append("### 抱最久的 8 筆")
            out.append(ok.sort_values("抱幾天", ascending=False).head(8).to_string(index=False))
            out.append("### 浮虧最深的 8 筆")
            out.append(s.sort_values("最大浮虧").head(8).to_string(index=False))
txt = "\n".join(out)
open(f"research/hold_{code}.txt", "w").write(txt)
print(txt)
