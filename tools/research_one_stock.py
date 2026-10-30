"""單一股票逐筆列出「綠K碰五日線 → 帶量紅K → 第三天開盤買」，照使用者做法：不停損。
停利：扣手續費(0.1425%×2)與證交稅(0.3%)後淨賺 3%（另記 5%）；最多看 10 個交易日，沒到就第 10 天收盤賣。
每次訊號都單獨計算（不管前一筆有沒有賣掉）。KD 用 (9,3,3)，同看盤軟體 K9／D9。
價格用 Yahoo 原始價（同看盤軟體），除權息日附近可能有缺口。"""
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
low9, high9 = d.L.rolling(9).min(), d.H.rolling(9).max()
rsv = ((d.C - low9) / (high9 - low9) * 100).fillna(50).to_numpy()
K, D = np.zeros(len(d)), np.zeros(len(d))
k = dd = 50.0
for i, r in enumerate(rsv):
    k = k * 2 / 3 + r / 3
    dd = dd * 2 / 3 + k / 3
    K[i], D[i] = k, dd
d["K"], d["D"] = K, D

FEE, TAX = 0.001425, 0.003
O, H, L, C = d.O.to_numpy(), d.H.to_numpy(), d.L.to_numpy(), d.C.to_numpy()
idx = list(d.index)
rows = []
for i in range(6, len(d) - 1):
    g, r = d.iloc[i - 1], d.iloc[i]
    if not (g.C < g.O and g.L <= g.MA5 <= g.H):
        continue
    if not (r.C > r.O and r.V > r.VOL5):
        continue
    e = i + 1
    px = O[e]
    cost = px * (1 + FEE)
    t3 = cost * 1.03 / (1 - FEE - TAX)
    t5 = cost * 1.05 / (1 - FEE - TAX)
    last = min(e + 9, len(d) - 1)
    done = last == e + 9
    hit3 = hit5 = None
    for j in range(e, last + 1):
        if hit3 is None and H[j] >= t3:
            hit3 = j
        if hit5 is None and H[j] >= t5:
            hit5 = j
    if hit3 is not None:
        res3, ret3, days = "成功", 0.03 if O[hit3] < t3 else (O[hit3] * (1 - FEE - TAX) / cost - 1), hit3 - e + 1
    elif done:
        res3, ret3, days = "失敗", C[last] * (1 - FEE - TAX) / cost - 1, 10
    else:
        res3, ret3, days = "未滿10天", C[last] * (1 - FEE - TAX) / cost - 1, last - e + 1
    end = hit3 if hit3 is not None else last
    worst = L[e:end + 1].min() / px - 1
    rows.append({
        "綠K日": idx[i - 1], "紅K日": idx[i], "進場日": idx[e], "進場價(開盤)": round(px, 2),
        "3%停利價": round(t3, 2), "結果(3%)": res3, "第幾天": days, "淨報酬": round(ret3 * 100, 2),
        "期間最大浮虧%": round(worst * 100, 1), "10天內到5%": "是" if hit5 is not None else "否",
        "紅K量/五日均量": round(r.V / r.VOL5, 2), "紅K有下影": "是" if min(r.O, r.C) > r.L else "否",
        "紅K日K": round(r.K, 1), "紅K日D": round(r.D, 1),
        "紅K均價": round((r.H + r.L + r.C) / 3, 2), "第3天買得到均價": "是" if L[e] <= (r.H + r.L + r.C) / 3 else "否",
    })
t = pd.DataFrame(rows)
t.to_csv(f"research/one_{code}.csv", index=False, encoding="utf-8-sig")
f = t[t["結果(3%)"] != "未滿10天"]
out = [f"# {code} 綠K碰五日線→帶量紅K→第三天開盤買｜不停損｜10 天內淨賺 3%（資料 {idx[0]}～{idx[-1]}）",
       f"訊號共 {len(t)} 次（已滿 10 天 {len(f)} 次）"]
def line(name, s):
    if len(s) == 0:
        return f"- {name}：0 次"
    return (f"- {name}：{len(s)} 次｜成功 {(s['結果(3%)']=='成功').sum()} 次 = {(s['結果(3%)']=='成功').mean():.0%}"
            f"｜10天內到5% {(s['10天內到5%']=='是').mean():.0%}｜平均每筆 {s['淨報酬'].mean():+.2f}%"
            f"｜失敗平均 {s.loc[s['結果(3%)']=='失敗','淨報酬'].mean():+.2f}%｜最差 {s['淨報酬'].min():+.2f}%")
out.append(line("全部", f))
out.append("## 加 KD 條件")
out.append(line("紅K當天 K>D", f[f["紅K日K"] > f["紅K日D"]]))
out.append(line("紅K當天 K≤D", f[f["紅K日K"] <= f["紅K日D"]]))
out.append(line("紅K當天 K<50", f[f["紅K日K"] < 50]))
out.append(line("紅K當天 K≥50", f[f["紅K日K"] >= 50]))
out.append(line("K<50 且 K>D", f[(f["紅K日K"] < 50) & (f["紅K日K"] > f["紅K日D"])]))
out.append(line("K≥80（高檔）", f[f["紅K日K"] >= 80]))
out.append(line("紅K有下影", f[f["紅K有下影"] == "是"]))
out.append("## 分年")
for y, s in f.groupby(f["進場日"].str[:4]):
    out.append(line(y, s))
out.append("## 逐筆")
out.append(t.to_string(index=False))
txt = "\n".join(out)
open(f"research/one_{code}.txt", "w").write(txt)
print(txt)
