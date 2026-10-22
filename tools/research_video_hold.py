"""
影片買點・不限持有天數的賣法比較（2026-10-22 第三次，依需求 v2.0「獲利要有彈性」）。
只做統計，不改規則。輸出 research/video_hold.txt。

進場：同 research_video_pullback.py 的影片訊號（多頭回檔、不破前低與月線、價漲量增線實站5均），訊號當天收盤買。
對照：多頭且站上月線的任一天收盤買。
賣法（都沒有 15 天限制；只設 250 個交易日安全上限，到了以收盤賣）：
  固定5%／固定10%／固定20%：扣費淨賺到就賣
  分批：一半淨賺 5% 先賣，另一半用「從最高收盤回落 10%」賣
  移動停利10%：不設停利，從持有期間最高收盤回落 10% 收盤賣
  跌破月線：收盤跌破 MA20 隔天開盤賣（趨勢持有）
  賺10%後破5日線：曾淨賺 10% 之後，收盤跌破 MA5 隔天開盤賣
停損（三選一，與賣法並用）：回檔低點但最多 −10%／固定 −20%／不設停損
每筆 20 萬、手續費 0.1425%×2、證交稅 0.3%、滑價每邊 0.1%、還原權息價、同一檔賣出前不重複買。
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_src = (ROOT / "tools" / "research_video_pullback.py").read_text(encoding="utf-8").split("def run(")[0]
exec(compile(_src, "research_video_pullback_head", "exec"))  # 取得 fetch()、features()、WL、NAMES、常數

OUT = Path(os.environ.get("VP_OUT", ROOT / "research"))
CAP = 250
BC, SC = FEE + SLIP, FEE + TAX + SLIP
EXITS = ["固定5%", "固定10%", "固定20%", "分批5%+移動10%", "移動停利10%", "跌破月線", "賺10%後破5日線"]
STOPS = ["回檔低點最多−10%", "固定−20%", "不停損"]


def net(px, x):
    return x * (1 - SC) / (px * (1 + BC)) - 1


def trade(d, i, ex, st, A):
    O, H, L, C, M5, M20, PBL = A
    n = len(C)
    px = C[i]
    if st == "回檔低點最多−10%":
        sl = max(min(L[i], PBL[i] if not np.isnan(PBL[i]) else L[i]), px * 0.90)
    elif st == "固定−20%":
        sl = px * 0.80
    else:
        sl = -1.0
    tgt = {"固定5%": 0.05, "固定10%": 0.10, "固定20%": 0.20}.get(ex)
    tp = px * (1 + BC) * (1 + tgt) / (1 - SC) if tgt else None
    tp5 = px * (1 + BC) * 1.05 / (1 - SC)
    tp10 = px * (1 + BC) * 1.10 / (1 - SC)
    peak = C[i]
    half = None          # 分批：前半出場價
    hit10 = False
    last = min(i + CAP, n - 1)
    if i + 1 > n - 1:
        return None
    for j in range(i + 1, last + 1):
        # 停損（先檢查，保守）
        if sl > 0 and L[j] <= sl:
            x = O[j] if O[j] <= sl else sl
            r = net(px, x) if half is None else 0.5 * net(px, half) + 0.5 * net(px, x)
            return r, j, "停損", (L[i + 1:j + 1].min() / px - 1), (H[i + 1:j + 1].max() / px - 1)
        if tp is not None and H[j] >= tp:
            x = max(O[j], tp)
            return net(px, x), j, "停利", (L[i + 1:j + 1].min() / px - 1), (H[i + 1:j + 1].max() / px - 1)
        if ex == "分批5%+移動10%" and half is None and H[j] >= tp5:
            half = max(O[j], tp5)
            peak = max(peak, C[j])
            continue
        peak = max(peak, C[j])
        sell_next = False
        if ex == "移動停利10%" or (ex == "分批5%+移動10%" and half is not None):
            sell_next = C[j] <= peak * 0.90
        elif ex == "跌破月線":
            sell_next = C[j] < M20[j]
        elif ex == "賺10%後破5日線":
            hit10 = hit10 or H[j] >= tp10
            sell_next = hit10 and C[j] < M5[j]
        if sell_next:
            if j + 1 > n - 1:
                return None
            x = O[j + 1]
            r = net(px, x) if half is None else 0.5 * net(px, half) + 0.5 * net(px, x)
            return r, j + 1, "訊號出場", (L[i + 1:j + 2].min() / px - 1), (H[i + 1:j + 2].max() / px - 1)
    if last < i + CAP:      # 資料結束還沒賣 → 持有中，不計
        return None
    x = C[last]
    r = net(px, x) if half is None else 0.5 * net(px, half) + 0.5 * net(px, x)
    return r, last, "到期250天", (L[i + 1:last + 1].min() / px - 1), (H[i + 1:last + 1].max() / px - 1)


def run_all(col, ex, st):
    rows = []
    for c, d in data.items():
        A = tuple(d[k].to_numpy() for k in ("open", "high", "low", "close", "ma5", "ma20", "pb_low"))
        sig = d[col].to_numpy()
        i, n = 80, len(d)
        while i < n:
            if not sig[i]:
                i += 1
                continue
            t = trade(d, i, ex, st, A)
            if t is None:
                break
            r, j, how, mae, mfe = t
            rows.append({"代號": c, "日": d.index[i], "報酬": r * 100, "天": j - i, "方式": how,
                         "MAE": mae * 100, "MFE": mfe * 100})
            i = j + 1
    return pd.DataFrame(rows)


def stats(t):
    if not len(t):
        return None
    r = t["報酬"]
    g, l_ = r[r > 0].sum(), -r[r <= 0].sum()
    yr = t.groupby(t["日"].dt.year)["報酬"].mean()
    return dict(n=len(t), w=(r > 0).mean() * 100, avg=r.mean(), tot=(r / 100 * BUDGET).sum(),
                pf=g / l_ if l_ > 0 else float("inf"), days=t["天"].mean(),
                eff=r.sum() / t["天"].sum() * 250 if t["天"].sum() else 0,
                h5=(t["MFE"] >= 5).mean() * 100, h10=(t["MFE"] >= 10).mean() * 100, h20=(t["MFE"] >= 20).mean() * 100,
                mae=-t["MAE"].quantile(0.1), worst=r.min(), best=r.max(),
                yp=int((yr > 0).sum()), yn=len(yr),
                a=r[t["日"] < "2023-01-01"].mean(), b=r[t["日"] >= "2023-01-01"].mean())


def line(lbl, s):
    if s is None:
        return f"  {lbl:<30} 0 筆"
    return (f"  {lbl:<30} {s['n']:>4}筆 成功率{s['w']:4.0f}% 平均{s['avg']:+6.2f}% 合計{s['tot']:+11,.0f} 獲利因子{s['pf']:4.2f} "
            f"抱{s['days']:5.1f}天 每抱一年約{s['eff']:+6.1f}% ｜曾達+5% {s['h5']:3.0f}% +10% {s['h10']:3.0f}% +20% {s['h20']:3.0f}% "
            f"｜一成曾浮虧≥{s['mae']:4.1f}% 最差{s['worst']:+6.1f}% 最好{s['best']:+6.1f}% ｜正年{s['yp']}/{s['yn']} 16-22 {s['a']:+.2f}% 23- {s['b']:+.2f}%")


t0 = time.time()
data = {}
for c, nme, m in WL:
    df = fetch(c, m)
    if df is not None:
        data[c] = features(df)
print(f"抓到 {len(data)} 檔，{time.time() - t0:.0f} 秒")

L_ = ["影片買點・不限持有天數的賣法比較（每筆 20 萬、扣費滑價、還原權息、250 天安全上限）",
      "「每抱一年約」＝總報酬 ÷ 總持有天數 × 250，用來比較資金效率（抱越久的方法要賺更多才划算）", ""]
best = []
for col, title in (("signal", "影片訊號"), ("base", "對照：多頭任一天")):
    L_.append(f"==================== {title} ====================")
    for st in STOPS:
        for ex in EXITS:
            s = stats(run_all(col, ex, st))
            L_.append(line(f"{ex}｜{st}", s))
            if s and col == "signal":
                best.append((s["avg"], s["eff"], ex, st, s))
        L_.append("")
L_.append("==================== 影片訊號：依平均每筆報酬排序前 8 ====================")
for avg, eff, ex, st, s in sorted(best, key=lambda x: -x[0])[:8]:
    L_.append(line(f"{ex}｜{st}", s))
L_.append("")
L_.append("==================== 影片訊號：依資金效率（每抱一年）排序前 8 ====================")
for avg, eff, ex, st, s in sorted(best, key=lambda x: -x[1])[:8]:
    L_.append(line(f"{ex}｜{st}", s))
txt = "\n".join(L_)
print(txt)
(OUT / "video_hold.txt").write_text(txt, encoding="utf-8")
