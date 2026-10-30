"""AI 產業 40 檔，2025-01～今：新規則（前一天綠K不管位置＋紅K被五日線穿過且收盤站上五日線）→ 第三天開盤買，不停損。
賣法 A：10 個交易日內淨賺 3% 賣，沒到第 10 天收盤賣。賣法 B：抱到淨賺 3% 才賣（不限天數）。
帳戶模擬：每筆 5 萬或 10 萬，同時最多 3 或 5 筆；同一天訊號太多時隨機挑（跑 20 次取平均）；損益記在出場月份。
對照：同一批股票任何一天買。原始價；扣手續費 0.1425%×2、證交稅 0.3%。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sectors import SECTORS  # noqa: E402

START = "2025-01-01"
FEE, TAX = 0.001425, 0.003
pool = {}
for items in SECTORS["AI 產業"].values():
    for c, n, m in items:
        pool.setdefault(c, (n, m))


def load(code, mkt):
    for suf in (mkt, "TWO" if mkt == "TW" else "TW"):
        try:
            h = yf.Ticker(f"{code}.{suf}").history(period="3y", auto_adjust=False, actions=False)
        except Exception:  # noqa: BLE001
            continue
        if h is None or len(h) < 100:
            continue
        h = h[h["Volume"] > 0]
        d = pd.DataFrame({"O": h["Open"], "H": h["High"], "L": h["Low"], "C": h["Close"], "V": h["Volume"] / 1000})
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d["MA5"] = d.C.rolling(5).mean()
        return d
    return None


def trade(O, H, L, C, e, limit):
    n = len(O); px = O[e]; cost = px * (1 + FEE); tp = cost * 1.03 / (1 - FEE - TAX); lo = px
    last = n - 1 if limit is None else min(e + limit - 1, n - 1)
    for j in range(e, last + 1):
        lo = min(lo, L[j])
        if H[j] >= tp:
            ret = 0.03 if (O[j] < tp or j == e) else O[j] * (1 - FEE - TAX) / cost - 1
            return "成功", j, ret, lo / px - 1
    ret = C[last] * (1 - FEE - TAX) / cost - 1
    if limit is not None and last == e + limit - 1:
        return "失敗", last, ret, lo / px - 1
    return "未完", last, ret, lo / px - 1


rows, base, fail = [], [], []
for code, (name, mkt) in pool.items():
    d = load(code, mkt)
    if d is None:
        fail.append(code); continue
    O, H, L, C, M = (d[k].to_numpy() for k in ("O", "H", "L", "C", "MA5"))
    dt = d.index; n = len(d)
    first = max(6, int(np.searchsorted(dt, pd.Timestamp(START))))
    for i in range(first, n - 1):
        e = i + 1
        for mode, lim in (("A", 10), ("B", None)):
            r = trade(O, H, L, C, e, lim)
            rec = dict(代碼=code, 名稱=name, 模式=mode, 紅K日=dt[i], 進場日=dt[e], 出場日=dt[r[1]], 進場價=round(O[e], 2),
                       結果=r[0], 抱幾天=r[1] - e + 1, 淨報酬=r[2], 最大浮虧=r[3])
            base.append(rec)
            if C[i - 1] < O[i - 1] and C[i] > O[i] and L[i] <= M[i] <= C[i] * 10 and C[i] >= M[i] and L[i] <= M[i]:
                rows.append(rec)
t = pd.DataFrame(rows); b = pd.DataFrame(base)
t.to_csv("research/ai_newrule_trades.csv", index=False, encoding="utf-8-sig")

out = [f"# AI 產業 {len(pool)} 檔｜{START}～今｜新規則、第三天開盤買、不停損",
       f"抓到 {len(pool)-len(fail)} 檔；失敗 {fail or '無'}"]


def summ(x, lab):
    done = x[x["結果"] != "未完"]
    s = (f"- {lab}：{len(x)} 次｜成功 {(done['結果']=='成功').mean():.0%}｜賺錢機率 {(done['淨報酬']>0).mean():.0%}"
         f"｜每筆平均 {done['淨報酬'].mean():+.2%}｜失敗平均 {done.loc[done['結果']!='成功','淨報酬'].mean():+.1%}"
         f"｜最慘 {done['淨報酬'].min():+.1%}｜中間曾虧>10% {(x['最大浮虧']<=-0.10).mean():.0%}")
    if (x["結果"] == "未完").any():
        u = x[x["結果"] == "未完"]
        s += f"｜還沒賣 {len(u)} 筆（目前平均 {u['淨報酬'].mean():+.1%}）"
    ok = done[done["結果"] == "成功"]
    s += f"｜成功的抱 中位數 {ok['抱幾天'].median():.0f} 天、最久 {ok['抱幾天'].max()} 天"
    return s


for mode, lab in (("A", "A：10 天內賺 3%，沒到第 10 天賣"), ("B", "B：抱到賺 3% 才賣")):
    out.append(f"\n## {lab}")
    out.append(summ(t[t["模式"] == mode], "新規則"))
    out.append(summ(b[b["模式"] == mode], "對照：任何一天買"))

out.append("\n## 各股（A 賣法，每筆平均，由差到好）")
ga = t[(t["模式"] == "A") & (t["結果"] != "未完")].groupby("名稱")["淨報酬"].agg(["count", "mean"]).sort_values("mean")
out.append("、".join(f"{k} {int(v['count'])}次 {v['mean']:+.1%}" for k, v in ga.iterrows()))


def account(x, per, slots, seed):
    rng = np.random.default_rng(seed)
    x = x.assign(_r=rng.random(len(x))).sort_values(["進場日", "_r"])
    held, taken = [], []
    for r in x.itertuples():
        held = [h for h in held if h >= r.進場日]
        if len(held) >= slots:
            continue
        held.append(r.出場日); taken.append(r)
    a = pd.DataFrame(taken)
    a["損益"] = a["淨報酬"] * per
    months = pd.period_range(START[:7], x["進場日"].max().strftime("%Y-%m"), freq="M")
    done = a[a["結果"] != "未完"]
    m = done.groupby(done["出場日"].dt.to_period("M"))["損益"].sum().reindex(months, fill_value=0)
    cnt = a.groupby(a["進場日"].dt.to_period("M")).size().reindex(months, fill_value=0)
    open_pl = a.loc[a["結果"] == "未完", "損益"].sum()
    return m, cnt, open_pl, len(a)


out.append("\n## 帳戶模擬（每個月損益；20 次隨機挑選取平均）")
for mode, lab in (("A", "A 10天內3%"), ("B", "B 抱到3%")):
    x = t[t["模式"] == mode]
    for per in (50_000, 100_000):
        for slots in (3, 5):
            res = [account(x, per, slots, s) for s in range(20)]
            mm = pd.concat([r[0] for r in res], axis=1).mean(axis=1)
            cc = pd.concat([r[1] for r in res], axis=1).mean(axis=1)
            op = np.mean([r[2] for r in res])
            out.append(f"- {lab}｜每筆 {per//10000} 萬｜同時 {slots} 筆（最多用 {per*slots//10000} 萬）："
                       f"每月平均 {mm.mean():+,.0f}｜中位數 {mm.median():+,.0f}｜≥3萬的月份 {(mm>=30000).mean():.0%}"
                       f"｜賠錢月份 {(mm<0).mean():.0%}｜最好月 {mm.max():+,.0f}｜最差月 {mm.min():+,.0f}"
                       f"｜總損益 {mm.sum():+,.0f}｜每月進場 {cc.mean():.1f} 筆｜還沒賣的帳面 {op:+,.0f}")
    if mode == "A":
        res = [account(x, 50_000, 5, s) for s in range(20)]
        mm = pd.concat([r[0] for r in res], axis=1).mean(axis=1)
        out.append("  每月明細（A｜5萬｜5筆）：" + "、".join(f"{p.strftime('%y/%m')} {v/1000:+.1f}k" for p, v in mm.items()))
txt = "\n".join(out)
open("research/ai_newrule.txt", "w").write(txt)
print(txt)
