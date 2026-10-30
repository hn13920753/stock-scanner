"""綜合機制研究（2026-10-30）：把目前所有方法放在同一個帳戶裡比較，找「更可靠」的組合。
股票池：⭐自選股＋AI 產業（去重 52 檔）＋0050；Yahoo 還原權息價；約 10 年。
成本：手續費 0.1425%×2、證交稅（股票 0.3%／ETF 0.1%）、滑價每邊 0.1%。

進場方法（訊號日收盤後判斷，隔天開盤買）：
  M1 綠K紅K（新規則）：前一天綠K＋紅K被五日線穿過且收盤站上 → 淨賺 3%／最多 10 天／不停損
  M2 回檔布局：收盤距近 60 日最高點跌 ≥35% → 淨賺 10%／停損 −20%／最多 60 天（App V1.5 現行）
  M3 回檔＋紅K確認：近 5 天曾跌到距 60 日高 ≥25%，今天出現新規則紅K → 10%／−20%／60 天
  M4 多頭回檔＋紅K：MA20>MA60、收盤>MA60，近 5 天曾距 20 日高回檔 ≥8%，今天新規則紅K → 10%／−10%／60 天
帳戶：50 萬，每筆 10 萬、同時最多 5 筆；同一檔賣出前不重複買；同日訊號太多隨機挑（10 次平均）。
閒置資金：現金不動 或 全部放 0050（要買股時賣 0050）。對照：50 萬全部 0050 買進放著。
期間：全期（2016-06～）、樣本內 2016～2022、樣本外 2023～、AI 行情 2025～。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sectors import SECTORS  # noqa: E402

FEE, SLIP = 0.001425, 0.001
CAP, PER = 500_000, 100_000
pool = {}
for sec in ("⭐ 自選股", "AI 產業"):
    for items in SECTORS[sec].values():
        for c, n, m in items:
            pool.setdefault(c, (n, m))


def grab(tks):
    got = {}
    d = yf.download(tks, period="11y", interval="1d", group_by="ticker", auto_adjust=True,
                    actions=False, threads=True, progress=False)
    for tk in tks:
        try:
            s = d[tk].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            s = s[(s.volume > 0) & (s.low > 0)]
            if len(s) > 200:
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                got[tk.split(".")[0]] = s.astype(float)
        except Exception:  # noqa: BLE001
            pass
    return got


P = grab([f"{c}.{m}" for c, (n, m) in pool.items()] + ["0050.TW"])
miss = [c for c in pool if c not in P]
if miss:
    P.update(grab([f"{c}.{'TWO' if pool[c][1] == 'TW' else 'TW'}" for c in miss]))
print("抓到", len(P), "缺", [c for c in pool if c not in P])


def trades_for(code, d, method):
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(d); dt = d.index
    ma5 = d.close.rolling(5).mean().to_numpy(); ma20 = d.close.rolling(20).mean().to_numpy()
    ma60 = d.close.rolling(60).mean().to_numpy()
    h60 = d.high.rolling(60).max().to_numpy(); h20 = d.high.rolling(20).max().to_numpy()
    dd60 = C / h60 - 1; dd20 = C / h20 - 1
    red = (C > O) & (L <= ma5) & (C >= ma5)
    green_prev = np.r_[False, (C < O)[:-1]]
    newrule = red & green_prev
    min5_dd60 = pd.Series(dd60).rolling(5).min().to_numpy()
    min5_dd20 = pd.Series(dd20).rolling(5).min().to_numpy()
    if method == "M1":
        sig = newrule; tgt, stop, hold = 0.03, None, 10
    elif method == "M2":
        sig = dd60 <= -0.35; tgt, stop, hold = 0.10, 0.20, 60
    elif method == "M3":
        sig = newrule & (min5_dd60 <= -0.25); tgt, stop, hold = 0.10, 0.20, 60
    else:
        sig = newrule & (ma20 > ma60) & (C > ma60) & (min5_dd20 <= -0.08); tgt, stop, hold = 0.10, 0.10, 60
    bc, sc = FEE + SLIP, FEE + 0.003 + SLIP
    out = []
    for i in range(60, n - 1):
        if not sig[i]:
            continue
        e = i + 1; px = O[e]; cost = px * (1 + bc); tp = cost * (1 + tgt) / (1 - sc)
        sp = px * (1 - stop) if stop else -1
        last = min(e + hold - 1, n - 1); ex = None
        for j in range(e, last + 1):
            if j > e and O[j] <= sp: ex = (j, O[j]); break
            if L[j] <= sp: ex = (j, sp); break
            if j > e and O[j] >= tp: ex = (j, O[j]); break
            if H[j] >= tp: ex = (j, tp); break
        if ex is None:
            if e + hold - 1 > n - 1:
                continue  # 尚未結束
            ex = (last, C[last])
        ret = ex[1] * (1 - sc) / cost - 1
        out.append((code, dt[i], dt[e], dt[ex[0]], px, ret))
    return out


M = {}
for m in ("M1", "M2", "M3", "M4"):
    rows = []
    for c in pool:
        if c in P:
            rows += trades_for(c, P[c], m)
    M[m] = pd.DataFrame(rows, columns=["代號", "訊號日", "進場日", "出場日", "進場價", "報酬"])
    M[m].to_csv(f"research/combo_{m}.csv", index=False, encoding="utf-8-sig")

c50 = P["0050"]


def account(tr, idle, start, end=None, seed=0):
    cal = c50.index[(c50.index >= pd.Timestamp(start)) & ((c50.index <= pd.Timestamp(end)) if end else True)]
    rng = np.random.default_rng(seed)
    by = {}
    for r in tr.itertuples():
        by.setdefault(r.進場日, []).append(r)
    cash, u50 = float(CAP), 0.0
    if idle:
        u50 = cash * (1 - FEE - SLIP) / c50.open.loc[cal[0]]; cash = 0.0
    pos, held, eq, n_tr = [], set(), [], 0
    for day in cal:
        o50, p50 = c50.open.loc[day], c50.close.loc[day]
        for p in list(pos):
            if day >= p["exit"]:
                val = PER * (1 + p["ret"]); pos.remove(p); held.discard(p["code"])
                if idle: u50 += val * (1 - FEE - SLIP) / p50
                else: cash += val
        c = [r for r in by.get(day, []) if r.代號 not in held]
        rng.shuffle(c)
        for r in c:
            if len(pos) >= CAP // PER: break
            if idle:
                need = PER / (1 - FEE - 0.001 - SLIP)
                if u50 * o50 < need: break
                u50 -= need / o50
            else:
                if cash < PER: break
                cash -= PER
            pos.append({"code": r.代號, "ret": r.報酬, "exit": r.出場日, "px": r.進場價, "cl": P[r.代號].close})
            held.add(r.代號); n_tr += 1
        mark = sum(PER * p["cl"].asof(day) / p["px"] for p in pos)
        eq.append(cash + u50 * p50 + mark)
    s = pd.Series(eq, index=cal)
    yrs = (cal[-1] - cal[0]).days / 365.25
    ann = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    mdd = (s / s.cummax() - 1).min()
    mon = s.resample("ME").last().diff().fillna(s.resample("ME").last().iloc[0] - s.iloc[0])
    return ann, mdd, mon, n_tr / max(len(mon), 1)


def hold50(start, end=None):
    s = c50.close[(c50.index >= pd.Timestamp(start)) & ((c50.index <= pd.Timestamp(end)) if end else True)] * CAP / c50.open[c50.index >= pd.Timestamp(start)].iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    mon = s.resample("ME").last().diff().fillna(s.resample("ME").last().iloc[0] - CAP)
    return (s.iloc[-1] / CAP) ** (1 / yrs) - 1, (s / s.cummax() - 1).min(), mon


NAMES = {"M1": "綠K紅K（3%／10天／不停損）", "M2": "回檔布局 跌35%（10%／−20%／60天）",
         "M3": "回檔25%＋紅K確認（10%／−20%／60天）", "M4": "多頭回檔8%＋紅K（10%／−10%／60天）"}
L = ["# 綜合機制研究（50 萬帳戶，每筆 10 萬、最多 5 筆）", f"股票池 {len(pool)} 檔（自選＋AI 產業）＋0050；還原權息；扣手續費、證交稅、滑價"]
L.append("\n## 一、逐筆統計（每筆平均淨報酬｜成功＝賺錢比例）")
for m, t in M.items():
    def st(x):
        return f"{len(x)} 筆、賺錢 {(x.報酬>0).mean():.0%}、每筆 {x.報酬.mean():+.2%}、最差 {x.報酬.min():+.1%}" if len(x) else "0 筆"
    L.append(f"- {NAMES[m]}：全期 {st(t)}｜2016～22 {st(t[t.進場日<'2023'])}｜2023～ {st(t[t.進場日>='2023'])}｜2025～ {st(t[t.進場日>='2025'])}")

L.append("\n## 二、帳戶（年化｜最大回撤｜每月平均損益｜≥3萬月份｜賠錢月份｜最差月｜每月成交）")
for start, end, lab in (("2016-06-01", None, "全期 2016/6～"), ("2016-06-01", "2022-12-31", "樣本內 2016～22"),
                        ("2023-01-01", None, "樣本外 2023～"), ("2025-01-01", None, "AI 行情 2025～")):
    a, d_, mon = hold50(start, end)
    L.append(f"\n### {lab}\n- 對照 0050 全部買進放著：年化 {a:+.1%}｜回撤 {d_:.0%}｜每月平均 {mon.mean():+,.0f}｜≥3萬 {(mon>=30000).mean():.0%}｜賠錢月 {(mon<0).mean():.0%}｜最差月 {mon.min():+,.0f}")
    for m in M:
        for idle in (False, True):
            res = [account(M[m], idle, start, end, s) for s in range(10)]
            ann = np.mean([r[0] for r in res]); mdd = np.mean([r[1] for r in res])
            mon = pd.concat([r[2] for r in res], axis=1).mean(axis=1); ntr = np.mean([r[3] for r in res])
            L.append(f"- {NAMES[m]}｜閒置{'放0050' if idle else '現金'}：年化 {ann:+.1%}｜回撤 {mdd:.0%}｜每月平均 {mon.mean():+,.0f}"
                     f"｜≥3萬 {(mon>=30000).mean():.0%}｜賠錢月 {(mon<0).mean():.0%}｜最差月 {mon.min():+,.0f}｜每月成交 {ntr:.1f}")
txt = "\n".join(L)
open("research/combo.txt", "w").write(txt)
print(txt)
