"""無時間限制版（只看停利／停損）：書本圖例規則回測（2026-10-31）：朱家泓「空轉多打底範例」2-2-14／2-2-15。定義見 book/README.md。
股票池：⭐自選股＋AI 產業（52 檔）＋0050；Yahoo 還原權息價、約 10 年；隔天開盤買。
成本：手續費 0.1425%×2、證交稅 0.3%、滑價每邊 0.1%。樣本內 2016～2022、樣本外 2023～。
規則：R0 綠K紅K新規則（使用者）｜R1 第1支腳反彈（①＋②）｜R2 多頭回檔大量紅K突破前5日收盤高（⑥）｜R3 回檔布局跌35%｜對照 任一天
賣法：E1 淨3%／10天／不停損｜E2 淨5%／20天／不停損｜E3 淨10%／60天／不停損｜E4 淨10%／60天／跌破訊號K最低停損｜E5 淨10%／60天／−20%停損
帳戶：每條規則用「樣本內每筆平均最好」的賣法，50 萬、每筆 10 萬、最多 5 筆，閒置現金或放 0050（10 次隨機平均）。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sectors import SECTORS  # noqa: E402

FEE, SLIP, TAX = 0.001425, 0.001, 0.003
BC, SC = FEE + SLIP, FEE + TAX + SLIP
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


RN = {"R0": "綠K紅K（你的新規則）", "R1": "書①②第1支腳反彈", "R2": "書⑥多頭回檔大量紅K",
      "R3": "回檔布局跌35%", "ANY": "對照：任一天買"}
TGTS = (0.03, 0.05, 0.10)
STOPS = {"siglow": "破訊號K低", 0.05: "−5%", 0.10: "−10%", 0.20: "−20%"}


def signals(d):
    O, H, L, C, V = (d[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    n = len(d)
    ma5 = d.close.rolling(5).mean().to_numpy(); ma20 = d.close.rolling(20).mean().to_numpy()
    ma60 = d.close.rolling(60).mean().to_numpy(); vol5 = d.volume.rolling(5).mean().to_numpy()
    dd60 = C / d.high.rolling(60).max().to_numpy() - 1
    red, black = C > O, C < O
    bigblack = black & (V > vol5)
    two_black = np.r_[False, bigblack[1:] & bigblack[:-1]]          # 第 j 天與前一天都是大量黑K
    tb_recent = pd.Series(two_black).shift(1).rolling(5).max().fillna(0).to_numpy() > 0  # 前 1～5 天內出現過
    prev_h = np.r_[np.nan, H[:-1]]
    hi5c = pd.Series(C).shift(1).rolling(5).max().to_numpy()
    hi10c = pd.Series(C).shift(1).rolling(10).max().to_numpy()
    lo5c = pd.Series(C).shift(1).rolling(5).min().to_numpy()
    ma60_up = ma60 > np.r_[np.full(5, np.nan), ma60[:-5]]
    s = {}
    s["R0"] = red & (L <= ma5) & (C >= ma5) & np.r_[False, black[:-1]]
    s["R1"] = (dd60 <= -0.15) & tb_recent & red & (C > ma5) & (C > prev_h)
    s["R2"] = (ma20 > ma60) & ma60_up & (hi10c >= lo5c * 1.05) & red & (V > vol5) & (C > hi5c)
    s["R3"] = dd60 <= -0.35
    s["ANY"] = np.ones(n, bool)
    for k in s:
        s[k] = np.nan_to_num(s[k]).astype(bool)
        s[k][:60] = False
    return s



def run(code, d, sig, tgt, stop):
    """不限天數：先碰停利或停損才賣；資料結束還沒賣的記為『還在抱』（以最後收盤估帳面）。"""
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    n, dt, out = len(d), d.index, []
    for i in np.flatnonzero(sig):
        e = i + 1
        if e >= n:
            continue
        px = O[e]; cost = px * (1 + BC); tp = cost * (1 + tgt) / (1 - SC)
        sp = L[i] if stop == "siglow" else px * (1 - stop)
        if sp >= px:
            continue
        ex_, why = None, "還在抱"
        for j in range(e, n):
            if j > e and O[j] <= sp: ex_, why = (j, O[j]), "停損"; break
            if L[j] <= sp: ex_, why = (j, sp), "停損"; break
            if j > e and O[j] >= tp: ex_, why = (j, O[j]), "停利"; break
            if H[j] >= tp: ex_, why = (j, tp), "停利"; break
        if ex_ is None:
            ex_ = (n - 1, C[-1])
        out.append((code, dt[i], dt[e], dt[ex_[0]], px, ex_[1] * (1 - SC) / cost - 1, why, ex_[0] - e + 1))
    return out


SIG = {c: signals(P[c]) for c in pool if c in P}
T = {}
for r in RN:
    for tgt in TGTS:
        for stop in STOPS:
            rows = []
            for c, s in SIG.items():
                rows += run(c, P[c], s[r], tgt, stop)
            T[(r, tgt, stop)] = pd.DataFrame(rows, columns=["代號", "訊號日", "進場日", "出場日", "進場價", "報酬", "結果", "天數"])

L = ["# 不限天數（只看停利／停損）｜52 檔＋0050、10 年、還原權息、扣成本",
     "每筆平均＝已賣出的平均淨報酬；成功＝先到停利比例；還在抱＝到今天兩個都還沒碰到"]
for r in RN:
    L.append(f"\n## {RN[r]}")
    for tgt in TGTS:
        for stop in STOPS:
            t = T[(r, tgt, stop)]
            done = t[t.結果 != "還在抱"]; op = t[t.結果 == "還在抱"]
            if len(done) == 0:
                continue
            L.append(f"- 停利{tgt:.0%}｜停損{STOPS[stop]}：{len(t)} 筆｜成功 {(done.結果=='停利').mean():.0%}｜每筆 {done.報酬.mean():+.2%}"
                     f"｜抱 中位數 {done.天數.median():.0f} 天、平均 {done.天數.mean():.0f} 天、最久 {done.天數.max()} 天"
                     f"｜前段 {done[done.進場日<'2023'].報酬.mean():+.2%}｜後段 {done[done.進場日>='2023'].報酬.mean():+.2%}"
                     f"｜還在抱 {len(op)} 筆（帳面 {op.報酬.mean() if len(op) else 0:+.1%}）")

c50 = P["0050"]



def account(tr, idle, start, seed):
    cal = c50.index[c50.index >= pd.Timestamp(start)]
    rng = np.random.default_rng(seed); by = {}
    for r in tr.itertuples():
        by.setdefault(r.進場日, []).append(r)
    cash, u50 = float(CAP), 0.0
    if idle:
        u50 = cash * (1 - FEE - SLIP) / c50.open.loc[cal[0]]; cash = 0.0
    pos, held, eq = [], set(), []
    for day in cal:
        o50, p50 = c50.open.loc[day], c50.close.loc[day]
        for p in list(pos):
            if day >= p["exit"]:
                val = PER * (1 + p["ret"]); pos.remove(p); held.discard(p["code"])
                if idle: u50 += val * (1 - FEE - SLIP) / p50
                else: cash += val
        c = [r for r in by.get(day, []) if r.代號 not in held]; rng.shuffle(c)
        for r in c:
            if len(pos) >= CAP // PER: break
            if idle:
                need = PER / (1 - FEE - 0.001 - SLIP)
                if u50 * o50 < need: break
                u50 -= need / o50
            else:
                if cash < PER: break
                cash -= PER
            pos.append({"code": r.代號, "ret": r.報酬, "exit": r.出場日, "px": r.進場價, "cl": P[r.代號].close}); held.add(r.代號)
        eq.append(cash + u50 * p50 + sum(PER * p["cl"].asof(day) / p["px"] for p in pos))
    s = pd.Series(eq, index=cal); yrs = (cal[-1] - cal[0]).days / 365.25
    return (s.iloc[-1] / CAP) ** (1 / yrs) - 1, (s / s.cummax() - 1).min()



L.append("\n## 帳戶（50 萬、每筆 10 萬、最多 5 筆、閒置放 0050；年化／最大回撤）")
for start in ("2016-06-01", "2025-01-01"):
    s0 = c50.close[c50.index >= start]; yrs = (s0.index[-1] - s0.index[0]).days / 365.25
    L.append(f"\n### 從 {start[:7]} 起｜0050 放著：年化 {(s0.iloc[-1]/s0.iloc[0])**(1/yrs)-1:+.1%}｜回撤 {(s0/s0.cummax()-1).min():.0%}")
    for r in ("R0", "R1", "R3"):
        for tgt, stop in ((0.03, "siglow"), (0.05, 0.10), (0.10, 0.20)):
            t = T[(r, tgt, stop)]
            res = [account(t, True, start, sd) for sd in range(10)]
            L.append(f"- {RN[r]}｜停利{tgt:.0%}／停損{STOPS[stop]}：年化 {np.mean([x[0] for x in res]):+.1%}｜回撤 {np.mean([x[1] for x in res]):.0%}")
txt = "\n".join(L)
open("research/book_nolimit.txt", "w").write(txt)
print(txt)
