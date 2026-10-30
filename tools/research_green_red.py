"""
使用者規則回測（2026-10-30）：「綠K碰五日線 → 隔天帶量紅K → 第三天進場」。
只做統計，不改兩個 App 的規則。輸出 research/green_red.txt、research/green_red_trades.csv。

使用者原話整理（事前固定，不看結果調整）：
  第 1 天 綠K：收盤 < 開盤；五日線（含當天，同看盤軟體）穿過這根K棒。
      嚴格：最低 ≤ MA5 ≤ 最高。
      寬鬆：K棒在五日線下方但差距不大也算：最高 ≥ MA5×0.98 且 最低 ≤ MA5×1.02。
  第 2 天 紅K：收盤 > 開盤；成交量 > 五日均量（含當天，同看盤軟體 VOL5）；最好有下影線。
  第 3 天 進場。
進場價（三種都測）：
  open  ＝ 第 3 天開盤價
  avg   ＝ 掛「紅K那天的均價」(最高+最低+收盤)/3 限價買；第 3 天最低價碰到才成交，開盤更低用開盤價
  close ＝ 第 3 天收盤價
出場：扣手續費、證交稅後淨賺 3%／5%／10% 停利；停損（無、紅K最低、綠紅兩天最低、−5%、−8%）；
      最多持有 10 或 20 個交易日，到期收盤賣。同一天碰到停利與停損先算停損；跳空用開盤價。
成本：手續費 0.1425%×2（未打折）、證交稅 0.3%、滑價每邊 0.1%。
價格：Yahoo 還原權息價計算。同一檔賣出前不重複進場。
對照組：①任一天第 2 天開盤買（同樣出場）②只要「帶量紅K」不看前一天綠K。
股票：⭐ 自選股 23 檔＋AI 產業 40 檔（去重）。樣本內 2016～2022、樣本外 2023～。
帳戶模擬：每筆 5 萬，同時最多 3 筆（15 萬）或 5 筆（25 萬），看每月損益能否 ≥ 3 萬。
"""
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sectors import SECTORS  # noqa: E402

OUT = Path(os.environ.get("GR_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
FAKE = bool(os.environ.get("GR_FAKE"))
FEE, TAX, SLIP = 0.001425, 0.003, 0.001
BC, SC = FEE + SLIP, FEE + TAX + SLIP
PER = 50_000
SPLIT = pd.Timestamp("2023-01-01")

POOL = {}
for sec in ("⭐ 自選股", "AI 產業"):
    for items in SECTORS[sec].values():
        for c, n, m in items:
            POOL.setdefault(c, (n, m))
CHECK = {"2409": ("友達", "TW")}   # 只用來核對使用者截圖，不納入統計


def fetch(code, mkt):
    if FAKE:
        rng = np.random.default_rng(int(code[:4]))
        idx = pd.bdate_range("2015-10-01", "2026-10-29")
        c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.022, len(idx))))
        o = c * (1 + rng.normal(0, 0.01, len(idx)))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.008, len(idx))))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, len(idx))))
        v = rng.lognormal(10, 0.5, len(idx))
        return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v,
                             "raw_open": o, "raw_high": hi, "raw_low": lo, "raw_close": c}, index=idx)
    import yfinance as yf
    for suf in (f".{mkt}", ".TWO" if mkt == "TW" else ".TW"):
        try:
            h = yf.Ticker(code + suf).history(period="11y", interval="1d", auto_adjust=False, actions=False)
        except Exception:  # noqa: BLE001
            continue
        if h is None or len(h) < 300:
            continue
        h = h.rename(columns=str.lower)
        h = h[(h["volume"] > 0) & (h["low"] > 0)].copy()
        r = h["adj close"] / h["close"]
        df = pd.DataFrame({"open": h["open"] * r, "high": h["high"] * r, "low": h["low"] * r,
                           "close": h["adj close"], "volume": h["volume"] / 1000,
                           "raw_open": h["open"], "raw_high": h["high"], "raw_low": h["low"],
                           "raw_close": h["close"]})
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df.astype(float)
    return None


def features(df):
    d = df.copy()
    O, H, L, C, V = (d[k] for k in ("open", "high", "low", "close", "volume"))
    ma5 = C.rolling(5).mean()
    vol5 = V.rolling(5).mean()
    rng = (H - L).replace(0, np.nan)
    green = C < O
    red = C > O
    strict = (L <= ma5) & (ma5 <= H)
    loose = (H >= ma5 * 0.98) & (L <= ma5 * 1.02)
    d["g_strict"] = (green & strict).shift(1, fill_value=False).astype(bool)
    d["g_loose"] = (green & loose).shift(1, fill_value=False).astype(bool)
    d["red"] = red
    d["vol_up"] = V > vol5
    lower = (np.minimum(O, C) - L) / rng
    d["shadow_any"] = lower > 0
    d["shadow_20"] = lower >= 0.2
    d["avg_px"] = (H + L + C) / 3
    d["low2"] = np.minimum(L, L.shift(1))
    d["ma5"], d["vol5"] = ma5, vol5
    d["ok"] = ma5.notna() & vol5.notna()
    return d


def simulate(d, sig, entry, target, stop, hold, code):
    """sig[i] 成立＝第 i 天是紅K（第 2 天），第 i+1 天進場。"""
    O, H, L, C = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    AVG, LOW2 = d["avg_px"].to_numpy(), d["low2"].to_numpy()
    dates = d.index
    n, out, i = len(d), [], 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        if entry == "open":
            px = O[e]
        elif entry == "close":
            px = C[e]
        else:  # 紅K均價限價
            lim = AVG[i]
            if L[e] > lim:
                i += 1
                continue
            px = min(O[e], lim)
        cost = px * (1 + BC)
        tp = cost * (1 + target) / (1 - SC)
        if stop == "none":
            sp = -np.inf
        elif stop == "redlow":
            sp = L[i]
        elif stop == "low2":
            sp = LOW2[i]
        else:
            sp = px * (1 - stop)
        if sp >= px:            # 進場價已在停損下方，不交易
            i += 1
            continue
        exit_px, x, why = None, None, ""
        first = e + 1 if entry == "close" else e
        last = min(e + hold - 1, n - 1)
        for k in range(first, last + 1):
            if k > e or entry != "avg":
                o = O[k]
            else:
                o = px          # 進場當天限價成交，從成交價起算
            if k > e and o <= sp:
                exit_px, x, why = o, k, "停損"
                break
            if L[k] <= sp:
                exit_px, x, why = sp, k, "停損"
                break
            if entry == "avg" and k == e:
                continue            # 限價成交當天只檢查停損（保守）
            if k > e and o >= tp:
                exit_px, x, why = o, k, "停利"
                break
            if H[k] >= tp:
                exit_px, x, why = tp, k, "停利"
                break
        if exit_px is None:
            if e + hold - 1 > n - 1:
                break           # 資料尾端尚未結束，不計
            x = e + hold - 1
            exit_px, why = C[x], "到期"
        ret = exit_px * (1 - SC) / cost - 1
        mfe = H[e:x + 1].max() * (1 - SC) / cost - 1
        mae = L[e:x + 1].min() / px - 1
        out.append((code, dates[i], dates[e], dates[x], px, exit_px, why, ret, x - e + 1, mfe, mae))
        i = x + 1
    return out


COLS = ["代碼", "紅K日", "進場日", "出場日", "進場價", "出場價", "結果", "淨報酬", "持有天", "MFE", "MAE"]


def stats(t):
    if len(t) == 0:
        return dict(n=0)
    r = t["淨報酬"]
    return dict(n=len(t), win=(t["結果"] == "停利").mean(), pos=(r > 0).mean(), avg=r.mean(),
                aw=r[r > 0].mean() if (r > 0).any() else 0, al=r[r <= 0].mean() if (r <= 0).any() else 0,
                hold=t["持有天"].mean(), worst=r.min(), ins=r[t["進場日"] < SPLIT].mean(),
                oos=r[t["進場日"] >= SPLIT].mean(), n_oos=int((t["進場日"] >= SPLIT).sum()),
                oos_win=(t.loc[t["進場日"] >= SPLIT, "結果"] == "停利").mean() if (t["進場日"] >= SPLIT).any() else np.nan)


def fmt(s):
    if s["n"] == 0:
        return "0 筆"
    return (f"{s['n']:>5} 筆｜先到停利 {s['win']:5.1%}｜淨賺比例 {s['pos']:5.1%}｜每筆 {s['avg']:+6.2%}"
            f"（賺 {s['aw']:+.2%}／賠 {s['al']:+.2%}）｜抱 {s['hold']:4.1f} 天｜最差 {s['worst']:+.1%}"
            f"｜樣本內 {s['ins']:+.2%}｜樣本外 {s['n_oos']} 筆 {s['oos_win']:.0%}／{s['oos']:+.2%}")


def account(t, slots, seed=0):
    """每筆 5 萬、同時最多 slots 筆；同一天多個訊號隨機排序；以出場月份記損益。"""
    rng = np.random.default_rng(seed)
    t = t.sample(frac=1, random_state=seed).sort_values("進場日", kind="stable")
    held, taken = [], []
    for _, r in t.iterrows():
        held = [x for x in held if x >= r["進場日"]]   # 出場當天收盤後釋出，隔天可再用
        if len(held) >= slots:
            continue
        held.append(r["出場日"])
        taken.append(r)
    del rng
    if not taken:
        return None
    a = pd.DataFrame(taken)
    a["損益"] = a["淨報酬"] * PER
    months = pd.period_range(a["進場日"].min().to_period("M"), a["出場日"].max().to_period("M"), freq="M")
    m = a.groupby(a["出場日"].dt.to_period("M"))["損益"].sum().reindex(months, fill_value=0)
    cnt = a.groupby(a["進場日"].dt.to_period("M")).size().reindex(months, fill_value=0)
    return a, m, cnt


def main():
    data, fail = {}, []
    for code, (name, mkt) in {**POOL, **CHECK}.items():
        df = fetch(code, mkt)
        if df is None or len(df) < 120:
            fail.append(code)
            continue
        data[code] = features(df)
    lines = []
    P = lines.append
    P("# 綠K碰五日線 → 帶量紅K → 第三天進場 回測（2026-10-30）")
    P(f"股票池：⭐自選股＋AI 產業 共 {len(POOL)} 檔；成功抓到 {len([c for c in data if c in POOL])} 檔；失敗：{fail or '無'}")
    P("成本：手續費 0.1425%×2（未打折）、證交稅 0.3%、滑價每邊 0.1%；還原權息價；樣本內 2016～2022、樣本外 2023～")
    P("")

    # ---- 1. 使用者截圖核對 ----
    P("## 1. 截圖核對（嚴格／寬鬆五日線、量是否過五日均量、下影線）")
    for code, day in [("2344", "2026-04-08"), ("2344", "2026-04-27"), ("2344", "2026-05-04"), ("2409", "2026-08-28")]:
        d = data.get(code)
        if d is None or pd.Timestamp(day) not in d.index:
            P(f"- {code} {day}：沒有資料")
            continue
        r = d.loc[pd.Timestamp(day)]
        P(f"- {code} 紅K {day}：前日綠K碰MA5 嚴格={bool(r.g_strict)} 寬鬆={bool(r.g_loose)}｜紅K={bool(r.red)}"
          f"｜量 {r.volume:,.0f} vs 五日均量 {r.vol5:,.0f} → {'過' if r.vol_up else '沒過'}"
          f"｜有下影線={bool(r.shadow_any)}｜原始收盤 {r.raw_close:.2f}")
    P("")

    pool = {c: d for c, d in data.items() if c in POOL}

    def collect(sigkey, entry, target, stop, hold):
        rows = []
        for c, d in pool.items():
            sig = sigkey(d).to_numpy() & d["ok"].to_numpy()
            rows += simulate(d, sig, entry, target, stop, hold, c)
        t = pd.DataFrame(rows, columns=COLS)
        return t

    base_sig = {
        "嚴格＋過量": lambda d: d.g_strict & d.red & d.vol_up,
        "嚴格＋沒過量": lambda d: d.g_strict & d.red & ~d.vol_up,
        "嚴格＋不看量": lambda d: d.g_strict & d.red,
        "寬鬆＋過量": lambda d: d.g_loose & d.red & d.vol_up,
        "嚴格＋過量＋有下影": lambda d: d.g_strict & d.red & d.vol_up & d.shadow_any,
        "嚴格＋過量＋下影≥20%": lambda d: d.g_strict & d.red & d.vol_up & d.shadow_20,
        "對照：帶量紅K（不看前一天）": lambda d: d.red & d.vol_up,
        "對照：任一天": lambda d: pd.Series(True, index=d.index),
    }

    # ---- 2. 訊號版本比較（固定出場：開盤買、淨 3%、停損紅K低、10 天） ----
    P("## 2. 訊號條件比較（固定：第三天開盤買｜淨賺 3% 賣｜跌破紅K最低停損｜最多 10 天）")
    for name, f in base_sig.items():
        P(f"- {name:<18} {fmt(stats(collect(f, 'open', 0.03, 'redlow', 10)))}")
    P("")

    # ---- 3. 進場價比較 ----
    P("## 3. 進場價比較（訊號：嚴格＋過量｜淨賺 3%｜紅K最低停損｜10 天）")
    main_sig = base_sig["嚴格＋過量"]
    for ent, lab in [("open", "第三天開盤"), ("avg", "紅K均價限價"), ("close", "第三天收盤")]:
        P(f"- {lab:<8} {fmt(stats(collect(main_sig, ent, 0.03, 'redlow', 10)))}")
    P("")

    # ---- 4. 停利 × 停損 × 天數 全表 ----
    P("## 4. 停利 × 停損 × 持有天數（訊號：嚴格＋過量｜第三天開盤買）；括號內為同出場的『任一天』對照")
    grid = []
    for target, stop, hold in product([0.03, 0.05, 0.10], ["none", "redlow", "low2", 0.05, 0.08], [10, 20]):
        t = collect(main_sig, "open", target, stop, hold)
        b = collect(base_sig["對照：任一天"], "open", target, stop if stop not in ("redlow", "low2") else 0.05, hold)
        s, sb = stats(t), stats(b)
        sl = {"none": "不停損", "redlow": "紅K最低", "low2": "兩天最低", 0.05: "−5%", 0.08: "−8%"}[stop]
        grid.append((target, sl, hold, s, sb, t))
        P(f"- 淨 {target:.0%}｜{sl:<4}｜{hold} 天：{fmt(s)}  （任一天 {sb['win']:.0%}／{sb['avg']:+.2%}）")
    P("")

    # ---- 5. 帳戶模擬 ----
    P("## 5. 帳戶模擬：每筆 5 萬；同時最多 3 筆（15 萬）／5 筆（25 萬）；目標每月 ≥ 3 萬")
    P("（只取 2023 年以後的近期，以及 10 年全期；同一天訊號太多時隨機挑，跑 5 次取平均）")
    best = sorted(grid, key=lambda g: g[3].get("avg", -9) if g[3]["n"] >= 50 else -9, reverse=True)[:3]
    cand = [g for g in grid if (g[0], g[1], g[2]) in {(0.03, "紅K最低", 10), (0.05, "紅K最低", 20)}] + best
    seen = set()
    for target, sl, hold, s, _, t in cand:
        key = (target, sl, hold)
        if key in seen or len(t) == 0:
            continue
        seen.add(key)
        for slots in (3, 5):
            for lab, tt in (("10 年", t), ("2023～", t[t["進場日"] >= SPLIT])):
                res = [account(tt, slots, seed) for seed in range(5)]
                res = [r for r in res if r is not None]
                if not res:
                    continue
                mm = pd.concat([r[1] for r in res], axis=1).mean(axis=1)
                cc = pd.concat([r[2] for r in res], axis=1).mean(axis=1)
                P(f"- 淨 {target:.0%}｜{sl}｜{hold} 天｜{slots} 筆（{slots*5} 萬）｜{lab}：每月平均 {mm.mean():+,.0f} 元"
                  f"｜中位數 {mm.median():+,.0f}｜≥3 萬的月份 {(mm >= 30000).mean():.0%}｜賠錢月份 {(mm < 0).mean():.0%}"
                  f"｜最差月 {mm.min():+,.0f}｜每月進場約 {cc.mean():.1f} 筆")
    P("")
    P("## 6. 3 萬／月需要多少")
    P("- 每筆 5 萬淨賺 3% ＝ 1,500 元 → 每月要 20 筆全部成功、零虧損才到 3 萬；淨賺 5% ＝ 2,500 元 → 12 筆。")

    # 明細
    t_main = collect(main_sig, "open", 0.03, "redlow", 10)
    t_main["名稱"] = t_main["代碼"].map(lambda c: POOL.get(c, ("", ""))[0])
    t_main.to_csv(OUT / "green_red_trades.csv", index=False, encoding="utf-8-sig")
    yr = t_main.groupby(t_main["進場日"].dt.year)["淨報酬"].agg(["count", "mean"])
    P("")
    P("## 7. 主版本分年（嚴格＋過量｜開盤買｜淨 3%｜紅K最低｜10 天）")
    for y, r in yr.iterrows():
        P(f"- {y}：{int(r['count'])} 筆，每筆 {r['mean']:+.2%}")
    bc = t_main.groupby("名稱")["淨報酬"].agg(["count", "mean"]).sort_values("mean")
    P("## 8. 主版本各股（每筆平均，最差 5 檔／最好 5 檔）")
    for nm, r in pd.concat([bc.head(5), bc.tail(5)]).iterrows():
        P(f"- {nm}：{int(r['count'])} 筆，{r['mean']:+.2%}")

    txt = "\n".join(lines)
    (OUT / "green_red.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
