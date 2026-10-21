"""
研究：V1.4「積極組」（測試中）——原本判定「股性波動過大」（ATR14 > 4.5%）的個股，
改用「寬停損＋小部位＋較高停利」後，歷史上是賺還是賠？（2026-10-21）
只做統計，不改規則。輸出 research/aggr.txt 與逐筆 research/aggr_trades_*.csv。

做法：
1. 股票池：產業清單全部個股＋台灣 50 常見成分股（只測個股；ETF 沒有高波動問題）。10 年日 K。
2. 進場訊號完全沿用引擎（18 型態、四種買點、否決條件、停損 ≥ 1 ATR 過濾），只對「訊號當天 ATR% 介於 4.5%～9%」的個股
   改變停損／停利／風報比門檻。訊號隔天開盤買、同一檔同時只持有一筆。
3. 比較：
   V1 只放寬股性（淨利 3%、結構停損、風報比 ≥ 1）＝「單純不排除高波動股」
   V2 淨利 5%、結構停損、風報比 ≥ 1
   V3 淨利 5%、停損 ≥ 1.5 ATR、風報比 ≥ 0.5
   V4 淨利 5%、停損 ≥ 2 ATR、風報比 ≥ 0.3
   V5 淨利 3%、停損 ≥ 1.5 ATR、風報比 ≥ 0.3
   V6 淨利 8%、停損 ≥ 2 ATR、風報比 ≥ 0.3（2026-10-21 起引擎預設；唯一前後兩段都賺）
4. 前段（~2024）／後段（2025~）分開看。已扣手續費與證交稅；另列「再扣滑價每邊 0.1%」。
5. 金額：每筆最大損失 20,000 元、投入上限 300,000 元反推股數，算平均每筆賺賠多少元。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402
import scanner as S  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

OUT = Path(os.environ.get("AG_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
CUT = "2025-01-01"
SLIP = 0.2  # 來回滑價 %（每邊 0.1%）
RISK, CAP = 20000, 300000

EXTRA = [
    "2330", "2317", "2454", "2308", "2382", "2303", "1301", "1303", "2002", "3711", "2357", "2395",
    "3045", "4904", "2912", "1216", "2207", "2603", "2609", "2615", "1101", "1326", "6505", "2327",
    "3008", "2379", "3034", "2301", "2409", "3481", "2344", "2408", "2618", "1402", "2105", "2474", "4938",
]

VARIANTS = {
    "V1 只放寬股性（3%・結構停損・RR≥1）": dict(aggr_target=3.0, aggr_stop_atr=0.0, aggr_min_rr=1.0, aggr_early=3.0),
    "V2 5%・結構停損・RR≥1": dict(aggr_target=5.0, aggr_stop_atr=0.0, aggr_min_rr=1.0, aggr_early=3.0),
    "V3 5%・停損≥1.5ATR・RR≥0.5": dict(aggr_target=5.0, aggr_stop_atr=1.5, aggr_min_rr=0.5, aggr_early=3.0),
    "V4 5%・停損≥2ATR・RR≥0.3": dict(aggr_target=5.0, aggr_stop_atr=2.0, aggr_min_rr=0.3, aggr_early=3.0),
    "V5 3%・停損≥1.5ATR・RR≥0.3": dict(aggr_target=3.0, aggr_stop_atr=1.5, aggr_min_rr=0.3, aggr_early=3.0),
    "V6 8%・停損≥2ATR・RR≥0.3（預設）": dict(aggr_target=8.0, aggr_stop_atr=2.0, aggr_min_rr=0.3, aggr_early=3.0),
}

t0 = time.time()
universe = {}
for sec in SECTORS:
    for s in stocks_of(sec):
        if not E.is_etf(s["code"]):
            universe.setdefault(s["code"], s)
for c in EXTRA:
    universe.setdefault(c, {"code": c, "name": c, "market": "TW"})

if os.environ.get("AG_FAKE"):  # 離線測試用合成資料
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2016-01-01", "2026-10-20")
    prices = {}
    for code in list(universe)[:8]:
        r = rng.normal(0.0006, 0.035, len(idx))
        c = 100 * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, 0.01, len(idx)))
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, .015, len(idx)))),
                           "low": np.minimum(o, c) * (1 - abs(rng.normal(0, .015, len(idx)))), "close": c,
                           "volume": rng.lognormal(8, .6, len(idx))}, index=idx)
        prices[code] = (df, "fake", [])
    mkt = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)
else:
    prices = S.fetch_batch(list(universe.values()))
    mkt = E.load_index(E.DATA_YEARS)
print(f"抓資料 {time.time() - t0:.0f} 秒；{sum(1 for v in prices.values() if v and v[0] is not None)} 檔有資料；"
      f"加權指數 {'OK' if mkt is not None else '失敗'}")

results = {v: [] for v in VARIANTS}
for code, p in prices.items():
    df = p[0] if p else None
    if df is None or len(df) < 80:
        continue
    try:
        an = E.Analyzer(df, code, {"f_aggr": True}, mkt=mkt)
    except Exception as e:  # noqa: BLE001
        print(f"{code} 失敗：{e}")
        continue
    if not np.nanmax(np.where(np.isfinite(an.ATRP), an.ATRP, 0)) > E.TH["atr_max"]:
        continue  # 從來不是高波動股，與積極組無關
    base = dict(an.p)
    for vname, ov in VARIANTS.items():
        an.p = {**base, **ov}
        try:
            bt = an.backtest()
        except Exception as e:  # noqa: BLE001
            print(f"{code} {vname} 失敗：{e}")
            continue
        if bt.empty:
            continue
        bt = bt[bt["積極組"].astype(bool)].copy()
        if bt.empty:
            continue
        bt["代號"] = code
        results[vname].append(bt)
    print(f"{code} 完成（{time.time() - t0:.0f} 秒）")

res = {v: (pd.concat(x, ignore_index=True) if x else pd.DataFrame()) for v, x in results.items()}


def money(g):
    """固定風險 20,000 元、投入上限 300,000 元下，每筆賺賠金額（已扣滑價）。"""
    buy_c, sell_c = E.cost_rates("2330", E.DEFAULT_PARAMS)
    loss = g.apply(lambda r: -E.net_return(r["進場價"], r["停損"], buy_c, sell_c), axis=1).clip(lower=0.3)
    pos = np.minimum(CAP, RISK / (loss / 100))
    return pos * (g["淨報酬%"] - SLIP) / 100


def line(name, g):
    if g is None or len(g) == 0:
        return f"{name}：0 筆"
    r = g["淨報酬%"]
    m = money(g)
    tag = "（樣本 < 30，僅供參考）" if len(g) < 30 else ""
    return (f"{name}：{len(g)} 筆・{g['代號'].nunique()} 檔・先到停利 {(g['結果'] == '停利').mean() * 100:.1f}%・"
            f"停損 {g['結果'].str.startswith('停損').mean() * 100:.1f}%・平均淨報酬 {r.mean():+.2f}%"
            f"（扣滑價 {r.mean() - SLIP:+.2f}%）・每筆平均 {m.mean():+,.0f} 元・合計 {m.sum():+,.0f} 元・"
            f"平均持有 {g['持有天數'].mean():.1f} 天{tag}")


out = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・引擎 {E.ENGINE_VERSION}",
       "只統計「訊號當天 ATR 4.5%～9% 的個股」（原本判定不適合本策略）。已扣手續費與證交稅 0.3%；"
       f"括號內再扣來回滑價 {SLIP}%。金額＝每筆最大損失 {RISK:,} 元、投入上限 {CAP:,} 元反推股數（含滑價）。",
       "\n== 一、各版本（全部期間）=="]
for v, g in res.items():
    out.append(line(v, g))
out.append("\n== 二、樣本外檢查（前段 ~2024 vs 後段 2025~）==")
for v, g in res.items():
    out.append(f"[{v}]")
    if g.empty:
        out.append("  無資料")
        continue
    out.append("  " + line("前段", g[g["訊號日"] < CUT]))
    out.append("  " + line("後段", g[g["訊號日"] >= CUT]))
dv = "V6 8%・停損≥2ATR・RR≥0.3（預設）"
g = res[dv]
if len(g):
    out.append(f"\n== 三、{dv}：依買點、依大盤 ==")
    for k, x in g.groupby("買點"):
        out.append("  " + line(k, x))
    for k, x in g.groupby("大盤"):
        out.append("  " + line(f"大盤{k}", x))
    out.append(f"  最差一筆 {g['淨報酬%'].min():+.2f}%・最大不利平均 {g['最大不利%'].mean():.2f}%・"
               f"最大有利平均 {g['最大有利%'].mean():.2f}%")
    yr = g.assign(年=g["訊號日"].str[:4]).groupby("年")["淨報酬%"].agg(["count", "mean"]).round(2)
    out.append("  逐年（筆數、平均淨報酬%）：\n" + yr.to_string())
text = "\n".join(out)
print(text)
(OUT / "aggr.txt").write_text(text, encoding="utf-8")
for i, (v, g) in enumerate(res.items(), 1):
    if len(g):
        g.to_csv(OUT / f"aggr_trades_V{i}.csv", index=False, encoding="utf-8-sig")
print(f"總耗時 {time.time() - t0:.0f} 秒")
