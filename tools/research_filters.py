"""
研究：V1.3 三項過濾（測試中）對歷史回測的影響。只做統計，不改任何規則。輸出到 research/filters.txt。

三項過濾：
  ① 停損 ≥ 1 ATR：停損距離小於 1 倍 ATR14 的訊號不進場
  ② 大盤濾網：加權指數空頭時股數減半（只影響部位大小，不影響單筆報酬率 → 用「空頭時的訊號表現」來評估）
  ③ 族群強弱：族群 20 日報酬落後大盤超過 3 個百分點時不進場

比較方式：全部產業、全部股票（含 ETF）的歷史訊號合併，比較「全關」「只開一項」「全開」，
並把資料切成前段（~2024）與後段（2025~）檢查兩段方向是否一致（避免只是剛好）。

執行：python tools/research_filters.py
"""
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

t0 = time.time()
stocks = {}
for sec in SECTORS:
    for s in stocks_of(sec):
        stocks.setdefault(s["code"], s)
prices = S.fetch_batch(list(stocks.values()))
mkt = E.load_index(5)
ctx = S.sector_context(list(stocks.values()), prices)
print(f"抓資料 {time.time() - t0:.0f} 秒；加權指數 {'OK' if mkt is not None else '失敗'}；"
      f"有族群指數 {sum(v[0] is not None for v in ctx.values())} 檔")

OFF = {"f_stop_atr": False, "f_market": False, "f_sector": False}
VARIANTS = {
    "全關（V1.2 現行規則）": OFF,
    "只開①停損≥1ATR": {**OFF, "f_stop_atr": True},
    "只開③族群強弱": {**OFF, "f_sector": True},
    "全開（①②③）": {"f_stop_atr": True, "f_market": True, "f_sector": True},
}

results, funnels = {}, {}
for vname, flags in VARIANTS.items():
    bts, fn = [], {}
    for code, p in prices.items():
        df = p[0] if p else None
        if df is None or len(df) < 80:
            continue
        sec, sname = ctx.get(code, (None, ""))
        try:
            an = E.Analyzer(df, code, flags, mkt=mkt, sec=sec, sec_name=sname)
            bt = an.backtest()
        except Exception as e:  # noqa: BLE001
            print(f"{code} 失敗：{e}")
            continue
        for k, v in an.funnel.items():
            fn[k] = fn.get(k, 0) + v
        if bt.empty:
            continue
        bt["代號"] = code
        bt["ETF"] = E.is_etf(code)
        bts.append(bt)
    results[vname] = pd.concat(bts, ignore_index=True) if bts else pd.DataFrame()
    funnels[vname] = fn
    print(f"{vname}：{len(results[vname])} 筆（{time.time() - t0:.0f} 秒）")

CUT = "2025-01-01"


def line(name, g):
    if g is None or len(g) == 0:
        return f"{name}：0 筆"
    r = g["淨報酬%"]
    return (f"{name}：{len(g)} 筆・先到停利 {(g['結果'] == '停利').mean() * 100:.1f}%・"
            f"停損 {g['結果'].str.startswith('停損').mean() * 100:.1f}%・平均淨報酬 {r.mean():+.2f}%・"
            f"總和 {r.sum():+.1f}%")


out = []
base = results["全關（V1.2 現行規則）"]
out.append(f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・引擎 {E.ENGINE_VERSION}")
out.append(f"樣本：{base['代號'].nunique() if len(base) else 0} 檔有成交訊號；"
           f"期間 {base['訊號日'].min() if len(base) else '—'}～{base['訊號日'].max() if len(base) else '—'}")
out.append("勝率＝先到停利比例。總和＝每筆報酬相加（同金額下單時的總獲利概念）。已扣手續費與證交稅，未含滑價。")

out.append("\n== 一、各版本比較（全部期間）==")
for v, bt in results.items():
    out.append(line(v, bt))

out.append("\n== 二、樣本外檢查（前段 ~2024 vs 後段 2025~）==")
for v, bt in results.items():
    if bt.empty:
        out.append(f"[{v}] 無資料")
        continue
    out.append(f"[{v}]")
    out.append("  " + line("前段", bt[bt["訊號日"] < CUT]))
    out.append("  " + line("後段", bt[bt["訊號日"] >= CUT]))

if len(base):
    out.append("\n== 三、②大盤濾網：依訊號當天大盤狀態（全關版本的訊號）==")
    for k in ("多頭", "盤整", "空頭", "無資料"):
        g = base[base["大盤"] == k]
        if len(g):
            out.append("  " + line(k, g))
            for nm, part in (("前段", g[g["訊號日"] < CUT]), ("後段", g[g["訊號日"] >= CUT])):
                out.append("      " + line(nm, part))

    out.append("\n== 四、③族群強弱：依族群 20 日相對大盤（百分點，全關版本、只含個股）==")
    st_ = base[~base["ETF"].astype(bool)].copy()
    st_["族群RS"] = pd.to_numeric(st_["族群RS"], errors="coerce")
    st_["族群分組"] = pd.cut(st_["族群RS"], [-1e9, -6, -3, 0, 3, 6, 1e9],
                         labels=["< -6", "-6 ~ -3", "-3 ~ 0", "0 ~ 3", "3 ~ 6", "> 6"])
    for k, g in st_.groupby("族群分組", observed=True):
        out.append("  " + line(str(k), g))
    g = st_[st_["族群RS"].isna()]
    if len(g):
        out.append("  " + line("無族群資料", g))
    for nm, part in (("前段", st_[st_["訊號日"] < CUT]), ("後段", st_[st_["訊號日"] >= CUT])):
        out.append(f"  [{nm}]")
        out.append("    " + line("落後 > 3 個百分點", part[part["族群RS"] < -3]))
        out.append("    " + line("其餘", part[~(part["族群RS"] < -3)]))

    out.append("\n== 五、①停損距離 ÷ ATR（全關版本）==")
    b2 = base.copy()
    b2["停損/ATR"] = pd.to_numeric(b2["停損/ATR"], errors="coerce")
    b2["分組"] = pd.cut(b2["停損/ATR"], [0, 0.5, 1, 1.5, 2, 3, 100], labels=["<0.5", "0.5~1", "1~1.5", "1.5~2", "2~3", ">3"])
    for k, g in b2.groupby("分組", observed=True):
        out.append("  " + line(str(k), g))

full = results["全開（①②③）"]
if len(full):
    out.append("\n== 六、全開版本：個股 vs ETF、依買點類型 ==")
    for k, g in full.groupby("ETF"):
        out.append("  " + line("ETF" if k else "個股", g))
    for k, g in full.groupby("買點"):
        out.append("  " + line(k, g))
    sh = full[full["大盤"] == "空頭"]
    out.append(f"  其中大盤空頭時進場（股數減半）：{len(sh)} 筆")

out.append("\n== 七、訊號漏斗（每一關擋掉幾次）==")
keys = list(next(iter(funnels.values())).keys()) if funnels else []
tab = pd.DataFrame({v: [funnels[v].get(k, 0) for k in keys] for v in funnels}, index=keys)
out.append(tab.to_string())

text = "\n".join(out)
print(text)
(ROOT / "research").mkdir(exist_ok=True)
(ROOT / "research" / "filters.txt").write_text(text, encoding="utf-8")
for v, bt in results.items():  # 明細（之後可再研究）
    if len(bt):
        tag = {"全關（V1.2 現行規則）": "off", "只開①停損≥1ATR": "atr", "只開③族群強弱": "sector", "全開（①②③）": "all"}[v]
        bt.to_csv(ROOT / "research" / f"filters_trades_{tag}.csv", index=False, encoding="utf-8-sig")
