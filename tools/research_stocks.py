"""
研究：為什麼個股表現比 ETF 差？測試幾種只針對「個股」的調整。只做統計，不改規則。
輸出 research/stocks.txt。

現行設定（V1.3 預設）：①停損≥1ATR 開、②大盤濾網 開（不影響單筆報酬）、③族群強弱 關。
比較的變體（都只跑個股，ETF 不變）：
  A 現行：淨利 3%、持有 15 日、股性 ATR 1.2～4.5%
  B 淨利 2%：停利較近，較不受成本與波動影響
  C 股性上限 3.0%：只做波動較小的個股
  D 淨利 2%＋股性上限 3.0%
  E 不開①（V1.2 原始規則，對照用）
  F 淨利 2%＋最低風報比 0.7（B 常因風報比 < 1 而沒有訊號，放寬看看）
"""
import sys
import time
from pathlib import Path

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
        if not E.is_etf(s["code"]):
            stocks.setdefault(s["code"], s)
prices = S.fetch_batch(list(stocks.values()))
mkt = E.load_index(E.DATA_YEARS)
print(f"個股 {len(stocks)} 檔，抓資料 {time.time() - t0:.0f} 秒")

BASE = {"f_stop_atr": True, "f_market": True, "f_sector": False}
VARIANTS = {
    "A 現行（3%、ATR≤4.5%）": (BASE, 4.5),
    "B 淨利 2%": ({**BASE, "target_net": 2.0}, 4.5),
    "C 股性上限 ATR≤3%": (BASE, 3.0),
    "D 淨利 2%＋ATR≤3%": ({**BASE, "target_net": 2.0}, 3.0),
    "E 不開①（V1.2 對照）": ({**BASE, "f_stop_atr": False}, 4.5),
    "F 淨利 2%＋風報比≥0.7": ({**BASE, "target_net": 2.0, "min_rr": 0.7}, 4.5),
}
CUT = "2025-01-01"
orig_max = E.TH["atr_max"]


def line(name, g):
    if g is None or len(g) == 0:
        return f"{name}：0 筆"
    r = g["淨報酬%"]
    return (f"{name}：{len(g)} 筆・先到停利 {(g['結果'] == '停利').mean() * 100:.1f}%・"
            f"停損 {g['結果'].str.startswith('停損').mean() * 100:.1f}%・平均淨報酬 {r.mean():+.2f}%・"
            f"總和 {r.sum():+.1f}%・{g['代號'].nunique()} 檔")


out = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・引擎 {E.ENGINE_VERSION}・個股 {len(stocks)} 檔",
       "只統計個股（ETF 不在內）。已扣手續費與證交稅 0.3%，未含滑價。", ""]
res = {}
for name, (params, amax) in VARIANTS.items():
    E.TH["atr_max"] = amax
    bts = []
    for code, p in prices.items():
        df = p[0] if p else None
        if df is None or len(df) < 80:
            continue
        try:
            bt = E.Analyzer(df, code, params, mkt=mkt).backtest()
        except Exception as e:  # noqa: BLE001
            print(code, e)
            continue
        if len(bt):
            bt["代號"] = code
            bts.append(bt)
    res[name] = bt = pd.concat(bts, ignore_index=True) if bts else pd.DataFrame()
    out.append(line(name, bt))
    if len(bt):
        out.append("    " + line("前段 ~2024", bt[bt["訊號日"] < CUT]))
        out.append("    " + line("後段 2025~", bt[bt["訊號日"] >= CUT]))
    print(f"{name} 完成（{time.time() - t0:.0f} 秒）")
E.TH["atr_max"] = orig_max

out.append("\n== 現行版本：依買點 ==")
a = res["A 現行（3%、ATR≤4.5%）"]
if len(a):
    for k, g in a.groupby("買點"):
        out.append("  " + line(k, g))

text = "\n".join(out)
print(text)
(ROOT / "research").mkdir(exist_ok=True)
(ROOT / "research" / "stocks.txt").write_text(text, encoding="utf-8")
