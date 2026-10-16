"""
研究：回測資料由 5 年延長為 10 年的影響（使用現行預設規則，不改任何規則）。輸出到 research/years.txt。
比較：近 5 年 vs 更早 5 年的訊號表現、各年度表現、每檔樣本數變化。
執行：python tools/research_years.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402
import scanner as S  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

stocks = {}
for sec in SECTORS:
    for s in stocks_of(sec):
        stocks.setdefault(s["code"], s)
prices = S.fetch_batch(list(stocks.values()))
mkt = E.load_index(E.DATA_YEARS)
ctx = S.sector_context(list(stocks.values()), prices)

bts, per = [], []
for code, p in prices.items():
    df = p[0] if p else None
    if df is None or len(df) < 80:
        continue
    sec, sname = ctx.get(code, (None, ""))
    try:
        bt = E.Analyzer(df, code, {}, mkt=mkt, sec=sec, sec_name=sname).backtest()
    except Exception as e:  # noqa: BLE001
        print(f"{code} 失敗：{e}")
        continue
    if bt.empty:
        per.append((code, 0, 0, df.index[0]))
        continue
    bt["代號"], bt["ETF"] = code, E.is_etf(code)
    cut = (df.index[-1] - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    per.append((code, len(bt), int((bt["訊號日"] >= cut).sum()), df.index[0]))
    bts.append(bt)
bt = pd.concat(bts, ignore_index=True)
last = max(p[0].index[-1] for p in prices.values() if p and p[0] is not None)
cut = (last - pd.DateOffset(years=5)).strftime("%Y-%m-%d")


def line(name, g):
    if len(g) == 0:
        return f"{name}：0 筆"
    r = g["淨報酬%"]
    return (f"{name}：{len(g)} 筆・先到停利 {(g['結果'] == '停利').mean() * 100:.1f}%・"
            f"停損 {g['結果'].str.startswith('停損').mean() * 100:.1f}%・平均淨報酬 {r.mean():+.2f}%")


pf = pd.DataFrame(per, columns=["代號", "10年筆數", "近5年筆數", "資料起日"])
out = [f"資料至 {last:%Y-%m-%d}；{len(pf)} 檔；分界 {cut}（現行預設規則）",
       line("10 年全部", bt), line(f"近 5 年（{cut}～）", bt[bt['訊號日'] >= cut]),
       line(f"更早 5 年（～{cut}）", bt[bt['訊號日'] < cut]),
       line("個股", bt[~bt["ETF"]]), line("ETF", bt[bt["ETF"]]),
       "\n== 每檔樣本數 ==",
       f"中位數：近 5 年 {pf['近5年筆數'].median():.0f} 筆 → 10 年 {pf['10年筆數'].median():.0f} 筆",
       f"樣本 ≥ 10 筆的檔數：近 5 年 {(pf['近5年筆數'] >= 10).sum()} 檔 → 10 年 {(pf['10年筆數'] >= 10).sum()} 檔",
       f"樣本 ≥ 30 筆的檔數：近 5 年 {(pf['近5年筆數'] >= 30).sum()} 檔 → 10 年 {(pf['10年筆數'] >= 30).sum()} 檔",
       f"資料不滿 10 年（上市較晚）：{(pf['資料起日'] > last - pd.DateOffset(years=9, months=11)).sum()} 檔",
       "\n== 各年度 =="]
bt["年"] = bt["訊號日"].str[:4]
for y, g in bt.groupby("年"):
    out.append("  " + line(y, g))
text = "\n".join(out)
print(text)
(ROOT / "research").mkdir(exist_ok=True)
(ROOT / "research" / "years.txt").write_text(text, encoding="utf-8")
