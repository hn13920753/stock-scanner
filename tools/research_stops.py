"""
研究：停損距離與回測結果的關係（全部產業、全部股票的歷史訊號合併統計）。
只做統計，不改任何規則。輸出到 research/stops.txt。
"""
import sys
from pathlib import Path

import numpy as np
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

all_bt = []
for code, (df, src, err) in prices.items():
    if df is None or len(df) < 80:
        continue
    try:
        an = E.Analyzer(df, code, {})
        bt = an.backtest()
    except Exception:  # noqa: BLE001
        continue
    if bt.empty:
        continue
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    bt["代號"] = code
    bt["ETF"] = E.is_etf(code)
    bt["停損距離%"] = (bt["進場價"] - bt["停損"]) / bt["進場價"] * 100
    bt["ATR%"] = [atr.loc[pd.Timestamp(d)] / df["close"].loc[pd.Timestamp(d)] * 100 for d in bt["訊號日"]]
    bt["停損/ATR"] = bt["停損距離%"] / bt["ATR%"]
    all_bt.append(bt)

bt = pd.concat(all_bt, ignore_index=True)


def table(col, bins, labels):
    g = bt.groupby(pd.cut(bt[col], bins, labels=labels), observed=False)
    t = pd.DataFrame({
        "筆數": g.size(),
        "先到停利%": g["結果"].apply(lambda s: (s == "停利").mean() * 100),
        "停損%": g["結果"].apply(lambda s: s.str.startswith("停損").mean() * 100),
        "平均淨報酬%": g["淨報酬%"].mean(),
        "平均持有天": g["持有天數"].mean(),
    }).round(2)
    return t.to_string()


out = []
out.append(f"樣本：{bt['代號'].nunique()} 檔、{len(bt)} 筆歷史訊號（{bt['訊號日'].min()}～{bt['訊號日'].max()}）")
r = bt["淨報酬%"]
out.append(f"整體：先到停利 {(bt['結果']=='停利').mean()*100:.1f}%・停損 {bt['結果'].str.startswith('停損').mean()*100:.1f}%"
           f"・期滿 {(bt['結果']=='期滿').mean()*100:.1f}%・平均淨報酬 {r.mean():+.2f}%・中位數停損距離 {bt['停損距離%'].median():.2f}%")
out.append("\n== 依停損距離（進場價到停損的 %）==")
out.append(table("停損距離%", [0, 0.5, 1, 2, 3, 5, 100], ["<0.5%", "0.5~1%", "1~2%", "2~3%", "3~5%", ">5%"]))
out.append("\n== 依停損距離 ÷ ATR14（停損寬度相對於每日正常波動）==")
out.append(table("停損/ATR", [0, 0.5, 1, 1.5, 2, 3, 100], ["<0.5", "0.5~1", "1~1.5", "1.5~2", "2~3", ">3"]))
out.append("\n== 股票 vs ETF ==")
for k, g in bt.groupby("ETF"):
    out.append(f"{'ETF' if k else '個股'}：{len(g)} 筆・先到停利 {(g['結果']=='停利').mean()*100:.1f}%・平均淨報酬 {g['淨報酬%'].mean():+.2f}%・中位數停損距離 {g['停損距離%'].median():.2f}%")
out.append("\n== 依買點類型 ==")
for k, g in bt.groupby("買點"):
    out.append(f"{k}：{len(g)} 筆・先到停利 {(g['結果']=='停利').mean()*100:.1f}%・平均淨報酬 {g['淨報酬%'].mean():+.2f}%")
text = "\n".join(out)
print(text)
(ROOT / "research").mkdir(exist_ok=True)
(ROOT / "research" / "stops.txt").write_text(text, encoding="utf-8")
