"""
每日收盤後自動執行（GitHub Actions）：
1. 掃描全部產業，把「可進場」的股票記進 records/signals.csv（當下的計畫，不可事後修改）
2. 用最新真實股價更新所有未結束紀錄的結果（成交否、停利／停損／期滿、報酬）

執行：python tools/daily_record.py
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine as E  # noqa: E402
import records as R  # noqa: E402
import scanner as S  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

BOOK = ROOT / "records" / "signals.csv"
today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d")

# 全部產業的股票（同一檔可能屬於多個產業）
stocks, sector_of = {}, {}
for sec in SECTORS:
    for s in stocks_of(sec):
        stocks.setdefault(s["code"], s)
        sector_of.setdefault(s["code"], []).append(sec)

book = R.load(BOOK)
# 已不在清單、但還有未結束紀錄的股票，也要抓價格來驗證
for rec in book.to_dict("records"):
    c = str(rec["代號"])
    if c not in stocks and rec["狀態"] not in ("已出場", "未成交"):
        stocks[c] = {"code": c, "name": rec["名稱"], "market": "TW", "group": rec["子類"]}

prices = S.fetch_batch(list(stocks.values()))
market = E.market_env()
rows = [S.analyze_one(s, prices.get(c), {}, market) for c, s in stocks.items() if c in sector_of]

book = R.evaluate(book, prices, today)          # 先更新舊紀錄
book, added = R.add_signals(book, rows, sector_of, market)  # 再加今天的新訊號

BOOK.parent.mkdir(exist_ok=True)
book.to_csv(BOOK, index=False, encoding="utf-8-sig")

dates = [x["r"]["date"] for x in rows if x["r"]]
sm = R.summary(book)
print(f"{today} 執行・資料日期 {max(dates) if dates else '—'}・掃描 {len(rows)} 檔・新增訊號 {added} 筆")
print(f"累計 {sm['signals']} 筆：待進場 {sm['pending']}・持有中 {sm['open']}・已出場 {sm['closed']}・未成交 {sm['skipped']}")
if sm["closed"]:
    print(f"已出場：先到停利 {sm['tp_rate']:.0f}%・停損 {sm['sl_rate']:.0f}%・平均淨報酬 {sm['avg']:+.2f}%")
print(book.tail(15).to_string())
