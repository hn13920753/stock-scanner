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
PAPER = ROOT / "records" / "paper_aggr.csv"  # 積極組紙上交易簿（V1.4 測試中，2026-10-22 起）
today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d")

# 全部產業的股票（同一檔可能屬於多個產業）
stocks, sector_of = {}, {}
for sec in SECTORS:
    for s in stocks_of(sec):
        stocks.setdefault(s["code"], s)
        sector_of.setdefault(s["code"], []).append(sec)

book = R.load(BOOK)
paper = R.load(PAPER)
# 已不在清單、但還有未結束紀錄的股票，也要抓價格來驗證
for rec in book.to_dict("records") + paper.to_dict("records"):
    c = str(rec["代號"])
    if c not in stocks and rec["狀態"] not in ("已出場", "未成交"):
        stocks[c] = {"code": c, "name": rec["名稱"], "market": "TW", "group": rec["子類"]}

prices = S.fetch_batch(list(stocks.values()))
mkt = E.load_index(E.DATA_YEARS)
market = E.market_env(mkt)
scan = [s for c, s in stocks.items() if c in sector_of]
ctx = S.sector_context(scan, prices)
rows = [S.analyze_one(s, prices.get(s["code"]), {}, market, mkt, ctx.get(s["code"])) for s in scan]

book = R.evaluate(book, prices, today)          # 先更新舊紀錄
book, added = R.add_signals(book, rows, sector_of, market)  # 再加今天的新訊號

BOOK.parent.mkdir(exist_ok=True)
book.to_csv(BOOK, index=False, encoding="utf-8-sig")

# 積極組紙上交易：不論 App 預設是否開啟，都用 f_aggr=True 另外記一本，只記積極組訊號
hi_vol = {x["code"] for x in rows if x["r"] and (x["r"].get("atrp") or 0) > E.TH["atr_max"]}  # 只有高波動股可能是積極組
rows_ag = [S.analyze_one(s, prices.get(s["code"]), {"f_aggr": True}, market, mkt, ctx.get(s["code"]))
           for s in scan if s["code"] in hi_vol]
paper = R.evaluate(paper, prices, today)
paper, added_ag = R.add_signals(paper, rows_ag, sector_of, market, only_aggr=True)
paper.to_csv(PAPER, index=False, encoding="utf-8-sig")
pm = R.summary(paper)
print(f"積極組紙上交易：新增 {added_ag} 筆；累計 {pm['signals']} 筆（待進場 {pm['pending']}・持有中 {pm['open']}・"
      f"已出場 {pm['closed']}・未成交 {pm['skipped']}）"
      + (f"；已出場平均 {pm['avg']:+.2f}%" if pm["closed"] else ""))

dates = [x["r"]["date"] for x in rows if x["r"]]
sm = R.summary(book)
print(f"{today} 執行・資料日期 {max(dates) if dates else '—'}・掃描 {len(rows)} 檔・新增訊號 {added} 筆")
print(f"累計 {sm['signals']} 筆：待進場 {sm['pending']}・持有中 {sm['open']}・已出場 {sm['closed']}・未成交 {sm['skipped']}")
if sm["closed"]:
    print(f"已出場：先到停利 {sm['tp_rate']:.0f}%・停損 {sm['sl_rate']:.0f}%・平均淨報酬 {sm['avg']:+.2f}%")
print(book.tail(15).to_string())
