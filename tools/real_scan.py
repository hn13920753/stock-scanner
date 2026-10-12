"""
用真實股價跑一次產業掃描（在 GitHub Actions 上執行，不需要 Streamlit）。
用途：確認每檔都抓得到真實資料、引擎在真實行情下正常產出結果。

執行：python tools/real_scan.py "AI 產業"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine as E  # noqa: E402
import scanner as S  # noqa: E402
from sectors import stocks_of  # noqa: E402

sector = sys.argv[1] if len(sys.argv) > 1 else "AI 產業"
stocks = stocks_of(sector)
t0 = time.time()
prices = S.fetch_batch(stocks)
t1 = time.time()
market = E.market_env()
rows = [S.analyze_one(s, prices.get(s["code"]), {}, market) for s in stocks]
t2 = time.time()
res = S.classify(rows)

print(f"產業：{sector}　共 {len(stocks)} 檔　抓資料 {t1 - t0:.0f} 秒・分析 {t2 - t1:.0f} 秒")
print(f"大盤：{market}")
print(f"可進場 {res['picks_total']}・等待 {res['wait_n']}・不可進場 {res['veto_n']}・資料失敗 {len(res['fail'])}")
print("\n== 每檔資料 ==")
for x in rows:
    p = prices.get(x["code"])
    df, src = (p[0], p[1]) if p else (None, None)
    info = f"{len(df)} 根 {df.index[0]:%Y-%m-%d}～{df.index[-1]:%Y-%m-%d} 收 {df['close'].iloc[-1]:.2f}" if df is not None else x["error"]
    why = ("；".join(x["r"]["why"][:1]) if x["r"] and x["r"]["why"] else "")
    print(f"{x['code']:>6} {x['name']:<8} {x['decision']:<5} {src or '':<26} {info}  {why}")
print("\n== 可進場 ==")
for no, x in enumerate(res["picks"], 1):
    r, pl = x["r"], x["r"]["plan"]
    print(f"#{no} {x['code']} {x['name']} 買 {pl['entry']} 停利 {pl['tp3']} 停損 {pl['stop']} "
          f"風報比 {pl['rr']:.2f} 勝率 {S.win_rate(r)} 樣本 {r['stats']['n']} 符合度 {r['fit']:.0f}% 數量 {pl['shares']}")
print("\n== 等待中、最接近觸發 ==")
for x in res["watch"]:
    print(f"{x['code']} {x['name']} 收 {x['r']['close']} 觸發 {x['r']['trigger']['price']} 差 {S.trigger_gap(x):+.1f}%")
