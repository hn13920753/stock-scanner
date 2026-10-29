"""
用真實股價跑一次產業掃描（在 GitHub Actions 上執行，不需要 Streamlit）。
用途：確認每檔都抓得到真實資料、引擎在真實行情下正常產出結果。

執行：python tools/real_scan.py            （全部產業）
      python tools/real_scan.py "AI 產業"  （單一產業）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine as E  # noqa: E402
import scanner as S  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

def run(sector, market):
    stocks = stocks_of(sector)
    t0 = time.time()
    prices = S.fetch_batch(stocks)
    ctx = S.sector_context(stocks, prices)
    t1 = time.time()
    rows = [S.analyze_one(s, prices.get(s["code"]), {}, market, MKT, ctx.get(s["code"])) for s in stocks]
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
        if x["r"]:
            why += "｜" + "；".join(x["r"]["filters"]["info"])
        print(f"{x['code']:>6} {x['name']:<8} {x['decision']:<5} {src or '':<26} {info}  {why}")
    print("\n== 可進場 ==")
    for no, x in enumerate(res["picks"], 1):
        r, pl = x["r"], x["r"]["plan"]
        print(f"#{no} {x['code']} {x['name']} 買 {pl['entry']} 停利 {pl['tp3']} 停損 {pl['stop']} "
              f"風報比 {pl['rr']:.2f} 勝率 {S.win_rate(r)} 樣本 {r['stats']['n']} 符合度 {r['fit']:.0f}% 數量 {pl['shares']}")
    print("\n== 等待中、最接近觸發 ==")
    for x in res["watch"]:
        print(f"{x['code']} {x['name']} 收 {x['r']['close']} 觸發 {x['r']['trigger']['price']} 差 {S.trigger_gap(x):+.1f}%")
    # 回檔布局觸發價（收盤後預先告訴使用者：收盤跌到這個價 → 隔天開盤買）
    dl = []
    for s in stocks:
        p = prices.get(s["code"])
        df = p[0] if p else None
        if df is None or len(df) < 80:
            continue
        try:
            an = E.Analyzer(df, s["code"], {}, mkt=MKT)
        except Exception:  # noqa: BLE001
            continue
        i = an.n - 1
        lvl, c, h = an.dip_level(i), float(an.C[i]), float(an.HI60[i])
        if lvl == lvl and c > 0:
            dl.append((c / lvl - 1, s, c, h, lvl, an.dip_drop()))
    dl.sort(key=lambda t: t[0])
    print("\n== 回檔布局觸發價（收盤 ≤ 觸發價 → 隔天開盤買）==")
    for gap, s, c, h, lvl, x in dl[:15 if sector.startswith("⭐") else 5]:
        tag = "已到價" if gap <= 0 else f"還差 {gap * 100:.1f}%"
        print(f"{s['code']} {s['name']} 收 {c:g}｜60日高 {h:g}｜跌 {x * 100:.0f}% 觸發價 {E.round_down(lvl, E.is_etf(s['code'])):g}｜{tag}")


MKT = E.load_index(E.DATA_YEARS)
market = E.market_env(MKT)
for sec in ([sys.argv[1]] if len(sys.argv) > 1 else list(SECTORS)):
    run(sec, market)
    print("\n" + "=" * 70 + "\n")
