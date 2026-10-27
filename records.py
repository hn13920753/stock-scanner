"""
進場紀錄與「事後驗證」（前瞻測試，forward test）。

做法：每個交易日收盤後，把系統判定「可進場」的股票原封不動記下來（當下的計畫停損、停利），
之後每天用真實股價檢查：隔日開盤有沒有成交 → 最長持有期內（「最長持有天」欄；舊紀錄 15 天）先碰到停利還是停損 → 記下實際結果。

和回測不同：回測是拿過去的資料驗證，可能「事後諸葛」；這裡的每一筆都是「當天就留下的預測」，
結果完全由之後的真實行情決定，不能改、不能挑。

驗證規則（與 engine.backtest 一致，另加 App 上寫的「開盤高於訊號價 1% 以上不追」）：
- 進場（收盤型買點）：訊號日的下一個交易日開盤價。開盤 > 訊號收盤 × 1.01 → 未成交（不追高）；開盤 ≤ 停損 → 未成交。
- 進場（掛單型買點，V1.3 起）：之後 3 個交易日內，開盤 ≤ 限價以開盤價成交、盤中最低 ≤ 限價以限價成交；
  成交前收盤跌破停損 → 取消；成交當天只檢查停損（與回測相同）。
- 停利價依實際進場價重算（依「淨利目標%」欄；V1.5 起短線 10%、回檔布局 10%）；停損沿用計畫停損，回檔布局改為實際進場價 −20%、不設不追價。
- 逐日檢查：跳空開盤低於停損 → 以開盤價停損；盤中碰停損 → 停損價出場；同一天同時碰到停損與停利，先算停損。
- 最長持有天數（V1.5 起 60 天）收盤仍未碰到 → 期滿以收盤價出場。已扣手續費與證交稅，未含滑價。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import engine as E

COLUMNS = [
    "訊號日", "代號", "名稱", "產業", "子類", "訊號收盤", "計畫停損", "計畫停利3%", "風報比",
    "回測樣本", "回測勝率%", "符合度%", "買點", "型態", "大盤",
    "狀態", "進場日", "進場價", "停利價", "出場日", "出場價", "結果", "淨報酬%", "持有天數",
    "最大不利%", "最大有利%", "更新日",
    # V1.3 新增（舊紀錄為空白）
    "計畫進場價", "掛單", "過濾", "引擎版本",
    # V1.4 新增：淨利目標（積極組 5%；空白＝預設 3%）
    "淨利目標%",
    # V1.5 新增：最長持有（空白＝舊紀錄 15 天）
    "最長持有天",
]
NO_CHASE = 0.01  # 開盤高於訊號收盤 1% 以上不追


def load(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"代號": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS]


def add_signals(book: pd.DataFrame, rows: list[dict], sector_of: dict, market,
                only_aggr: bool = False) -> tuple[pd.DataFrame, int]:
    """把今天「可進場」的股票加進紀錄（同一檔同一訊號日只記一次）。only_aggr=True 只記積極組（紙上交易簿）。"""
    have = set(zip(book["代號"].astype(str), book["訊號日"].astype(str)))
    new = []
    for x in rows:
        r = x.get("r")
        if not r or not r["decision"].startswith("可進場") or not r.get("plan"):
            continue
        if only_aggr and not r["plan"].get("aggr"):
            continue
        key = (x["code"], r["date"])
        if key in have:
            continue
        have.add(key)
        pl, s = r["plan"], r["stats"]
        new.append({
            "訊號日": r["date"], "代號": x["code"], "名稱": x["name"],
            "產業": "、".join(sector_of.get(x["code"], [])), "子類": x["group"],
            "訊號收盤": round(r["close"], 2), "計畫停損": pl["stop"], "計畫停利3%": pl["tp3"],
            "風報比": round(pl["rr"], 2) if math.isfinite(pl["rr"]) else None,
            "回測樣本": s.get("n", 0),
            "回測勝率%": round(s["tp_rate"], 1) if s.get("n", 0) >= 10 else None,
            "符合度%": round(r["fit"]), "買點": r["signal"]["type"],
            "型態": "、".join(E.PATTERNS[k] for k in r["signal"]["ids"]),
            "大盤": (market or {}).get("state", "無法判定"), "狀態": "待進場",
            "計畫進場價": pl["entry"], "掛單": "是" if r["signal"].get("mode") == "limit" else "否",
            "過濾": "；".join((r.get("filters") or {}).get("info", []) + pl.get("halve", [])),
            "引擎版本": E.ENGINE_VERSION, "淨利目標%": pl.get("target", E.DEFAULT_PARAMS["target_net"]),
            "最長持有天": pl.get("hold", 15),
        })
    if new:
        book = pd.concat([book, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
    return book, len(new)


def evaluate_one(rec: dict, df: pd.DataFrame, hold: int = 15) -> dict:
    """依真實股價更新一筆紀錄的狀態。df 為該股日K（index 為日期）。"""
    if rec.get("狀態") in ("已出場", "未成交"):
        return rec
    code = str(rec["代號"])
    etf = E.is_etf(code)
    buy_c, sell_c = E.cost_rates(code, E.DEFAULT_PARAMS)
    sig_day = pd.Timestamp(rec["訊號日"])
    after = df[df.index > sig_day]
    if after.empty:
        return {**rec, "狀態": "待進場"}

    O, H, L, C = (after[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    dates = [d.strftime("%Y-%m-%d") for d in after.index]
    stop = float(rec["計畫停損"])
    dip = str(rec.get("買點", "")).startswith("回檔布局")
    hh = pd.to_numeric(rec.get("最長持有天"), errors="coerce")
    hold = int(hh) if pd.notna(hh) and hh > 0 else hold
    limit_mode = str(rec.get("掛單")) == "是"
    if limit_mode:  # 掛單型：3 個交易日內觸價成交
        limit = float(rec["計畫進場價"])
        k0 = None
        for k in range(min(E.TH["limit_valid"], len(after))):
            if O[k] <= limit:
                entry, k0 = O[k], k
                break
            if L[k] <= limit:
                entry, k0 = limit, k
                break
            if C[k] < stop:
                return {**rec, "狀態": "未成交", "結果": f"成交前收盤跌破停損 {stop:g}，掛單取消", "進場日": dates[k]}
        if k0 is None:
            if len(after) >= E.TH["limit_valid"]:
                return {**rec, "狀態": "未成交", "結果": f"{E.TH['limit_valid']} 日內未觸及限價 {limit:g}"}
            return {**rec, "狀態": "待進場"}
        O, H, L, C, dates = O[k0:], H[k0:], L[k0:], C[k0:], dates[k0:]
        after = after.iloc[k0:]
    else:
        entry = O[0]
        if dip:
            stop = E.round_down(entry * (1 - E.TH["dip_stop"]), etf)  # 回檔布局：依實際進場價 −20%
        elif entry > float(rec["訊號收盤"]) * (1 + NO_CHASE):
            return {**rec, "狀態": "未成交", "結果": f"開盤 {entry:g} 高於訊號價 1% 以上，不追", "進場日": dates[0]}
        if entry <= stop:
            return {**rec, "狀態": "未成交", "結果": f"開盤 {entry:g} 已低於停損", "進場日": dates[0]}

    tgt = pd.to_numeric(rec.get("淨利目標%"), errors="coerce")
    tgt = float(tgt) if pd.notna(tgt) and tgt > 0 else E.DEFAULT_PARAMS["target_net"]
    tp = E.target_price(entry, tgt, buy_c, sell_c, etf)
    out = {**rec, "進場日": dates[0], "進場價": round(entry, 2), "停利價": tp}
    mae = mfe = 0.0
    last = min(hold, len(after)) - 1
    for j in range(0, last + 1):
        mae = min(mae, L[j] / entry - 1)
        mfe = max(mfe, H[j] / entry - 1)
        exit_ = why = None
        if (j > 0 or limit_mode or dip) and O[j] <= stop:
            exit_, why = (min(O[j], entry) if j == 0 else O[j]), "停損（跳空）"
        elif L[j] <= stop:
            exit_, why = stop, "停損"
        elif limit_mode and j == 0:
            pass  # 掛單成交當天只檢查停損（保守，與回測相同）
        elif H[j] >= tp:
            exit_, why = (O[j] if j > 0 and O[j] >= tp else tp), "停利"
        elif j == hold - 1:
            exit_, why = C[j], "期滿"
        if exit_ is not None:
            out.update(狀態="已出場", 出場日=dates[j], 出場價=round(float(exit_), 2), 結果=why,
                       持有天數=j + 1)
            out["淨報酬%"] = round(E.net_return(entry, exit_, buy_c, sell_c), 2)
            break
    else:  # 還在持有期間內
        out.update(狀態="持有中", 持有天數=last + 1, 結果="",
                   出場價=round(float(C[last]), 2), 出場日=dates[last])
        out["淨報酬%"] = round(E.net_return(entry, C[last], buy_c, sell_c), 2)  # 目前帳面
    out["最大不利%"] = round(mae * 100, 2)
    out["最大有利%"] = round(mfe * 100, 2)
    return out


def evaluate(book: pd.DataFrame, prices: dict, today: str) -> pd.DataFrame:
    recs = []
    for rec in book.to_dict("records"):
        p = prices.get(str(rec["代號"]))
        if p and p[0] is not None and rec.get("狀態") not in ("已出場", "未成交"):
            new = evaluate_one(rec, p[0])
            if new != rec:
                new["更新日"] = today
            rec = new
        recs.append(rec)
    return pd.DataFrame(recs, columns=COLUMNS)


def summary(book: pd.DataFrame) -> dict:
    done = book[book["狀態"] == "已出場"]
    r = pd.to_numeric(done["淨報酬%"], errors="coerce").dropna()
    n = len(r)
    out = {
        "signals": len(book), "closed": n,
        "open": int((book["狀態"] == "持有中").sum()),
        "pending": int((book["狀態"] == "待進場").sum()),
        "skipped": int((book["狀態"] == "未成交").sum()),
    }
    if n:
        wins, losses = r[r > 0], r[r <= 0]
        out.update(
            tp_rate=(done["結果"] == "停利").mean() * 100,
            sl_rate=done["結果"].astype(str).str.startswith("停損").mean() * 100,
            win_rate=(r > 0).mean() * 100, avg=r.mean(),
            avg_win=wins.mean() if len(wins) else 0.0, avg_loss=losses.mean() if len(losses) else 0.0,
            avg_days=pd.to_numeric(done["持有天數"], errors="coerce").mean(),
        )
    return out
