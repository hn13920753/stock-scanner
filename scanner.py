"""
產業掃描邏輯（不含畫面，可單獨測試）。

流程：一次批次抓整個產業的 5 年日 K → 每檔丟進 engine.Analyzer（與股票系統完全相同的規則）
→ 依決策分成「可進場／等待／不可進場／資料失敗」→ 可進場依排序規則取前 N 檔。

本檔不新增、不修改任何進出場判斷；判斷全部來自 engine.py。
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import engine as E

TOP_N = 10        # 可進場最多列幾檔
WATCH_N = 10      # 「等待中、最接近觸發」最多列幾檔
MIN_SAMPLES = 10  # 回測樣本 ≥ 10 筆才把勝率拿來排序（與單檔報告顯示規則一致）


# ------------------------------------------------------------------
# 資料：批次抓取，失敗的再逐檔用 engine.load_prices 補抓
# ------------------------------------------------------------------
def _from_yf_frame(h: pd.DataFrame):
    if h is None or h.empty:
        return None
    h = h.rename(columns=str.lower)
    if not {"open", "high", "low", "close", "volume"}.issubset(h.columns):
        return None
    h = h.dropna(subset=["close"])
    if len(h) < 80:
        return None
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    h = h.copy()
    h["volume"] = h["volume"] / 1000.0  # 股 → 張（與 engine.fetch_yahoo 相同）
    h = h[h["volume"] > 0]
    h = E._clean(h)
    return h if len(h) >= 80 else None


def fetch_batch(stocks: list[dict], years: int = 5) -> dict:
    """回傳 {code: (df 或 None, 來源, 錯誤列表)}。"""
    out: dict = {}
    tickers = {s["code"]: f"{s['code']}.{s['market']}" for s in stocks}
    try:
        import yfinance as yf

        data = yf.download(list(tickers.values()), period=f"{years}y", interval="1d",
                           group_by="ticker", auto_adjust=False, actions=False,
                           threads=True, progress=False)
        for code, tk in tickers.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if tk not in data.columns.get_level_values(0):
                        continue
                    sub = data[tk]
                else:
                    sub = data
                df = _from_yf_frame(sub)
                if df is not None:
                    out[code] = (df, f"Yahoo Finance（{tk}）", [])
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass

    missing = [s["code"] for s in stocks if s["code"] not in out]
    if missing:  # 批次沒抓到的（例如市場別填錯、被限流）→ 逐檔走原本的多資料源流程
        with ThreadPoolExecutor(max_workers=6) as ex:
            for code, res in zip(missing, ex.map(E.load_prices, missing)):
                out[code] = res
    return out


# ------------------------------------------------------------------
# 分析
# ------------------------------------------------------------------
def analyze_one(stock: dict, price_res, params: dict, market) -> dict:
    """單檔分析；任何錯誤都只影響這一檔。"""
    row = {**stock, "decision": "資料失敗", "r": None, "src": None, "error": ""}
    df, src, errs = price_res if price_res else (None, None, ["未抓取"])
    if df is None:
        row["error"] = "；".join(errs) or "抓不到歷史資料"
        return row
    try:
        r = E.Analyzer(df, stock["code"], params).report(market)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"分析失敗：{e}"
        return row
    row.update(decision=r["decision"], r=r, src=src)
    return row


def win_rate(r) -> float | None:
    s = r["stats"]
    return s["tp_rate"] if s.get("n", 0) >= MIN_SAMPLES else None


def sort_key(row):
    """排序：① 風險報酬比（取到小數 1 位，差不多時才比下一項）② 回測勝率（樣本 ≥ 10 筆才算，否則排後）③ 規則符合度。"""
    r = row["r"]
    rr = r["plan"]["rr"] if r["plan"] else 0.0
    rr = 99.0 if not math.isfinite(rr) else round(rr, 1)
    wr = win_rate(r)
    return (-rr, -(wr if wr is not None else -1.0), -r["fit"])


def trigger_gap(row) -> float | None:
    """等待中個股：收盤距觸發價還差幾 %。"""
    r = row["r"]
    t = r.get("trigger")
    if not t or not t.get("price") or r["close"] <= 0:
        return None
    return (t["price"] / r["close"] - 1) * 100


def classify(rows: list[dict]) -> dict:
    picks = sorted([x for x in rows if x["decision"] == "可進場"], key=sort_key)
    watch = [x for x in rows if x["decision"] == "等待" and trigger_gap(x) is not None]
    watch.sort(key=lambda x: (trigger_gap(x), -x["r"]["fit"]))
    return {
        "picks": picks[:TOP_N],
        "picks_total": len(picks),
        "watch": watch[:WATCH_N],
        "wait_n": sum(x["decision"] == "等待" for x in rows),
        "veto_n": sum(x["decision"] == "不可進場" for x in rows),
        "fail": [x for x in rows if x["decision"] == "資料失敗"],
    }


def export_table(rows: list[dict], sector: str, params: dict) -> pd.DataFrame:
    """完整結果表（給下載 CSV 做複盤紀錄用）。"""
    import datetime as dt

    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    out = []
    for x in rows:
        r = x["r"] or {}
        pl = r.get("plan") if r.get("decision") == "可進場" else None
        st_ = r.get("stats") or {}
        trig = r.get("trigger") or {}
        out.append({
            "掃描時間": now, "產業": sector, "資料日期": r.get("date", ""),
            "代號": x["code"], "名稱": x["name"], "子類": x["group"], "決策": x["decision"],
            "收盤": r.get("close"), "買進價": pl["entry"] if pl else None,
            "停利": pl["tp3"] if pl else None, "2%停利": pl["tp2"] if pl else None,
            "停損": pl["stop"] if pl else None,
            "風報比": round(pl["rr"], 2) if pl and math.isfinite(pl["rr"]) else None,
            "股數": pl["shares"] if pl else None, "預估最大損失": round(pl["max_loss"]) if pl else None,
            "回測樣本": st_.get("n"),
            "回測勝率%": round(st_["tp_rate"], 1) if st_.get("n", 0) >= MIN_SAMPLES else None,
            "符合度%": round(r["fit"]) if r else None,
            "觸發價": trig.get("price") if r.get("decision") != "可進場" else None,
            "主要原因": "；".join(r.get("why", [])[:2]) or x.get("error", ""),
            "淨利目標%": params.get("target_net"), "最長持有日": params.get("hold_days"),
            "資料來源": x.get("src") or "",
        })
    return pd.DataFrame(out)


def pooled_stats(rows: list[dict], group: str | None = None) -> dict:
    """把同一次掃描中（可限定同子類）所有股票的歷史回測合併統計，增加樣本數。只做參考，不影響判斷。"""
    import engine as E

    bts = [x["r"]["bt"] for x in rows
           if x.get("r") is not None and x["r"].get("bt") is not None and not x["r"]["bt"].empty
           and (group is None or x["group"] == group)]
    if not bts:
        return {"n": 0}
    s = E.Analyzer.summarize(pd.concat(bts, ignore_index=True))
    s["stocks"] = len(bts)
    return s
