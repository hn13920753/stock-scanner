"""
即時報價（模擬交易用）。

來源：證交所「基本市況報導」mis.twse.com.tw 的 getStockInfo（上市 tse、上櫃 otc 都可查，約延遲 5～20 秒）。
抓不到時改用 Yahoo Finance 最近一筆價格（延遲約 20 分鐘）。
回傳 {代號: {"price": 最新成交價, "prev": 昨收, "time": "HH:MM:SS", "src": 來源}}
"""
import time

import requests

MIS = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
HEAD = {"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}


def _f(x):
    try:
        v = float(str(x).split("_")[0])
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def parse_mis(j: dict) -> dict:
    out = {}
    for r in (j or {}).get("msgArray", []) or []:
        code = r.get("c")
        if not code:
            continue
        price = _f(r.get("z"))  # 最新成交價（剛好沒成交時是 "-"）
        if price is None:  # 用最佳買價、再用開盤價、最後用昨收
            price = _f(r.get("b")) or _f(r.get("o")) or _f(r.get("y"))
        if price is None:
            continue
        out[code] = {"price": price, "prev": _f(r.get("y")), "time": r.get("t", ""),
                     "name": r.get("n", ""), "src": "證交所即時"}
    return out


def fetch_mis(codes, session=None, batch=40) -> dict:
    s = session or requests.Session()
    out = {}
    codes = [c for c in dict.fromkeys(codes) if c]
    for i in range(0, len(codes), batch):
        part = codes[i:i + batch]
        ch = "|".join(f"tse_{c}.tw|otc_{c}.tw" for c in part)
        try:
            r = s.get(MIS, params={"ex_ch": ch, "json": "1", "delay": "0", "_": int(time.time() * 1000)},
                      headers=HEAD, timeout=8)
            out.update(parse_mis(r.json()))
        except Exception:  # noqa: BLE001
            continue
    return out


def fetch_yahoo_last(codes) -> dict:
    """備援：Yahoo Finance 最近一筆（.TW 找不到再試 .TWO）。"""
    out = {}
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return out
    for c in codes:
        for sfx in (".TW", ".TWO"):
            try:
                h = yf.Ticker(c + sfx).history(period="5d", interval="1d")
                if len(h):
                    out[c] = {"price": float(h["Close"].iloc[-1]),
                              "prev": float(h["Close"].iloc[-2]) if len(h) > 1 else None,
                              "time": f"{h.index[-1]:%m/%d}", "src": "Yahoo（延遲）"}
                    break
            except Exception:  # noqa: BLE001
                continue
    return out


def get_quotes(codes) -> dict:
    codes = list(dict.fromkeys(codes))
    q = fetch_mis(codes)
    miss = [c for c in codes if c not in q]
    if miss and len(miss) <= 30:
        q.update(fetch_yahoo_last(miss))
    return q


if __name__ == "__main__":
    import sys
    cs = sys.argv[1:] or ["2330", "6515", "0050", "00981A", "2059"]
    t0 = time.time()
    m = fetch_mis(cs)
    print(f"MIS {len(m)}/{len(cs)} 檔，{time.time() - t0:.1f} 秒")
    for k, v in m.items():
        print(k, v)
    print("Yahoo 備援：", fetch_yahoo_last([c for c in cs if c not in m]))
