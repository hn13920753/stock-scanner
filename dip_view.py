"""🪂 跌深清單（AI＋自選）：回檔布局＋閒置放 0050（測試中）。

按一個按鈕 → 抓「⭐ 自選股」＋「AI 產業」最新日K → 列出收盤價距近 60 日最高點已跌 ≥ 門檻的股票。
只列清單、不改其他頁面的判斷規則。近 60 日最高點與 engine.py 相同：最近 60 個交易日（含今天）最高價。
"""
import math

import pandas as pd
import streamlit as st

import engine as E
import scanner as S
import sectors as SEC

FEE, TAX_STOCK, TAX_ETF = 0.001425, 0.003, 0.001
POOLS = ("⭐ 自選股", "AI 產業")


def pool_stocks():
    out, seen = [], set()
    for sec in POOLS:
        if sec not in SEC.SECTORS:
            continue
        for s in SEC.stocks_of(sec):
            if s["code"] not in seen:
                seen.add(s["code"])
                out.append({**s, "pool": "自選" if sec.startswith("⭐") else "AI"})
    return out


def scan_dip(stocks, prices, drop_pct, per, near_pct=8.0):
    """純計算（可單獨測試）。回傳 (已達門檻, 接近中, 失敗代碼, 最新資料日)。"""
    hit, near, fail, last_dates = [], [], [], []
    for s in stocks:
        res = prices.get(s["code"])
        df = res[0] if res else None
        if df is None or len(df) < 60:
            fail.append(f"{s['code']} {s['name']}")
            continue
        close = float(df["close"].iloc[-1])
        hi60 = float(df["high"].iloc[-60:].max())
        if not (close > 0 and hi60 > 0):
            fail.append(f"{s['code']} {s['name']}")
            continue
        etf = E.is_etf(s["code"])
        dist = (close / hi60 - 1) * 100
        trig = E.round_down(hi60 * (1 - drop_pct / 100), etf)
        day = df.index[-1]
        last_dates.append(day)
        cost = close * (1 + FEE)
        shares = int(per // cost) if cost > 0 else 0
        tax = TAX_ETF if etf else TAX_STOCK
        row = {
            "代號": s["code"], "名稱": s["name"], "清單": s["pool"],
            "收盤": close, "60日高": hi60, "距高點%": round(dist, 1), "觸發價": trig,
            "參考股數": shares,
            "停利參考(淨賺10%)": E.round_up(cost * 1.10 / (1 - FEE - tax), etf),
            "停損參考(−20%)": E.round_down(close * 0.80, etf),
            "資料日": day.strftime("%Y-%m-%d"),
        }
        if dist <= -drop_pct:
            hit.append(row)
        elif dist <= -(drop_pct - near_pct):
            row["還差(百分點)"] = round(dist + drop_pct, 1)
            near.append(row)
    hit.sort(key=lambda r: r["距高點%"])
    near.sort(key=lambda r: r["距高點%"])
    return hit, near, fail, (max(last_dates) if last_dates else None)


def _qty(n):
    lots, odd = divmod(int(n), 1000)
    return ((f"{lots} 張" if lots else "") + (f" {odd} 股" if odd else "")).strip() or "0 股"


def _fmt(rows, extra=()):
    df = pd.DataFrame(rows)
    p = lambda x: f"{x:,.2f}".rstrip("0").rstrip(".")  # noqa: E731
    out = pd.DataFrame({
        "股票": df["代號"] + " " + df["名稱"],
        "收盤": df["收盤"].map(p),
        "距60日高": df["距高點%"].map(lambda x: f"{x:+.1f}%"),
        "觸發價": df["觸發價"].map(p),
    })
    for c in extra:
        out[c] = df[c]
    return out


def render(get_prices):
    st.markdown("### 🪂 跌深清單｜AI＋自選股")
    st.caption("回檔布局＋閒置資金放 0050（測試中）・只列清單，不是保證獲利的買點")

    with st.form("dip_form"):
        c1, c2 = st.columns(2)
        drop = c1.number_input("跌幅門檻（%）", 10.0, 60.0, 35.0, 5.0)
        per = c2.number_input("每筆金額（元）", 10_000, 2_000_000, 100_000, 10_000)
        go = st.form_submit_button(f"列出跌 {drop:g}% 以上的股票", type="primary", use_container_width=True)

    stocks = pool_stocks()
    key = (tuple((s["code"], s["market"]) for s in stocks), drop, per)
    if go:
        with st.spinner(f"抓取 {len(stocks)} 檔最新日K…"):
            prices = get_prices(key[0])
        st.session_state["dip_res"] = (key, scan_dip(stocks, prices, drop, per))
    saved = st.session_state.get("dip_res")
    if not saved or saved[0] != key:
        st.info(f"按上面的按鈕，掃描「⭐ 自選股」＋「AI 產業」共 {len(stocks)} 檔（收盤後資料）。")
        _rules()
        return

    hit, near, fail, last = saved[1]
    if last is not None:
        lag = (pd.Timestamp.today().normalize() - last).days
        st.caption(f"資料日期：{last:%Y-%m-%d}（Yahoo Finance 盤後日K）" + ("　⚠️ 資料超過 5 天未更新" if lag > 5 else ""))

    st.markdown(f"#### ✅ 已跌到 {drop:g}% 以上：{len(hit)} 檔")
    if hit:
        t = _fmt(hit)
        t["參考數量"] = [_qty(r["參考股數"]) for r in hit]
        t["停利參考"] = [f"{r['停利參考(淨賺10%)']:g}" for r in hit]
        t["停損參考"] = [f"{r['停損參考(−20%)']:g}" for r in hit]
        st.dataframe(t, hide_index=True, use_container_width=True)
        st.caption("數量、停利、停損用今天收盤價估算；實際依明天開盤成交價重算。")
        if len(hit) > 5:
            st.warning(f"共 {len(hit)} 檔，規則最多同時 5 檔；而且多半是同一波族群下跌，不算分散。")
    else:
        st.write("今天沒有股票跌到門檻，資金維持在 0050。")

    st.markdown(f"#### ⏳ 接近中（還差 8 個百分點內，**不是買點**）：{len(near)} 檔")
    if near:
        st.dataframe(_fmt(near, extra=("還差(百分點)",)), hide_index=True, use_container_width=True)
    if fail:
        st.caption("資料抓取失敗：" + "、".join(fail))
    _rules()


def _rules():
    with st.expander("操作規則與回測依據"):
        st.markdown(
            "**操作規則（測試中）**\n"
            "1. 平常資金全部放 0050。\n"
            "2. 收盤距近 60 日最高點跌 ≥ 35%，且手上沒有這檔 → 隔天開盤賣掉 10 萬 0050，買進這檔。\n"
            "3. 同時最多 5 檔。\n"
            "4. 扣手續費與稅淨賺 10% 賣出／買價 −20% 停損／最多抱 60 個交易日。\n"
            "5. 賣出後的錢買回 0050。\n\n"
            "**回測（還原權息、扣手續費、證交稅、滑價）**\n"
            "- 自選＋AI 52 檔、10 年、50 萬帳戶：年化 +30.7%（0050 放著 +23.8%），最大回撤 −33%（0050 −34%）。\n"
            "- 2024～今只盯單一檔：AI 40 檔中 24 檔贏 0050、16 檔輸，平均只多約 7 千元；"
            "必須同時掃描很多檔才有效果。\n"
            "- 限制：股票清單是今天挑的強勢股（存活者偏差），結果可能偏樂觀；用 331 檔大池子時沒有贏 0050。"
        )
