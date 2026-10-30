"""
產業掃描器（Streamlit）
選產業 → 批次抓整個產業的日K → 每檔套用「股票系統」同一套規則 → 列出今天「可進場」的股票（最多 10 檔）。
判斷邏輯全部在 engine.py（與股票系統 App 相同），本檔只負責畫面。
"""
import datetime as dt
import importlib

import pandas as pd
import streamlit as st

import chips_view as CV
import dip_view as DV
import engine as E
import market_chips as MC
import records as REC
import records_view as RECV
import report_view as RV
import scanner as S
import sectors as SEC

# Streamlit Cloud 更新程式時，可能先換掉 app.py、卻還在用舊版的其他檔案（造成 ImportError）。
# 每次執行都重新載入本專案的模組，確保所有檔案是同一個版本。
for _m in (E, MC, SEC, S, REC, RV, CV, RECV, DV):
    importlib.reload(_m)
SECTORS, parse_csv, stocks_of = SEC.SECTORS, SEC.parse_csv, SEC.stocks_of

st.set_page_config(page_title="產業掃描", page_icon="🔎", layout="centered")

st.markdown("""
<style>
.block-container{padding-top:4.2rem;padding-bottom:3rem;max-width:760px}
.verdict{border-radius:16px;padding:18px 18px 14px;margin:6px 0 12px;color:#fff}
.verdict .big{font-size:34px;font-weight:800;line-height:1.1}
.verdict .sub{font-size:14px;opacity:.92;margin-top:6px;line-height:1.5}
.v-green{background:#1f8a5b}.v-orange{background:#b7791f}.v-red{background:#c0392b}
.kpi{border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:10px 12px;margin-bottom:8px}
.kpi .lab{font-size:12px;opacity:.75}.kpi .val{font-size:24px;font-weight:700}
.kpi .note{font-size:12px;opacity:.75}
.small{font-size:12px;opacity:.75;line-height:1.6}
.pick{border:1px solid rgba(31,138,91,.55);border-left:6px solid #1f8a5b;border-radius:12px;
      padding:10px 12px;margin:8px 0}
.pick .t{font-size:17px;font-weight:700}.pick .g{font-size:12px;opacity:.7;font-weight:400}
.pick .row{font-size:14px;line-height:1.7;margin-top:2px}
.pick .m{font-size:12px;opacity:.8;margin-top:2px}
.sum{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px}
.sum div{flex:1;min-width:70px;border-radius:10px;padding:8px;text-align:center;color:#fff}
.sum b{display:block;font-size:22px}.sum span{font-size:12px}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(stock_key):
    """stock_key = ((代號, 市場), ...)；同一份清單一小時內只抓一次。"""
    return S.fetch_batch([{"code": c, "market": m} for c, m in stock_key])


@st.cache_data(ttl=3600, show_spinner=False)
def get_market():
    """回傳 (加權指數 10 年收盤序列, 大盤環境 dict)。"""
    idx = E.load_index(E.DATA_YEARS)
    return idx, E.market_env(idx)


price, money = RV.price, RV.money


def qty_text(shares):
    lots, odd = divmod(int(shares), 1000)
    return ((f"{lots} 張" if lots else "") + (f" {odd} 股" if odd else "")).strip() or "0 股"


def pool_text(s):
    if s.get("n", 0) < S.MIN_SAMPLES:
        return f"樣本 {s.get('n', 0)} 筆（不計勝率）"
    return (f"{s['stocks']} 檔共 {s['n']} 筆・先到停利 {s['tp_rate']:.0f}%・平均淨報酬 {s['avg']:+.2f}%"
            + ("" if s["n"] >= 30 else "・僅供參考"))


def win_text(r):
    s = r["stats"]
    if s.get("n", 0) == 0:
        return "無歷史同類訊號"
    if s["n"] < S.MIN_SAMPLES:
        return f"樣本 {s['n']} 筆（不計勝率）"
    return f"{s['tp_rate']:.0f}%（{s['n']} 筆{'' if s['n'] >= 30 else '・僅供參考'}）"


# ------------------------------------------------------------------
page = st.radio("切換頁面", ["🔎 產業掃描", "🪂 跌深清單", "📒 進場紀錄與驗證"], horizontal=True)
if page.startswith("📒"):
    RECV.render()
    st.stop()
if page.startswith("🪂"):
    DV.render(get_prices)
    st.stop()

st.markdown("### 🔎 產業掃描｜今日可進場")
st.caption("同一套股票系統規則・短線買點＋回檔布局（測試中）・風控優先")

with st.form("q"):
    sector = st.selectbox("產業", list(SECTORS.keys()))
    with st.expander("進階設定（通常不用改）"):
        c1, c2 = st.columns(2)
        capital = c1.number_input("每檔投入上限（元）", 1000, 10_000_000, 50_000, 1000)
        max_loss = c2.number_input("每檔最大損失（元）", 100, 1_000_000, 1_500, 100)
        _t = [10.0, 5.0, 3.0, 2.0]
        _h = [60, 30, 15, 10, 5]
        target = c1.selectbox("淨獲利目標（短線買點）", _t, index=_t.index(E.DEFAULT_PARAMS["target_net"]),
                              format_func=lambda x: f"{x:g}%")
        hold = c2.selectbox("最長持有（交易日）", _h, index=_h.index(E.DEFAULT_PARAMS["hold_days"]))
        discount = c1.number_input("手續費折扣（6 折填 0.6）", 0.1, 1.0, 1.0, 0.05)
        min_rr = c2.number_input("最低風險報酬比", 0.5, 3.0, 1.0, 0.1)
        st.markdown("**過濾與積極組（測試中）**")
        f1 = st.checkbox("停損距離至少 1 倍 ATR（太近不進場）", value=E.DEFAULT_PARAMS["f_stop_atr"])
        f2 = st.checkbox("大盤空頭時股數減半", value=E.DEFAULT_PARAMS["f_market"])
        f3 = st.checkbox("族群明顯弱於大盤時等待", value=E.DEFAULT_PARAMS["f_sector"])
        f4 = st.checkbox("積極組：高波動個股改用寬停損 2 ATR、淨利 8% 出場、固定最大損失（測試中・紙上交易驗證中）", value=E.DEFAULT_PARAMS["f_aggr"])
        aggr_loss = st.number_input("積極組單筆最大損失（元）", 1000, 1_000_000, E.DEFAULT_PARAMS["aggr_max_loss"], 1000)
        st.markdown("**回檔布局（測試中）**")
        f5 = st.checkbox("回檔布局：離 60 日高點跌到門檻就買、淨賺 10% 賣、−20% 停損、最多 60 天（高波動股也適用）",
                         value=E.DEFAULT_PARAMS["f_dip"])
        c3, c4 = st.columns(2)
        dip_s = c3.number_input("個股跌幅門檻（%）", 10.0, 60.0, E.DEFAULT_PARAMS["dip_drop_stock"], 5.0)
        dip_e = c4.number_input("ETF 跌幅門檻（%）", 5.0, 60.0, E.DEFAULT_PARAMS["dip_drop_etf"], 5.0)
        dip_cap = c3.number_input("回檔布局每筆金額（元）", 10_000, 5_000_000, E.DEFAULT_PARAMS["dip_capital"], 10_000)
        csv = st.file_uploader("自訂清單 CSV（選填，上傳後改掃這份清單）：欄位 代號,名稱,分類,市場（TW/TWO）",
                               type=["csv"])
    go = st.form_submit_button("掃描", type="primary", use_container_width=True)

params = {"capital": capital, "max_loss": max_loss, "target_net": target, "hold_days": hold,
          "fee_discount": discount, "min_rr": min_rr, "f_stop_atr": f1, "f_market": f2, "f_sector": f3,
          "f_aggr": f4, "aggr_max_loss": aggr_loss, "f_dip": f5, "dip_drop_stock": dip_s,
          "dip_drop_etf": dip_e, "dip_capital": dip_cap}

if go:
    if csv is not None:
        try:
            custom = parse_csv(csv)
        except Exception as e:  # noqa: BLE001
            st.error(f"自訂清單讀取失敗：{e}")
            st.stop()
        st.session_state["custom"] = custom
        sector = f"自訂清單（{csv.name}）"
    else:
        st.session_state.pop("custom", None)
    st.session_state["req"] = (sector, tuple(sorted(params.items())))

req = st.session_state.get("req")
if not req:
    n = len(stocks_of(sector))
    st.markdown(f'<div class="small">選產業後按「掃描」。目前「{sector}」清單共 {n} 檔，第一次約需 1～2 分鐘，'
                '同一小時內再掃會快很多。資料為收盤後日K，建議收盤後使用。研究用途，不是下單指令。</div>',
                unsafe_allow_html=True)
    st.stop()

sector, ptuple = req
if sector not in SECTORS and not st.session_state.get("custom"):
    st.session_state.pop("req", None)
    st.rerun()
params = dict(ptuple)
target, hold = params["target_net"], params["hold_days"]
stocks = st.session_state.get("custom") or stocks_of(sector)
stock_key = tuple((s["code"], s["market"]) for s in stocks)
key = (sector, stock_key, ptuple, dt.datetime.now().strftime("%Y%m%d%H"))

# ---------------- 掃描（同一小時、同設定只跑一次）----------------
if st.session_state.get("scan_key") != key:
    with st.spinner(f"抓取 {len(stocks)} 檔歷史股價…"):
        prices = get_prices(stock_key)
        mkt, market = get_market()
        ctx = S.sector_context(stocks, prices)
    bar = st.progress(0.0, text="套用股票系統規則中…")
    rows = []
    for k, s in enumerate(stocks):
        bar.progress((k + 1) / len(stocks), text=f"分析 {s['code']} {s['name']}（{k + 1}/{len(stocks)}）")
        rows.append(S.analyze_one(s, prices.get(s["code"]), params, market, mkt, ctx.get(s["code"])))
    bar.empty()
    st.session_state.update(scan_key=key, rows=rows, prices=prices, market=market)

rows = st.session_state["rows"]
prices = st.session_state["prices"]
market = st.session_state["market"]
res = S.classify(rows)
picks, watch = res["picks"], res["watch"]

# ---------------- 摘要 ----------------
dates = [x["r"]["date"] for x in rows if x["r"]]
data_date = max(dates) if dates else "—"
st.markdown(f"**{sector}**　掃描 {len(stocks)} 檔・資料日期 {data_date}")
st.markdown(
    '<div class="sum">'
    f'<div style="background:#1f8a5b"><b>{res["picks_total"]}</b><span>可進場</span></div>'
    f'<div style="background:#b7791f"><b>{res["wait_n"]}</b><span>等待</span></div>'
    f'<div style="background:#c0392b"><b>{res["veto_n"]}</b><span>不可進場</span></div>'
    f'<div style="background:#7f8c8d"><b>{len(res["fail"])}</b><span>資料失敗</span></div>'
    '</div>', unsafe_allow_html=True)
ps = S.pooled_stats(rows)
st.caption(f"本類別合併回測（全部 {len(stocks)} 檔的歷史同類訊號）：{pool_text(ps)}")
if market:
    st.caption(f"大盤（加權指數 {market['date']}）：{market['state']}・收盤 {market['close']:,.0f}・"
               f"MA20 {market['ma20']:,.0f}・MA60 {market['ma60']:,.0f}")
else:
    st.caption("大盤環境資料抓取失敗，模組 01 標示為無法判定。")

# ---------------- 可進場 ----------------
st.markdown(f"#### ✅ 今日可進場（前 {S.TOP_N} 檔）")
if not picks:
    st.info("今天這個產業沒有任何股票同時通過全部規則與風控，所以不列買點（不硬湊數量）。"
            "可以看下方「等待中、最接近觸發」的名單，觸發當天收盤後再掃一次。")
else:
    if res["picks_total"] > S.TOP_N:
        st.caption(f"共 {res['picks_total']} 檔可進場，依排序規則只列前 {S.TOP_N} 檔。")
    for no, x in enumerate(picks, 1):
        r, pl = x["r"], x["r"]["plan"]
        early = ("・股數已減半（" + "、".join(h.split("（")[0] for h in pl["halve"]) + "）") if pl.get("halve") else ""
        sg = r["signal"]
        tag = "、".join(E.PATTERNS[k] for k in sg["ids"]) if sg["ids"] else sg.get("note", "")
        buy = ("限價掛單" if sg.get("mode") == "limit" else "買進約")
        st.markdown(
            f'<div class="pick"><div class="t">#{no}　{x["code"]} {x["name"]}　'
            f'<span class="g">{x["group"]}</span>{"　⚡積極組" if pl.get("aggr") else ""}</div>'
            f'<div class="row">收盤 {price(r["close"])}（{r["chg"]:+.2f}%）・{buy} <b>{price(pl["entry"])}</b>'
            f'・停利 <b>{price(pl["tp3"])}</b>・停損 <b>{price(pl["stop"])}</b></div>'
            f'<div class="row">風報比 <b>{pl["rr"]:.2f}</b>・回測勝率 {win_text(r)}・符合度 {r["fit"]:.0f}%</div>'
            f'<div class="m">同子類合併回測：{pool_text(S.pooled_stats(rows, x["group"]))}</div>'
            f'<div class="m">{qty_text(pl["shares"])}（約 {money(pl["cost"])} 元，最大損失約 {money(pl["max_loss"])} 元）'
            f'{early}・{sg["type"]}・{tag}</div>'
            f'<div class="m">過濾（測試中）：{"；".join(r["filters"]["info"]) or "—"}</div></div>',
            unsafe_allow_html=True)
    if len(picks) > 1:
        st.warning("這些股票都屬同一個產業鏈，股價常同漲同跌。同時買多檔不等於分散風險；"
                   "「每檔最大損失」是每檔各自計算，全部停損時的總損失會加總。")
    st.caption("排序：① 風險報酬比 ② 回測勝率（樣本 ≥ 10 筆才採用）③ 規則符合度。"
               "收盤型買點：明日開盤高於買進價 1% 以上不追；掛單型買點：3 個交易日內有效。")

# ---------------- 等待中、最接近觸發 ----------------
st.markdown("#### ⏳ 等待中、最接近觸發（不是買點）")
if not watch:
    st.write("（無）")
else:
    st.caption("以下股票今天「不可買」，只是距離觸發價最近。觸發當天收盤後再掃描一次，系統才會算出停損與數量。")
    wt = pd.DataFrame([{
        "代碼": x["code"], "名稱": x["name"], "收盤": price(x["r"]["close"]),
        "觸發價": price(x["r"]["trigger"]["price"]), "差距": f"{S.trigger_gap(x):+.1f}%",
        "需要量(張)": money(x["r"]["trigger"]["vol"]),
    } for x in watch])
    st.dataframe(wt, hide_index=True, use_container_width=True)
    st.caption("觸發 = 收盤站上觸發價、收紅、成交量達「需要量」。各股完整條件請在下方單檔報告查看。")

# ---------------- 接近回檔布局價 ----------------
dipw = res.get("dip_watch") or []
if dipw:
    st.markdown("#### 🪂 接近回檔布局價（測試中，不是買點）")
    st.caption("離近 60 日高點的跌幅還差 8 個百分點以內。收盤跌到「布局價」以下，隔天開盤可買（淨賺 10% 賣、−20% 停損、最多 60 天）。")
    st.dataframe(pd.DataFrame([{
        "代碼": x["code"], "名稱": x["name"], "收盤": price(x["r"]["close"]),
        "距高點": f"{x['r']['dist_high']:+.1f}%", "布局價": price(E.round_down(x["r"]["dip_level"], E.is_etf(x["code"]))),
        "還差": f"{x['r']['dist_high'] + x['r']['dip_drop']:.1f} 個百分點",
    } for x in dipw]), hide_index=True, use_container_width=True)

# ---------------- 單檔完整報告 ----------------
st.markdown("#### 📄 單檔完整報告")
shown = {x["code"] for x in picks + watch}
order = picks + watch + [x for x in dipw if x["code"] not in shown]
shown |= {x["code"] for x in dipw}
order += [x for x in rows if x["r"] and x["code"] not in shown]
label = {x["code"]: f"{x['code']} {x['name']}｜{x['decision']}" for x in order}
pick_code = st.selectbox("選一檔看完整分析（10 模組、18 型態、回測、K 線）",
                         ["（請選擇）"] + [x["code"] for x in order],
                         format_func=lambda c: label.get(c, c))
if pick_code != "（請選擇）":
    x = next(v for v in order if v["code"] == pick_code)
    df, src, _ = prices[pick_code]
    try:
        an = E.Analyzer(df, pick_code, params)  # 只用來畫 K 線（指標），判斷沿用掃描結果
        RV.render_report(x["r"], an, df, src, target, hold, title=f"{x['code']} {x['name']}")
    except Exception as e:  # noqa: BLE001
        st.error(f"報告顯示失敗：{e}")

# ---------------- 全部結果 ----------------
with st.expander(f"全部 {len(rows)} 檔結果"):
    allt = pd.DataFrame([{
        "代碼": x["code"], "名稱": x["name"], "子類": x["group"], "決策": x["decision"],
        "收盤": price(x["r"]["close"]) if x["r"] else "—",
        "符合度": f"{x['r']['fit']:.0f}%" if x["r"] else "—",
        "主要原因": ("；".join(x["r"]["why"][:1]) if x["r"] and x["r"]["why"] else "") or x["error"][:40],
    } for x in rows])
    st.dataframe(allt, hide_index=True, use_container_width=True)
    if res["fail"]:
        st.caption("資料失敗：" + "、".join(f"{x['code']} {x['name']}" for x in res["fail"])
                   + "。可能是代碼變更、下市或資料源暫時無回應。")
    st.download_button("下載完整結果 CSV（複盤紀錄用）",
                       S.export_table(rows, sector, params).to_csv(index=False).encode("utf-8-sig"),
                       f"scan_{data_date}.csv", "text/csv", use_container_width=True)

# ---------------- 大盤籌碼 ----------------
if st.toggle("顯示大盤籌碼（三大法人）"):
    try:
        CV.render_chips()
    except Exception as e:  # noqa: BLE001
        st.warning(f"大盤籌碼區塊載入失敗：{str(e)[:120]}")

if st.button("重新掃描（忽略快取）"):
    get_prices.clear()
    get_market.clear()
    st.session_state.pop("scan_key", None)
    st.rerun()

flt = "、".join(f"{v[0]}{'（開）' if params.get(k, E.DEFAULT_PARAMS.get(k)) else '（關）'}" for k, v in E.FILTERS.items())
st.markdown(f'<div class="small">引擎 {E.ENGINE_VERSION}・三項過濾（測試中）：{flt}・產業清單為自訂分類（sectors.py），非官方分類。'
            '未納入進場判斷：個股法人籌碼、財報事件、週K。研究用途，不是投資建議或下單指令。</div>',
            unsafe_allow_html=True)
