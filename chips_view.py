"""
大盤籌碼區塊（畫面）。與「股票系統」App 相同：證交所三大法人、期交所台指期三大法人。
僅供對照，不參與進場判斷。
"""
import streamlit as st

import market_chips as MC

@st.cache_data(ttl=1800, show_spinner=False)
def get_spot_chips():
    return MC.spot_institutions(10)


@st.cache_data(ttl=1800, show_spinner=False)
def get_fut_chips():
    df, src, errs = MC.futures_institutions(10)
    px, perr = MC.futures_close(10)
    return df, src, errs + ([perr] if perr else []), px


@st.cache_data(ttl=1800, show_spinner=False)
def get_taiex():
    return MC.taiex_close()


def tw_color(v):
    """台股慣例：正數紅、負數綠。"""
    try:
        return "color:#e53935" if v > 0 else "color:#2e9d5b" if v < 0 else ""
    except TypeError:
        return ""


def chip_chart(df, cols, unit, line_s, line_name):
    try:
        _chip_chart(df, cols, unit, line_s, line_name)
    except Exception:  # noqa: BLE001
        st.bar_chart(df.set_index("日期")[cols])


def _chip_chart(df, cols, unit, line_s, line_name):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = df["日期"].dt.strftime("%m/%d")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=df["合計"], name="合計", marker_color="rgba(56,140,230,.75)"))
    for c, clr in zip(cols, ("#e53935", "#2e9d5b", "#f39c12")):
        fig.add_trace(go.Scatter(x=x, y=df[c], name=c, mode="lines+markers", line=dict(color=clr, width=2)))
    if line_s is not None and len(line_s):
        ys = [line_s.get(d) for d in df["日期"]]
        fig.add_trace(go.Scatter(x=x, y=ys, name=line_name, mode="lines+markers",
                                 line=dict(color="#f1c40f", width=2, dash="dot")), secondary_y=True)
    fig.update_layout(height=330, margin=dict(l=4, r=4, t=8, b=4), dragmode=False,
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)))
    fig.update_yaxes(title_text=unit, secondary_y=False, zeroline=True)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chip_table(df, cols, line_s, line_name, fmt):
    t = df[["日期"] + cols + ["合計"]].copy()
    if line_s is not None and len(line_s):
        t[line_name] = [line_s.get(d) for d in t["日期"]]
    t = t.iloc[::-1]
    t["日期"] = t["日期"].dt.strftime("%m/%d")
    sty = (t.style.format({c: fmt for c in cols + ["合計"]} | ({line_name: "{:,.0f}"} if line_name in t else {}),
                          na_rep="—")
           .map(tw_color, subset=cols + ["合計"]))
    st.dataframe(sty, hide_index=True, use_container_width=True)


def render_chips():
    st.markdown("#### 大盤籌碼（官方盤後資料）")
    st.caption("證交所、期交所每天約 15:00 後公布；30 分鐘自動重新抓取。是盤後資料，不是盤中即時。")
    with st.spinner("抓取三大法人資料…"):
        spot, latest, s_err = get_spot_chips()
        fut, f_src, f_err, fut_px = get_fut_chips()
        taiex = get_taiex()

    # ---- 外資今昨比較 ----
    m1, m2 = st.columns(2)
    if spot is not None and len(spot) >= 2:
        a, b = spot.iloc[-1], spot.iloc[-2]
        m1.metric(f"外資現貨買賣超（{a['日期']:%m/%d}）", f"{a['外資']:,.1f} 億",
                  f"{a['外資'] - b['外資']:+,.1f} 億 vs 前日", delta_color="inverse")
    if fut is not None and len(fut) >= 2:
        a, b = fut.iloc[-1], fut.iloc[-2]
        m2.metric(f"外資台指期淨未平倉（{a['日期']:%m/%d}）", f"{a['外資']:,.0f} 口",
                  f"{a['外資'] - b['外資']:+,.0f} 口 vs 前日", delta_color="inverse")
    st.caption("數字為負＝賣超／淨空單；台股慣例紅色為正、綠色為負。")

    t1, t2, t3 = st.tabs(["現貨三大法人", "期貨三大法人", "當日買賣金額"])
    with t1:
        if spot is None:
            st.warning("無法取得證交所三大法人資料。")
            st.caption("；".join(s_err[:3]))
        else:
            chip_chart(spot, ["外資", "投信", "自營"], "億元", taiex, "加權指數")
            chip_table(spot, ["外資", "投信", "自營"], taiex, "加權指數", "{:,.2f}")
            st.caption("來源：臺灣證券交易所 BFI82U（上市）。外資含外資自營商；自營含自行買賣與避險。單位：億元。")
    with t2:
        if fut is None:
            st.warning("無法取得期交所臺股期貨三大法人資料。")
            st.caption("；".join(f_err[:3]))
        else:
            use_px, nm = (fut_px, "期貨收盤") if fut_px is not None and len(fut_px) else (taiex, "加權指數")
            chip_chart(fut, ["外資", "投信", "自營"], "口", use_px, nm)
            chip_table(fut, ["外資", "投信", "自營"], use_px, nm, "{:,.0f}")
            st.caption(f"來源：{f_src}。臺股期貨（TX）未平倉多空淨額，單位：口。"
                       + ("期貨收盤價取得失敗，改顯示加權指數。" if nm == "加權指數" else "期貨收盤為近月一般盤。"))
    with t3:
        if latest is None:
            st.warning("無法取得當日三大法人買賣金額。")
        else:
            d, tb = latest
            st.markdown(f"**{d:%Y/%m/%d} 三大法人買賣金額統計（上市）**")
            sty = (tb.style.format({"買進金額": "{:,.0f}", "賣出金額": "{:,.0f}", "買賣差額": "{:,.0f}"})
                   .map(tw_color, subset=["買賣差額"]))
            st.dataframe(sty, hide_index=True, use_container_width=True)
            st.caption("單位：元。來源：臺灣證券交易所。")

