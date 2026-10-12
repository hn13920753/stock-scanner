"""
「進場紀錄與驗證」頁面（畫面）。資料由 GitHub Actions 每個交易日收盤後自動更新（records/signals.csv）。
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

import records as R

RAW_URL = "https://raw.githubusercontent.com/hn13920753/stock-scanner/main/records/signals.csv"
LOCAL = Path(__file__).with_name("records") / "signals.csv"


@st.cache_data(ttl=600, show_spinner=False)
def load_book():
    try:
        import requests

        resp = requests.get(RAW_URL, timeout=10)
        if resp.ok and resp.text.strip():
            return R.load(io.StringIO(resp.content.decode("utf-8-sig"))), "GitHub（最新）"
    except Exception:  # noqa: BLE001
        pass
    return R.load(LOCAL), "App 內建檔案"


def render():
    st.markdown("### 📒 進場紀錄與驗證")
    st.caption("每個交易日 15:15 自動掃描全部產業：把「可進場」的股票當天就記下來，之後用真實股價驗證結果。"
               "紀錄不能事後修改或挑選，用來檢驗系統規則是否真的有效。")
    book, src = load_book()
    if book.empty:
        st.info("還沒有紀錄。第一筆會在下一個交易日收盤後自動產生。")
        return
    sm = R.summary(book)
    c = st.columns(4)
    c[0].metric("累計訊號", sm["signals"])
    c[1].metric("持有中", sm["open"])
    c[2].metric("已出場", sm["closed"])
    c[3].metric("未成交", sm["skipped"])

    st.markdown("#### 實際結果（已出場）")
    if sm["closed"] == 0:
        st.write("還沒有已出場的紀錄。每筆最長 15 個交易日會有結果。")
    else:
        note = "" if sm["closed"] >= 30 else f"・目前只有 {sm['closed']} 筆，少於 30 筆僅供參考"
        k = st.columns(3)
        k[0].metric("先到停利", f"{sm['tp_rate']:.0f}%")
        k[1].metric("平均每筆淨報酬", f"{sm['avg']:+.2f}%")
        k[2].metric("平均持有", f"{sm['avg_days']:.1f} 天")
        st.caption(f"停損 {sm['sl_rate']:.0f}%・淨報酬為正 {sm['win_rate']:.0f}%・平均獲利 {sm['avg_win']:+.2f}%／"
                   f"平均虧損 {sm['avg_loss']:+.2f}%{note}")
        done = book[book["狀態"] == "已出場"].copy()
        done["回測勝率%"] = pd.to_numeric(done["回測勝率%"], errors="coerce")
        g = done.groupby("買點")["淨報酬%"].agg(["count", "mean"]).round(2)
        g.columns = ["筆數", "平均淨報酬%"]
        st.markdown("**依買點類型**")
        st.dataframe(g, use_container_width=True)

    show = ["訊號日", "代號", "名稱", "狀態", "結果", "進場價", "計畫停損", "停利價", "出場價", "淨報酬%",
            "持有天數", "最大不利%", "風報比", "回測勝率%", "子類"]
    st.markdown("#### 持有中・待進場")
    live = book[book["狀態"].isin(["持有中", "待進場"])]
    if live.empty:
        st.write("（無）")
    else:
        st.dataframe(live[show].iloc[::-1], hide_index=True, use_container_width=True)
        st.caption("持有中的「淨報酬%」是以最新收盤計算的帳面報酬。")
    st.markdown("#### 全部紀錄")
    st.dataframe(book[show].iloc[::-1], hide_index=True, use_container_width=True)
    st.download_button("下載全部紀錄 CSV", book.to_csv(index=False).encode("utf-8-sig"),
                       "signals.csv", "text/csv", use_container_width=True)
    st.caption(f"資料來源：{src}。驗證規則：訊號隔日開盤進場（開盤高於訊號價 1% 以上不追）→ 停利依實際進場價重算淨 3% → "
               "停損用訊號當天計畫 → 15 個交易日內先碰停損或停利，同日都碰算停損 → 期滿收盤出場。已扣手續費、證交稅，未含滑價。")
