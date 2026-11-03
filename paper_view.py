"""💰 模擬交易：100 萬虛擬帳戶，自己決定買賣、每筆記錄，對照股票系統的買訊，看操作哪裡要修正。

- 股票：App 裡所有產業的全部股票（sectors.py）。
- 報價：證交所即時（約延遲 5～20 秒），抓不到改用 Yahoo（延遲約 20 分鐘）。
- 存檔：GitHub 倉庫 paper-data 分支的 paper/trades.json（paper_store.py）。
- 成本：手續費 0.1425% × 折扣（整股最低 20 元、零股最低 1 元），賣出證交稅個股 0.3%、ETF 0.1%。
只是紀錄與計算，不是買賣建議，也不會真的下單。
"""
import datetime as dt
import math
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

import paper_store as PS
import quotes as Q
import sectors as SEC

ROOT = Path(__file__).resolve().parent
TZ = dt.timezone(dt.timedelta(hours=8))


# ------------------------------------------------------------------ 基本計算
def is_etf(code: str) -> bool:
    return str(code).startswith("00")


def fee(amount: float, shares: int, disc: float) -> int:
    return max(20 if shares >= 1000 else 1, math.floor(amount * 0.001425 * (disc or 1)))


def buy_cost(p, s, disc):
    a = p * s
    return a + fee(a, s, disc)


def sell_net(code, p, s, disc):
    a = p * s
    return a - fee(a, s, disc) - math.floor(a * (0.001 if is_etf(code) else 0.003))


def qty(s: int) -> str:
    lots, odd = divmod(int(s), 1000)
    return ((f"{lots} 張" if lots else "") + (f" {odd} 股" if odd else "")).strip() or "0 股"


def money(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.0f}"


def signed(x) -> str:
    return "—" if x is None else f"{x:+,.0f}"


def price_txt(x) -> str:
    return "—" if x is None else f"{x:,.2f}".rstrip("0").rstrip(".")


def now_tw():
    return dt.datetime.now(TZ)


def market_open() -> bool:
    n = now_tw()
    return n.weekday() < 5 and dt.time(9, 0) <= n.time() <= dt.time(13, 35)


def positions(data: dict, quotes: dict):
    disc = data.get("disc", 1.0)
    cash, realized = float(data.get("capital", 1000000)), 0.0
    book, closed = {}, []
    for t in sorted(data["trades"], key=lambda x: (x["date"], x.get("createdAt", ""))):
        p = book.setdefault(t["code"], {"code": t["code"], "name": t.get("name", ""), "shares": 0, "cost": 0.0,
                                        "stop": None, "target": None, "follow": t.get("follow"), "first": t["date"]})
        if t.get("name"):
            p["name"] = t["name"]
        if t["side"] == "buy":
            c = buy_cost(t["price"], t["shares"], disc)
            if p["shares"] == 0:
                p["first"], p["follow"] = t["date"], t.get("follow")
            cash -= c
            p["shares"] += t["shares"]
            p["cost"] += c
            p["stop"] = t.get("stop") or p["stop"]
            p["target"] = t.get("target") or p["target"]
        else:
            s = min(t["shares"], p["shares"])
            if s <= 0:
                continue
            net = sell_net(t["code"], t["price"], s, disc)
            out = p["cost"] / p["shares"] * s
            cash += net
            realized += net - out
            closed.append({"code": t["code"], "name": p["name"], "follow": p["follow"], "pnl": net - out,
                           "ret": (net - out) / out * 100 if out else 0, "date": t["date"]})
            p["shares"] -= s
            p["cost"] -= out
    opens = []
    mv = unreal = 0.0
    for p in book.values():
        if p["shares"] <= 0:
            continue
        q = quotes.get(p["code"])
        p["avg"] = p["cost"] / p["shares"]
        p["last"] = q["price"] if q else None
        p["prev"] = q.get("prev") if q else None
        if p["last"] is not None:
            p["mv"] = sell_net(p["code"], p["last"], p["shares"], disc)
            p["pnl"] = p["mv"] - p["cost"]
            p["ret"] = p["pnl"] / p["cost"] * 100
        else:
            p["mv"], p["pnl"], p["ret"] = p["cost"], None, None
        mv += p["mv"]
        unreal += p["pnl"] or 0
        opens.append(p)
    return {"open": opens, "cash": cash, "mv": mv, "realized": realized, "unreal": unreal, "closed": closed}


# ------------------------------------------------------------------ 資料
def universe():
    out = {}
    for sec in SEC.SECTORS:
        for s in SEC.stocks_of(sec):
            out.setdefault(s["code"], {"code": s["code"], "name": s["name"], "secs": []})["secs"].append(sec)
    return out


@st.cache_data(ttl=15, show_spinner=False)
def cached_quotes(codes: tuple):
    return Q.get_quotes(list(codes)), now_tw().strftime("%H:%M:%S")


@st.cache_data(ttl=600, show_spinner=False)
def dip_signals(days: int = 15):
    """records/signals.csv 裡最近的回檔布局買訊。"""
    f = ROOT / "records" / "signals.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    df = df[df["買點"].fillna("").str.contains("回檔布局")]
    if df.empty:
        return df
    dates = sorted(df["訊號日"].unique())[-days:]
    df = df[df["訊號日"].isin(dates)].copy()
    for c in ("訊號收盤", "計畫停損", "計畫停利3%"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def store() -> PS.Store:
    try:
        tok = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("PAPER_REPO", PS.DEFAULT_REPO)
        br = st.secrets.get("PAPER_BRANCH", PS.DEFAULT_BRANCH)
    except Exception:  # noqa: BLE001  沒有 secrets 檔
        tok, repo, br = None, PS.DEFAULT_REPO, PS.DEFAULT_BRANCH
    return PS.Store(tok, repo, br)


def load(force=False):
    ss = st.session_state
    if force or "paper" not in ss:
        try:
            ss.paper, ss.paper_sha = store().load()
            ss.paper_err = ""
        except Exception as e:  # noqa: BLE001
            ss.setdefault("paper", {"capital": 1000000, "disc": 1.0, "trades": []})
            ss.setdefault("paper_sha", None)
            ss.paper_err = f"讀不到存檔：{e}"
    return ss.paper


def commit(new_data: dict, msg: str) -> bool:
    ss = st.session_state
    try:
        ss.paper_sha = store().save(new_data, ss.get("paper_sha"), msg)
        ss.paper = new_data
        return True
    except PS.ConflictError as e:
        load(force=True)
        st.error(str(e))
    except PermissionError as e:
        st.error(f"沒有存進去：{e}。請看下方「存檔設定」。")
    except Exception as e:  # noqa: BLE001
        st.error(f"沒有存進去：{e}")
    return False


# ------------------------------------------------------------------ 畫面
def render():
    st.markdown("### 💰 模擬交易｜100 萬虛擬帳戶")
    st.caption("自己決定買賣，每一筆都記下來，再對照股票系統的買訊。虛擬帳戶，不會真的下單。")
    data = load()
    stor = store()
    if st.session_state.get("paper_err"):
        st.warning(st.session_state.paper_err)
    if not stor.can_write:
        st.warning("還沒設定存檔金鑰：現在可以看，但新的買賣存不進去。設定方法在最下面「存檔設定」。")

    U = universe()
    sig = dip_signals()
    held = {t["code"] for t in data["trades"]}
    want = sorted(held | set(sig["代號"]) if len(sig) else held)

    top = st.columns([3, 2])
    auto = top[1].toggle("盤中每 30 秒自動更新", value=market_open(), key="pt_auto")
    if top[0].button("🔄 更新報價", use_container_width=True):
        cached_quotes.clear()

    def summary():
        qs, at = cached_quotes(tuple(want)) if want else ({}, now_tw().strftime("%H:%M:%S"))
        P = positions(data, qs)
        cap = data.get("capital", 1000000)
        total = P["cash"] + P["mv"]
        c1, c2 = st.columns(2)
        c1.metric("帳戶總值", money(total), f"{total - cap:+,.0f}（{(total / cap - 1) * 100:+.2f}%）", delta_color="inverse")
        c2.metric("現金", money(P["cash"]))
        c3, c4, c5 = st.columns(3)
        c3.metric("持股市值", money(P["mv"]))
        c4.metric("未實現", signed(P["unreal"]))
        c5.metric("已實現", signed(P["realized"]))
        st.caption(f"報價時間 {at}・{'盤中' if market_open() else '非交易時間（顯示最近成交價）'}・"
                   "紅＝賺、綠＝虧（台股習慣）・市值已扣賣出手續費與稅")
        if not P["open"]:
            st.info("還沒有持股。到「買賣」分頁選股票買進。")
            return
        rows = []
        for p in sorted(P["open"], key=lambda x: -(x["mv"] or 0)):
            flag = ""
            if p["last"] is not None and p["stop"] and p["last"] <= p["stop"]:
                flag = "🟩 已破停損"
            elif p["last"] is not None and p["target"] and p["last"] >= p["target"]:
                flag = "🟥 已到停利"
            elif p["last"] is not None and p["stop"] and p["last"] <= p["stop"] * 1.03:
                flag = "⚠️ 接近停損"
            day = (p["last"] / p["prev"] - 1) * 100 if p.get("prev") and p["last"] else None
            rows.append({"股票": f"{p['code']} {p['name']}", "股數": qty(p["shares"]), "均價": round(p["avg"], 2),
                         "現價": p["last"], "今日%": None if day is None else round(day, 2),
                         "損益": None if p["pnl"] is None else round(p["pnl"]),
                         "報酬%": None if p["ret"] is None else round(p["ret"], 2),
                         "停損": p["stop"], "停利": p["target"], "提醒": flag,
                         "依據": "照系統" if p["follow"] == "system" else "自己判斷"})
        df = pd.DataFrame(rows)

        def color(v):
            if not isinstance(v, (int, float)) or pd.isna(v) or v == 0:
                return ""
            return "color:#d23a2c" if v > 0 else "color:#1b8a4f"

        st.dataframe(df.style.map(color, subset=["今日%", "損益", "報酬%"]).format(
            {"均價": "{:,.2f}", "現價": "{:,.2f}", "今日%": "{:+.2f}", "損益": "{:+,.0f}", "報酬%": "{:+.2f}",
             "停損": "{:,.2f}", "停利": "{:,.2f}"}, na_rep="—"), hide_index=True, use_container_width=True)

    if auto and hasattr(st, "fragment"):
        st.fragment(run_every="30s")(summary)()
    else:
        summary()

    tab_trade, tab_log, tab_cmp = st.tabs(["買賣", "交易紀錄", "對照系統"])
    with tab_trade:
        trade_form(data, U, sig)
    with tab_log:
        log_view(data)
    with tab_cmp:
        compare_view(data, sig)

    with st.expander("⚙️ 存檔設定與手續費"):
        settings(data, stor)


def trade_form(data, U, sig):
    side = st.radio("要做什麼", ["買進", "賣出"], horizontal=True, key="pt_side")
    disc = data.get("disc", 1.0)
    latest = sig[sig["訊號日"] == sig["訊號日"].max()] if len(sig) else sig
    if side == "買進":
        if len(latest):
            st.caption(f"系統 {latest['訊號日'].iloc[0]} 收盤後買訊（隔天開盤買）："
                       + "、".join(f"{r['代號']} {r['名稱']}" for _, r in latest.iterrows()))
        opts = [f"{c} {u['name']}" for c, u in U.items()]
        pre = [f"{r['代號']} {r['名稱']}" for _, r in latest.iterrows() if r["代號"] in U]
        order = pre + [o for o in opts if o not in pre]
        pick = st.selectbox("股票（可直接打代號或名稱搜尋）", order, index=None, placeholder="例：2330 或 台積電", key="pt_pick")
        if not pick:
            return
        code = pick.split(" ")[0]
        name = U[code]["name"]
        s_row = latest[latest["代號"] == code]
        qs, at = cached_quotes((code,))
        q = qs.get(code)
        st.write(f"**{code} {name}** 現價 **{price_txt(q['price']) if q else '查不到'}**"
                 + (f"（{q['src']} {q['time']}）" if q else ""))
        c1, c2 = st.columns(2)
        price = c1.number_input("成交價", min_value=0.0, value=float(q["price"]) if q else 0.0, step=0.05,
                                format="%.2f", key=f"pt_bp_{code}")
        amt = c2.number_input("投入金額", min_value=0, value=200000, step=10000, key=f"pt_amt_{code}")
        shares = st.number_input("股數（1 張＝1000 股）", min_value=0, step=1,
                                 value=int(amt // price) if price > 0 else 0, key=f"pt_bs_{code}_{amt}_{price}")
        d_stop = float(s_row["計畫停損"].iloc[0]) if len(s_row) else 0.0
        d_tgt = float(s_row["計畫停利3%"].iloc[0]) if len(s_row) else 0.0
        c3, c4 = st.columns(2)
        stop = c3.number_input("停損價（可不填）", min_value=0.0, value=d_stop, step=0.05, format="%.2f", key=f"pt_st_{code}")
        tgt = c4.number_input("停利價（可不填）", min_value=0.0, value=d_tgt, step=0.05, format="%.2f", key=f"pt_tg_{code}")
        follow = st.radio("這筆的依據", ["照系統", "自己判斷"], index=0 if len(s_row) else 1, horizontal=True,
                          key=f"pt_fo_{code}")
        why = st.text_input("理由／當下想法", key=f"pt_why_{code}",
                            placeholder="例：系統回檔布局到價；或：怕大盤再跌，先買一半")
        if price > 0 and shares > 0:
            P = positions(data, {})
            cost = buy_cost(price, shares, disc)
            st.caption(f"{qty(shares)}・付出 {money(cost)}（含手續費 {money(cost - price * shares)}）・"
                       f"買完現金剩 {money(P['cash'] - cost)}")
        if st.button("✅ 存下這筆買進", type="primary", disabled=not (price > 0 and shares > 0), key="pt_buy_go"):
            t = {"id": uuid.uuid4().hex[:20], "code": code, "name": name, "side": "buy",
                 "date": now_tw().strftime("%Y-%m-%d"), "price": round(price, 2), "shares": int(shares),
                 "stop": stop or None, "target": tgt or None, "follow": "system" if follow == "照系統" else "own",
                 "why": why.strip(), "createdAt": now_tw().isoformat(timespec="seconds")}
            new = {**data, "trades": data["trades"] + [t]}
            if commit(new, f"模擬買進 {code} {qty(shares)} @ {price:g}"):
                st.success(f"已存下：買進 {code} {name} {qty(shares)}，每股 {price:g}")
                st.rerun()
    else:
        P = positions(data, {})
        if not P["open"]:
            st.info("目前沒有持股可以賣。")
            return
        labels = {f"{p['code']} {p['name']}（{qty(p['shares'])}）": p for p in P["open"]}
        pick = st.selectbox("要賣哪一檔", list(labels), key="pt_sell_pick")
        p = labels[pick]
        qs, at = cached_quotes((p["code"],))
        q = qs.get(p["code"])
        st.write(f"現價 **{price_txt(q['price']) if q else '查不到'}**" + (f"（{q['src']} {q['time']}）" if q else "")
                 + f"・均價 {price_txt(round(p['avg'], 2))}")
        c1, c2 = st.columns(2)
        price = c1.number_input("成交價", min_value=0.0, value=float(q["price"]) if q else 0.0, step=0.05,
                                format="%.2f", key=f"pt_sp_{p['code']}")
        shares = c2.number_input("股數", min_value=0, max_value=int(p["shares"]), value=int(p["shares"]), step=1,
                                 key=f"pt_ss_{p['code']}")
        why = st.text_input("為什麼賣", key=f"pt_swhy_{p['code']}", placeholder="例：到停利；破停損；怕跌先跑")
        if price > 0 and shares > 0:
            net = sell_net(p["code"], price, shares, disc)
            pnl = net - p["avg"] * shares
            st.caption(f"實收 {money(net)}（扣手續費與{'ETF 0.1%' if is_etf(p['code']) else '0.3%'}稅）・"
                       f"這次損益 {pnl:+,.0f}（{pnl / (p['avg'] * shares) * 100:+.2f}%）")
        if st.button("✅ 存下這筆賣出", type="primary", disabled=not (price > 0 and shares > 0), key="pt_sell_go"):
            t = {"id": uuid.uuid4().hex[:20], "code": p["code"], "name": p["name"], "side": "sell",
                 "date": now_tw().strftime("%Y-%m-%d"), "price": round(price, 2), "shares": int(shares),
                 "follow": p["follow"], "why": why.strip(), "createdAt": now_tw().isoformat(timespec="seconds")}
            new = {**data, "trades": data["trades"] + [t]}
            if commit(new, f"模擬賣出 {p['code']} {qty(shares)} @ {price:g}"):
                st.success(f"已存下：賣出 {p['code']} {p['name']} {qty(shares)}，每股 {price:g}")
                st.rerun()


def log_view(data):
    tr = sorted(data["trades"], key=lambda x: (x["date"], x.get("createdAt", "")), reverse=True)
    if not tr:
        st.info("還沒有交易紀錄。")
        return
    df = pd.DataFrame([{"日期": t["date"], "買賣": "買" if t["side"] == "buy" else "賣",
                        "股票": f"{t['code']} {t.get('name', '')}", "價格": t["price"], "股數": qty(t["shares"]),
                        "金額": round(t["price"] * t["shares"]),
                        "依據": "照系統" if t.get("follow") == "system" else "自己判斷", "理由": t.get("why", "")}
                       for t in tr])
    st.dataframe(df, hide_index=True, use_container_width=True)
    with st.expander("刪除記錯的一筆"):
        labels = {f"{t['date']} {'買' if t['side'] == 'buy' else '賣'} {t['code']} {t.get('name', '')} "
                  f"{qty(t['shares'])} @ {t['price']:g}": t["id"] for t in tr}
        pick = st.selectbox("選一筆", list(labels), index=None, key="pt_del_pick")
        ok = st.checkbox("我確定要刪除這筆", key="pt_del_ok")
        if st.button("刪除", disabled=not (pick and ok), key="pt_del_go"):
            new = {**data, "trades": [t for t in data["trades"] if t["id"] != labels[pick]]}
            if commit(new, f"刪除模擬交易 {pick}"):
                st.success("已刪除")
                st.rerun()


def compare_view(data, sig):
    qs_codes = tuple(sorted({t["code"] for t in data["trades"]} | (set(sig["代號"]) if len(sig) else set())))
    qs, _ = cached_quotes(qs_codes) if qs_codes else ({}, "")
    P = positions(data, qs)
    st.markdown("**照系統 vs 自己判斷**")
    cols = st.columns(2)
    for col, key, title in ((cols[0], "system", "照系統"), (cols[1], "own", "自己判斷")):
        c = [x for x in P["closed"] if x["follow"] == key]
        o = [x for x in P["open"] if x["follow"] == key and x["pnl"] is not None]
        wins = sum(x["pnl"] > 0 for x in c)
        col.metric(title, f"已實現 {sum(x['pnl'] for x in c):+,.0f}",
                   f"帳面 {sum(x['pnl'] for x in o):+,.0f}", delta_color="inverse")
        col.caption(f"已出場 {len(c)} 筆（賺 {wins} 筆）・持有中 {len(o)} 筆")
    st.markdown("**系統回檔布局買訊，你有沒有做？**")
    if not len(sig):
        st.info("最近沒有回檔布局買訊。")
        return
    buys = [t for t in data["trades"] if t["side"] == "buy"]
    rows = []
    for _, r in sig.sort_values(["訊號日", "代號"], ascending=[False, True]).iterrows():
        d0 = r["訊號日"]
        d1 = (dt.date.fromisoformat(d0) + dt.timedelta(days=7)).isoformat()
        did = any(t["code"] == r["代號"] and d0 < t["date"] <= d1 for t in buys)  # 訊號隔天起 7 天內買進才算
        q = qs.get(r["代號"])
        chg = (q["price"] / r["訊號收盤"] - 1) * 100 if q and r["訊號收盤"] else None
        rows.append({"訊號日": d0, "股票": f"{r['代號']} {r['名稱']}", "訊號價": r["訊號收盤"],
                     "現價": q["price"] if q else None, "之後%": None if chg is None else round(chg, 2),
                     "你": "✅ 有做" if did else "— 沒做"})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.map(lambda v: "" if not isinstance(v, (int, float)) or pd.isna(v) or v == 0
                              else ("color:#d23a2c" if v > 0 else "color:#1b8a4f"), subset=["之後%"])
                 .format({"訊號價": "{:,.2f}", "現價": "{:,.2f}", "之後%": "{:+.2f}"}, na_rep="—"),
                 hide_index=True, use_container_width=True)
    st.caption("「之後%」＝訊號當天收盤到現在的漲跌，用來看沒照做的那幾檔錯過或躲過多少。"
               "回檔布局規則：每筆約 20 萬、停損 −20%、淨賺 10% 賣。研究用途，不構成投資建議。")


def settings(data, stor):
    st.write("**存檔狀態：** " + ("✅ 已設定金鑰，可以存檔" if stor.can_write else "❌ 還沒設定金鑰"))
    st.caption(f"存在 GitHub {stor.repo} 的 {stor.branch} 分支 paper/trades.json（公開倉庫，別人看得到這份虛擬紀錄）。")
    disc = st.number_input("手續費折扣（1＝不打折，6 折填 0.6）", min_value=0.1, max_value=1.0,
                           value=float(data.get("disc", 1.0)), step=0.05, key="pt_disc")
    if disc != data.get("disc", 1.0) and st.button("儲存折扣", key="pt_disc_go"):
        if commit({**data, "disc": float(disc)}, f"模擬交易：手續費折扣 {disc:g}"):
            st.rerun()
    if st.button("重新讀取存檔", key="pt_reload"):
        load(force=True)
        st.rerun()
