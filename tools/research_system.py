"""
完整研究（2026-10-21 第四輪）：依使用者「需求總結 v1.0」——
  A 突破追漲、B 回檔提前布局（到底跌多少可以買）、C 加碼攤平、彈性停利 3／5／10%、
  持有時間與停損由數據決定、整體 100 萬帳戶的最大回撤與年報酬。
只做統計，不改規則。輸出 research/system.txt。

方法
- 股票池：產業清單＋台灣 50＋證交所成交值前 300 普通股＋全部 ETF 清單；Yahoo 還原權息價；約 10 年。
- B 回檔：收盤距「近 60 日最高點」第一次跌超過 D%（D＝5,10,15,20,25,30,40）的那天，隔天開盤買。
  四種條件：無／年線之上（長期多頭）／超賣（RSI14<30 或 K<20）／止跌紅K（跌到位後 5 天內第一根收紅且過昨高）。
- A 突破：RS 前 20% 突破 20 日高、O'Neil 底部突破、Minervini VCP、跳空起漲缺口（定義同第二輪）。
- 對照：任一天買。
- C 加碼（報酬一律以「整筆預算」計算，沒用到的錢當閒置現金，才能公平比較）：
    C0 一次買（預算 1 份）／C1 先 1 份、跌 10% 再 1 份（預算 2）／
    C2 先 1、跌 10% 加 1、跌 20% 加 1（預算 3）／C3 先 1、跌 15% 加 2 份（預算 3）
  停利以「平均成本」算淨報酬 3／5／10%。
  停損：C0、C1 為第一筆價格 −20%；C2、C3 為 −30%（因為 −20% 還要加碼）；或不停損。
  持有上限：15／60／250 個交易日，到期以收盤賣。
- 驗證：2016～2022 年「挑」（樣本內），2023 年以後「驗」（樣本外）；只用樣本內挑出的組合看樣本外。
- 帳戶模擬：100 萬、每筆預算 10 萬（含加碼備用金，進場時先保留）、現金不足就不買；逐日計算淨值，
  得出年化報酬、最大回撤，與「0050 買進放著」比較。
- 成本：手續費 0.1425%×2＋證交稅（股票 0.3%、ETF 0.1%）＋滑價每邊 0.1%。
- 偏差：股票池為今天仍存在的股票（存活者偏差），回檔買進類會被高估；2016～2026 為台股大多頭。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402
from sectors import SECTORS, stocks_of  # noqa: E402

OUT = Path(os.environ.get("SYS_OUT", ROOT / "research"))
OUT.mkdir(exist_ok=True)
SPLIT = pd.Timestamp("2023-01-01")
SLIP = 0.001
FAKE = bool(os.environ.get("SYS_FAKE"))
t0 = time.time()

try:
    from numba import njit
except Exception:  # noqa: BLE001
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

EXTRA = ["2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", "2303", "2412", "2886", "2884",
         "1301", "1303", "2002", "2885", "2892", "2880", "5880", "2883", "2887", "3711", "2357", "2395",
         "3045", "4904", "2912", "1216", "2207", "2603", "2609", "2615", "1101", "1326", "6505", "2327",
         "3008", "2379", "3034", "2301", "5871", "2890", "9910", "2409", "3481", "2344", "2408", "2618",
         "3653", "0050"]


# ------------------------------------------------------------------ 資料
def universe():
    u = {}
    for sec in SECTORS:
        for s in stocks_of(sec):
            u.setdefault(s["code"], s["market"])
    for c in EXTRA:
        u.setdefault(c, "TW")
    if FAKE:
        return u
    try:
        import requests
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30)
        df = pd.DataFrame(r.json())
        df = df[df["Code"].str.fullmatch(r"[1-9]\d{3}")]
        df["tv"] = pd.to_numeric(df["TradeValue"], errors="coerce")
        for c in df.nlargest(300, "tv")["Code"]:
            u.setdefault(c, "TW")
    except Exception as ex:  # noqa: BLE001
        print("證交所清單失敗：", ex)
    return u


def fetch(u):
    import yfinance as yf

    def grab(tks):
        got = {}
        for k in range(0, len(tks), 80):
            part = tks[k:k + 80]
            try:
                d = yf.download(part, period="11y", interval="1d", group_by="ticker", auto_adjust=True,
                                actions=False, threads=True, progress=False)
            except Exception:  # noqa: BLE001
                continue
            for tk in part:
                try:
                    sub = d[tk] if isinstance(d.columns, pd.MultiIndex) else d
                    sub = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                    sub = sub[(sub["volume"] > 0) & (sub["high"] >= sub["low"]) & (sub["low"] > 0)]
                    if len(sub) >= 300:
                        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                        got[tk.split(".")[0]] = sub.astype(float)
                except Exception:  # noqa: BLE001
                    continue
        return got

    out = grab([f"{c}.{m}" for c, m in u.items()])
    miss = [c for c in u if c not in out]
    out.update(grab([f"{c}.{'TWO' if u[c] == 'TW' else 'TW'}" for c in miss]))
    return out


U = universe()
if FAKE:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-10-01", "2026-10-20")
    data = {}
    for code in list(U)[:int(os.environ.get("SYS_N", 24))] + ["0050", "3653"]:
        c = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.022, len(idx))))
        o = c * (1 + rng.normal(0, 0.007, len(idx)))
        data[code] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, .007, len(idx)))),
                                   "low": np.minimum(o, c) * (1 - abs(rng.normal(0, .007, len(idx)))),
                                   "close": c, "volume": rng.integers(500, 9000, len(idx)).astype(float)}, index=idx)
else:
    data = fetch(U)
n_etf = sum(E.is_etf(k) for k in data)
print(f"有資料 {len(data)} 檔（個股 {len(data) - n_etf}、ETF {n_etf}），{time.time() - t0:.0f} 秒")


# ------------------------------------------------------------------ 指標與訊號
def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prep(df):
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    for n in (20, 50, 150, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
    d["hh60"] = h.rolling(60).max()
    d["dd60"] = c / d["hh60"] - 1
    d["rsi14"] = rsi(c, 14)
    lo9, hi9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = ((c - lo9) / (hi9 - lo9).replace(0, np.nan) * 100).fillna(50).to_numpy()
    k = np.empty(len(d)); k[0] = 50
    for i in range(1, len(d)):
        k[i] = k[i - 1] * 2 / 3 + rsv[i] / 3
    d["k"] = k
    d["chg"] = c.pct_change() * 100
    d["red"] = (c > o) & (d["chg"] > 0)
    d["red_turn"] = d["red"] & (c > h.shift(1))
    d["vol50"] = v.shift(1).rolling(50).mean()
    d["vr50"] = v / d["vol50"]
    d["hh252"] = h.shift(1).rolling(252).max()
    d["ll252"] = l.shift(1).rolling(252).min()
    d["hh20"] = h.shift(1).rolling(20).max()
    d["hh10"] = h.shift(1).rolling(10).max()
    d["hh30"] = h.shift(1).rolling(30).max()
    d["ll30"] = l.shift(1).rolling(30).min()
    d["rng10"] = (h.rolling(10).max() - l.rolling(10).min()) / c
    d["rng60"] = (h.rolling(60).max() - l.rolling(60).min()) / c
    d["ibs"] = (c - l) / (h - l).replace(0, np.nan)
    d["r60"] = c.pct_change(60)
    return d


P = {k: prep(v) for k, v in data.items()}
r60 = pd.DataFrame({k: v["r60"] for k, v in P.items() if not E.is_etf(k)})
RS = r60.rank(axis=1, pct=True)

DS = [5, 10, 15, 20, 25, 30, 40]
FILT = ["無條件", "年線之上", "超賣(RSI<30或K<20)", "止跌紅K"]


def signals(code, d):
    c, h, o = d["close"], d["high"], d["open"]
    s = {"對照：任一天買": np.ones(len(d), bool)}
    dd = d["dd60"].to_numpy()
    prev = np.r_[0.0, dd[:-1]]
    up200 = (c > d["ma200"]).to_numpy()
    over = ((d["rsi14"] < 30) | (d["k"] < 20)).to_numpy()
    rt = d["red_turn"].fillna(False).to_numpy()
    for D in DS:
        cross = (dd <= -D / 100) & (prev > -D / 100)
        s[f"B 跌{D}%｜無條件"] = cross
        s[f"B 跌{D}%｜年線之上"] = cross & up200
        # 超賣：跌到位後 5 天內出現超賣的第一天
        zone = pd.Series(cross).rolling(5, min_periods=1).max().to_numpy() > 0
        deep = dd <= -D / 100 * 0.9
        s[f"B 跌{D}%｜超賣(RSI<30或K<20)"] = zone & deep & over
        s[f"B 跌{D}%｜止跌紅K"] = zone & deep & rt
    rs = RS[code].reindex(d.index) if code in RS else pd.Series(np.nan, index=d.index)
    tmpl = ((c > d["ma50"]) & (d["ma50"] > d["ma150"]) & (d["ma150"] > d["ma200"])
            & (d["ma200"] > d["ma200"].shift(20)) & (c >= 0.75 * d["hh252"]) & (c >= 1.3 * d["ll252"]))
    red = d["red"]
    s["A RS前20%突破20日高"] = ((rs >= 0.8) & (c > d["hh20"]) & (d["vr50"] >= 1.5) & red).fillna(False).to_numpy()
    depth = (d["hh30"] - d["ll30"]) / d["hh30"]
    s["A O'Neil底部突破"] = (depth.between(0.08, 0.25) & (c > d["hh30"]) & (d["vr50"] >= 1.4) & (c > d["ma50"])
                         & (d["ibs"] > 0.75)).fillna(False).to_numpy()
    s["A Minervini VCP"] = (tmpl & ((d["rng10"] / d["rng60"]).shift(1) < 0.4) & (c > d["hh10"])
                            & (d["vr50"] >= 1.5) & red).fillna(False).to_numpy()
    s["A 跳空起漲缺口"] = ((o > h.shift(1) * 1.01) & (c >= o) & (d["vr50"] >= 2) & (c > d["ma20"])).fillna(False).to_numpy()
    for k in s:
        s[k] = np.array(s[k], dtype=np.bool_)
        s[k][:260] = False
    return s


SCHEMES = {  # 名稱: (加碼跌幅list, 加碼份數list, 預算份數, 停損%)
    "C0 一次買": ([], [], 1.0, 20.0),
    "C1 跌10%加1份": ([10.0], [1.0], 2.0, 20.0),
    "C2 跌10%、20%各加1份": ([10.0, 20.0], [1.0, 1.0], 3.0, 30.0),
    "C3 跌15%加2份": ([15.0], [2.0], 3.0, 30.0),
}
TGTS = [3.0, 5.0, 10.0]
HOLDS = [15, 60, 250]


@njit(cache=False)
def sim(O, H, L, C, sig, add_lv, add_u, budget, tgt, stop, hold, bc, sc):
    n = len(O)
    m = 0
    res = np.zeros((n, 8))  # i, xj, ret_budget, code, mae, days, n_add, be_need
    i = 0
    while i < n - 2:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        p1 = O[e]
        units = 1.0
        cost = p1 * (1 + bc)
        filled = np.zeros(len(add_lv))
        last = e + hold - 1
        if last > n - 1:
            break
        code = 3.0
        xj = last
        x = C[last]
        lo = p1
        for j in range(e, last + 1):
            if L[j] < lo:
                lo = L[j]
            added_today = False
            for k in range(len(add_lv)):
                if filled[k] == 0:
                    lv = p1 * (1 - add_lv[k] / 100)
                    if L[j] <= lv:
                        fp = lv
                        if j > e and O[j] < lv:
                            fp = O[j]
                        cost += fp * (1 + bc) * add_u[k]
                        units += add_u[k]
                        filled[k] = 1
                        added_today = True
            if stop > 0:
                sl = p1 * (1 - stop / 100)
                if L[j] <= sl:
                    code = 2.0
                    xj = j
                    x = sl
                    if j > e and O[j] < sl:
                        x = O[j]
                    break
            avg = cost / units
            tp = avg * (1 + tgt / 100) / (1 - sc)
            if added_today:
                if j > e and O[j] >= tp:
                    code = 1.0
                    xj = j
                    x = O[j]
                    break
            elif H[j] >= tp:
                code = 1.0
                xj = j
                x = tp
                if j > e and O[j] > tp:
                    x = O[j]
                break
        ret = (x * (1 - sc) * units - cost) / (p1 * (1 + bc) * budget) * 100
        nadd = filled.sum()
        res[m, 0] = i
        res[m, 1] = xj
        res[m, 2] = ret
        res[m, 3] = code
        res[m, 4] = (lo / p1 - 1) * 100
        res[m, 5] = xj - e + 1
        res[m, 6] = nadd
        res[m, 7] = ((cost / units) / (1 - sc) / lo - 1) * 100
        m += 1
        i = xj + 1
    return res[:m]


# ------------------------------------------------------------------ 主迴圈：累積每個組合的結果
ACC = {}  # (類型, 進場, 攤平, 停利, 停損, 持有) -> list of arrays [year, ret, code, mae, days, nadd, be]
t1 = time.time()
SIGS = {}
for code, d in P.items():
    typ = "ETF" if E.is_etf(code) else "個股"
    sg = signals(code, d)
    SIGS[code] = sg
    O, H, L, C = (d[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    years = d.index.year.to_numpy()
    bc0, sc0 = E.cost_rates(code, E.DEFAULT_PARAMS)
    bc, sc = bc0 + SLIP, sc0 + SLIP
    for sname, sig in sg.items():
        if not sig.any():
            continue
        for cname, (lv, uu, bud, stp) in SCHEMES.items():
            lv_a, uu_a = np.array(lv, dtype=np.float64), np.array(uu, dtype=np.float64)
            for stop in (stp, 0.0):
                for tgt in TGTS:
                    for hold in HOLDS:
                        r = sim(O, H, L, C, sig, lv_a, uu_a, bud, tgt, stop, hold, bc, sc)
                        if len(r) == 0:
                            continue
                        yr = years[r[:, 0].astype(np.int64)]
                        arr = np.column_stack([yr, r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6], r[:, 7]]).astype(np.float32)
                        ACC.setdefault((typ, sname, cname, tgt, stop > 0, hold), []).append(arr)
print(f"模擬完成 {len(ACC)} 組，{time.time() - t1:.0f} 秒")


def stats(a):
    if a is None or len(a) == 0:
        return None
    yr, ret, code, mae, days, nadd, be = (a[:, k] for k in range(7))
    return {"n": len(a), "win": (code == 1).mean() * 100, "pos": (ret > 0).mean() * 100, "avg": float(ret.mean()),
            "worst": float(ret.min()), "stop": (code == 2).mean() * 100, "days": float(days.mean()),
            "days_med": float(np.median(days)), "mae50": float(np.median(mae)), "mae10": float(np.percentile(mae, 10)),
            "add": (nadd > 0).mean() * 100, "be": float(np.median(be[nadd > 0])) if (nadd > 0).any() else 0.0}


S = {}
for k, lst in ACC.items():
    a = np.concatenate(lst)
    S[k] = {"all": stats(a), "IS": stats(a[a[:, 0] < 2023]), "OOS": stats(a[a[:, 0] >= 2023]),
            "y22": stats(a[a[:, 0] == 2022]),
            "yr": {int(y): float(a[a[:, 0] == y, 1].mean()) for y in np.unique(a[:, 0])}}
del ACC


def f1(s):
    if s is None:
        return "0 筆"
    return f"{s['n']} 筆・成功率 {s['win']:.0f}%・平均 {s['avg']:+.2f}%"


def f2(s):
    if s is None:
        return "0 筆"
    t = (f"{s['n']} 筆・成功率 {s['win']:.0f}%（淨賺 {s['pos']:.0f}%）・每筆平均 {s['avg']:+.2f}%・停損 {s['stop']:.0f}%"
         f"・抱 {s['days_med']:.0f} 天(中位)・過程最大浮虧 中位 {s['mae50']:.0f}% 最糟一成 {s['mae10']:.0f}%・最差一筆 {s['worst']:+.0f}%")
    if s["add"] > 0:
        t += f"・有加碼 {s['add']:.0f}%（加碼後需從最低點回升 {s['be']:.0f}% 才回本）"
    return t


def name(k):
    typ, sname, cname, tgt, stop, hold = k
    stp = SCHEMES[cname][3]
    return f"{sname}｜{cname}｜停利{tgt:.0f}%｜{'停損' + str(int(stp)) + '%' if stop else '不停損'}｜最多{hold}天"


L_ = [f"研究日期：{pd.Timestamp.now(tz='Asia/Taipei'):%Y-%m-%d %H:%M}・{len(data)} 檔（個股 {len(data) - n_etf}、ETF {n_etf}）・還原權息・約 10 年・共 {len(S)} 種組合",
      "報酬以「整筆預算」計（含加碼備用金），已扣手續費、證交稅、滑價。成功率＝先碰到停利。",
      "⚠ 存活者偏差＋10 年大多頭，結果偏樂觀；請看樣本外與 2022 空頭年。", ""]

# 一、跌幅地圖
for typ in ("個股", "ETF"):
    for (tgt, hold) in ((5.0, 60), (10.0, 250)):
        L_.append(f"==================== 一、{typ}：到底跌多少可以買？（一次買、停利{tgt:.0f}%、停損20%、最多{hold}天）====================")
        bk = (typ, "對照：任一天買", "C0 一次買", tgt, True, hold)
        if bk in S:
            L_.append(f"  對照任一天買：{f1(S[bk]['all'])}・過程浮虧中位 {S[bk]['all']['mae50']:.0f}%")
        for D in DS:
            parts = []
            for fl in FILT:
                k = (typ, f"B 跌{D}%｜{fl}", "C0 一次買", tgt, True, hold)
                if k in S and S[k]["all"] and S[k]["all"]["n"] >= 20:
                    s_ = S[k]["all"]
                    parts.append(f"{fl} {s_['n']}筆 成功{s_['win']:.0f}% 均{s_['avg']:+.1f}%")
            if parts:
                L_.append(f"  跌{D:>2}%：" + "｜".join(parts))
        L_.append("")

# 二、樣本內挑、樣本外驗（依持有時間、停損與否分組）
for typ in ("個股", "ETF"):
    L_.append(f"==================== 二、{typ}：用 2016～2022 挑最好的，2023 以後驗證 ====================")
    for hold in HOLDS:
        for stop in (True, False):
            cand = [k for k in S if k[0] == typ and k[5] == hold and k[4] == stop and not k[1].startswith("對照")
                    and S[k]["IS"] and S[k]["OOS"] and S[k]["IS"]["n"] >= 40 and S[k]["OOS"]["n"] >= 20]
            win70 = [k for k in cand if S[k]["IS"]["win"] >= 70 and S[k]["IS"]["avg"] > 0]
            pick = sorted(win70, key=lambda k: -S[k]["IS"]["avg"])[:4]
            L_.append(f"【最多抱 {hold} 天・{'有停損' if stop else '不停損'}】樣本內成功率≥70% 且賺錢：{len(win70)} 組，依樣本內平均取前 4：")
            for k in pick:
                s = S[k]
                L_.append(f"  {name(k)}")
                L_.append(f"    樣本內 {f1(s['IS'])}｜樣本外 {f2(s['OOS'])}")
                bk = (typ, "對照：任一天買", k[2], k[3], k[4], k[5])
                y22 = s["y22"]
                L_.append(f"    對照任一天買（同出場）樣本外：{f1(S[bk]['OOS']) if bk in S else '無'}｜2022 空頭年：{f1(y22) if y22 else '無'}"
                          f"｜{len(s['yr'])} 年中 {sum(v > 0 for v in s['yr'].values())} 年為正")
            if not pick:
                L_.append("  （無）")
        L_.append("")

# 三、攤平方式比較（用跌 20%｜無條件、停利 5%、最多 60 天）
L_.append("==================== 三、攤平加碼有沒有幫助？（個股・B 跌20%｜無條件・停利5%・最多60天，有停損）====================")
for cname in SCHEMES:
    k = ("個股", "B 跌20%｜無條件", cname, 5.0, True, 60)
    if k in S:
        L_.append(f"  {cname}：{f2(S[k]['all'])}")
L_.append("（每筆平均以整筆預算計；C2、C3 預算是 3 份，C1 是 2 份）")
L_.append("")

# 四、突破 vs 回檔（用相同出場）
L_.append("==================== 四、突破追漲 vs 回檔布局（個股・一次買・停利10%・停損20%・最多250天）====================")
for sname in ["對照：任一天買", "A RS前20%突破20日高", "A O'Neil底部突破", "A Minervini VCP", "A 跳空起漲缺口",
              "B 跌10%｜無條件", "B 跌20%｜無條件", "B 跌30%｜無條件", "B 跌20%｜年線之上", "B 跌20%｜止跌紅K"]:
    k = ("個股", sname, "C0 一次買", 10.0, True, 250)
    if k in S:
        L_.append(f"  {sname}：{f2(S[k]['all'])}｜樣本外 {f1(S[k]['OOS'])}")
L_.append("")


# ------------------------------------------------------------------ 五、100 萬帳戶模擬
def portfolio(cfg, typ, capital=1_000_000, per=100_000):
    _, sname, cname, tgt, stop, hold = cfg
    lv, uu, bud, stp = SCHEMES[cname]
    stp = stp if stop else 0.0
    codes = [c for c in P if (E.is_etf(c) == (typ == "ETF"))]
    cal = sorted(set().union(*[set(P[c].index) for c in codes]))
    cal = [d for d in cal if d >= pd.Timestamp("2016-06-01")]
    pos_ix = {c: {d: i for i, d in enumerate(P[c].index)} for c in codes}
    arr = {c: tuple(P[c][k].to_numpy() for k in ("open", "high", "low", "close")) for c in codes}
    cost_r = {c: E.cost_rates(c, E.DEFAULT_PARAMS) for c in codes}
    sig_by_day = {}
    for c in codes:
        sg = SIGS[c][sname]
        idx = P[c].index
        for i in np.nonzero(sg)[0]:
            if i + 1 < len(idx):
                sig_by_day.setdefault(idx[i + 1], []).append(c)  # 隔天進場
    rng_ = np.random.default_rng(0)
    cash, open_ = float(capital), {}
    eq, n_tr, wins, rets = [], 0, 0, []
    for day in cal:
        # 先處理持倉
        for c in list(open_):
            p = open_[c]
            j = pos_ix[c].get(day)
            if j is None:
                continue
            O, H, L, C = arr[c]
            bc, sc = cost_r[c][0] + SLIP, cost_r[c][1] + SLIP
            p["last"] = C[j]
            added = False
            for kk in range(len(lv)):
                if not p["filled"][kk]:
                    level = p["p1"] * (1 - lv[kk] / 100)
                    if L[j] <= level:
                        fp = min(O[j], level) if j > p["e"] else level
                        sh = p["unit_sh"] * uu[kk]
                        p["cost"] += fp * (1 + bc) * sh
                        p["sh"] += sh
                        p["reserve"] -= fp * (1 + bc) * sh
                        p["filled"][kk] = True
                        added = True
            exit_px = None
            if stp > 0 and L[j] <= p["p1"] * (1 - stp / 100):
                exit_px = min(O[j], p["p1"] * (1 - stp / 100)) if j > p["e"] else p["p1"] * (1 - stp / 100)
                res = "停損"
            else:
                tp = p["cost"] / p["sh"] * (1 + tgt / 100) / (1 - sc)
                if (not added and H[j] >= tp) or (added and j > p["e"] and O[j] >= tp):
                    exit_px = max(O[j], tp) if j > p["e"] else tp
                    res = "停利"
                elif j - p["e"] + 1 >= hold:
                    exit_px, res = C[j], "到期"
            if exit_px is not None:
                proceeds = exit_px * (1 - sc) * p["sh"]
                cash += proceeds + p["reserve"]
                r = (proceeds - p["cost"]) / per * 100
                rets.append(r)
                n_tr += 1
                wins += res == "停利"
                del open_[c]
        # 新進場（開盤）
        cands = [c for c in sig_by_day.get(day, []) if c not in open_]
        rng_.shuffle(cands)
        for c in cands:
            if cash < per:
                break
            j = pos_ix[c][day]
            O = arr[c][0]
            bc = cost_r[c][0] + SLIP
            unit_val = per / bud
            unit_sh = unit_val / (O[j] * (1 + bc))
            cost = unit_sh * O[j] * (1 + bc)
            cash -= per
            open_[c] = {"e": j, "p1": O[j], "sh": unit_sh, "unit_sh": unit_sh, "cost": cost,
                        "reserve": per - cost, "filled": [False] * len(lv), "last": O[j]}
        val = cash + sum(p["reserve"] + p["sh"] * p["last"] for p in open_.values())
        eq.append(val)
    eq = pd.Series(eq, index=cal)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    worst_year = eq.groupby(eq.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1).min()
    return cagr * 100, mdd * 100, n_tr, (wins / n_tr * 100 if n_tr else 0), eq, worst_year * 100


L_.append("==================== 五、100 萬帳戶模擬（每筆預算 10 萬，現金不足就不買）====================")
try:
    if "0050" in P:
        c50 = P["0050"]["close"]
        c50 = c50[c50.index >= pd.Timestamp("2016-06-01")]
        y = (c50.index[-1] - c50.index[0]).days / 365.25
        L_.append(f"  對照 0050 買進放著：年化 {((c50.iloc[-1] / c50.iloc[0]) ** (1 / y) - 1) * 100:+.1f}%・最大回撤 {(c50 / c50.cummax() - 1).min() * 100:.0f}%")
    chosen = []
    for typ in ("個股", "ETF"):
        for hold in HOLDS:
            cand = [k for k in S if k[0] == typ and k[5] == hold and not k[1].startswith("對照")
                    and S[k]["IS"] and S[k]["OOS"] and S[k]["IS"]["n"] >= 40 and S[k]["OOS"]["n"] >= 20
                    and S[k]["IS"]["win"] >= 70 and S[k]["IS"]["avg"] > 0]
            if cand:
                chosen.append((typ, max(cand, key=lambda k: S[k]["IS"]["avg"])))
            bk = (typ, "對照：任一天買", "C0 一次買", 10.0, True, hold)
            if bk in S:
                chosen.append((typ, bk))
    for typ, k in chosen:
        cagr, mdd, n_tr, wr, eq, wy = portfolio(k, typ)
        L_.append(f"  [{typ}] {name(k)}")
        L_.append(f"    年化 {cagr:+.1f}%・最大回撤 {mdd:.0f}%・最差年度 {wy:+.0f}%・交易 {n_tr} 筆・成功率 {wr:.0f}%")
except Exception:  # noqa: BLE001
    import traceback
    L_.append("帳戶模擬失敗：" + traceback.format_exc()[-1200:])

txt = "\n".join(L_)
(OUT / "system.txt").write_text(txt, encoding="utf-8")
print(txt)
print(f"總耗時 {time.time() - t0:.0f} 秒")
