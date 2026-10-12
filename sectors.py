"""
產業鏈股票清單（自訂分類，非證交所官方分類）。

新增產業：在 SECTORS 加一個 key，值是「子類 → 股票清單」。
每檔股票格式：("代碼", "名稱", "TW" 上市 / "TWO" 上櫃)。
市場別填錯也沒關係，抓不到時程式會自動改試另一個市場。

清單整理：2026-10，依公開資訊與市場常見歸類；個股業務比重會變，請定期檢查增刪。
"""

SECTORS = {
    "AI 產業": {
        "IC 設計／晶圓代工": [
            ("2330", "台積電", "TW"), ("2454", "聯發科", "TW"), ("3661", "世芯-KY", "TW"),
            ("3443", "創意", "TW"), ("3035", "智原", "TW"), ("5274", "信驊", "TWO"),
        ],
        "AI 伺服器代工": [
            ("2317", "鴻海", "TW"), ("2382", "廣達", "TW"), ("3231", "緯創", "TW"),
            ("6669", "緯穎", "TW"), ("2356", "英業達", "TW"), ("2376", "技嘉", "TW"),
            ("3706", "神達", "TW"),
        ],
        "散熱": [
            ("3017", "奇鋐", "TW"), ("3324", "雙鴻", "TWO"), ("2421", "建準", "TW"),
            ("3653", "健策", "TW"),
        ],
        "電源": [
            ("2308", "台達電", "TW"), ("2301", "光寶科", "TW"),
        ],
        "PCB／銅箔基板": [
            ("2383", "台光電", "TW"), ("6274", "台燿", "TWO"), ("3037", "欣興", "TW"),
            ("2368", "金像電", "TW"), ("8046", "南電", "TW"),
        ],
        "機殼／滑軌": [
            ("8210", "勤誠", "TW"), ("2059", "川湖", "TW"), ("6805", "富世達", "TW"),
        ],
        "網通／連接器／光通訊": [
            ("2345", "智邦", "TW"), ("3533", "嘉澤", "TW"), ("3665", "貿聯-KY", "TW"),
            ("4979", "華星光", "TWO"), ("3081", "聯亞", "TWO"), ("3363", "上詮", "TWO"),
            ("6442", "光聖", "TW"),
        ],
        "先進封裝／測試設備": [
            ("3711", "日月光投控", "TW"), ("6515", "穎崴", "TW"), ("6223", "旺矽", "TWO"),
            ("3131", "弘塑", "TWO"), ("3583", "辛耘", "TW"), ("6187", "萬潤", "TWO"),
        ],
    },
    "半導體": {
        "晶圓代工": [("2330", "台積電", "TW"), ("2303", "聯電", "TW")],
        "IC 設計": [
            ("2454", "聯發科", "TW"), ("3034", "聯詠", "TW"), ("2379", "瑞昱", "TW"),
            ("3443", "創意", "TW"), ("3661", "世芯-KY", "TW"), ("3035", "智原", "TW"),
            ("5274", "信驊", "TWO"),
        ],
        "封測": [("3711", "日月光投控", "TW")],
        "設備／測試介面": [
            ("6515", "穎崴", "TW"), ("6223", "旺矽", "TWO"), ("3131", "弘塑", "TWO"),
            ("3583", "辛耘", "TW"),
        ],
    },
    "PCB／載板": {
        "銅箔基板": [("2383", "台光電", "TW"), ("6274", "台燿", "TWO")],
        "PCB": [("2368", "金像電", "TW")],
        "IC 載板": [("3037", "欣興", "TW"), ("8046", "南電", "TW"), ("3189", "景碩", "TW")],
    },
    "散熱／電源": {
        "散熱": [
            ("3017", "奇鋐", "TW"), ("3324", "雙鴻", "TWO"), ("2421", "建準", "TW"),
            ("3653", "健策", "TW"),
        ],
        "電源": [("2308", "台達電", "TW"), ("2301", "光寶科", "TW")],
    },
    "ETF": {
        "市值型": [("0050", "元大台灣50", "TW"), ("006208", "富邦台50", "TW")],
        "高股息": [
            ("0056", "元大高股息", "TW"), ("00878", "國泰永續高股息", "TW"),
            ("00919", "群益台灣精選高息", "TW"), ("00929", "復華台灣科技優息", "TW"),
            ("00940", "元大台灣價值高息", "TW"),
        ],
    },
}


def stocks_of(sector: str):
    """回傳 [{"code","name","market","group"}, ...]，同一代碼只出現一次。"""
    out, seen = [], set()
    for group, items in SECTORS[sector].items():
        for code, name, mkt in items:
            if code in seen:
                continue
            seen.add(code)
            out.append({"code": code, "name": name, "market": mkt, "group": group})
    return out


def parse_csv(file) -> list[dict]:
    """
    讀使用者上傳的自訂清單。必要欄位：代號（或代碼）、名稱；選填：分類（或子類）、市場（TW/TWO，預設 TW）。
    回傳格式與 stocks_of() 相同。
    """
    import io
    import re

    import pandas as pd

    raw = file.read() if hasattr(file, "read") else file
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("無法辨識檔案編碼，請存成 UTF-8 的 CSV")
    df.columns = [c.strip() for c in df.columns]
    code_col = next((c for c in ("代號", "代碼", "股票代號", "code") if c in df.columns), None)
    if code_col is None or "名稱" not in df.columns:
        raise ValueError("CSV 第一列必須有「代號」與「名稱」欄；可選填「分類」「市場」（TW 上市／TWO 上櫃）")
    group_col = next((c for c in ("分類", "子類", "產業") if c in df.columns), None)
    out, seen = [], set()
    for _, r in df.fillna("").iterrows():
        code = str(r[code_col]).strip().upper()
        if not re.fullmatch(r"\d{4,6}[A-Z]?", code) or code in seen:
            continue
        seen.add(code)
        mkt = str(r.get("市場", "")).strip().upper()
        out.append({"code": code, "name": str(r["名稱"]).strip() or code,
                    "market": mkt if mkt in ("TW", "TWO") else "TW",
                    "group": (str(r[group_col]).strip() if group_col else "") or "自訂清單"})
    if not out:
        raise ValueError("CSV 裡沒有有效的股票代號")
    return out
