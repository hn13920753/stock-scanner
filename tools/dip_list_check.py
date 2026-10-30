"""真實資料檢查：跌深清單（與 App「🪂 跌深清單」同一個函式）。輸出 research/dip_list.txt。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dip_view as DV  # noqa: E402
import scanner as S  # noqa: E402

stocks = DV.pool_stocks()
prices = S.fetch_batch([{"code": s["code"], "market": s["market"]} for s in stocks])
hit, near, fail, last = DV.scan_dip(stocks, prices, 35, 100_000)
L = [f"資料日 {last}｜共 {len(stocks)} 檔｜跌 ≥35%：{len(hit)} 檔｜接近：{len(near)} 檔｜失敗：{fail}"]
for tag, rows in (("已達門檻", hit), ("接近中", near)):
    L.append(f"## {tag}")
    for r in rows:
        L.append(f"{r['代號']} {r['名稱']}｜收盤 {r['收盤']:g}｜60日高 {r['60日高']:g}｜{r['距高點%']:+.1f}%｜觸發價 {r['觸發價']:g}"
                 f"｜參考 {r['參考股數']} 股｜停利 {r['停利參考(淨賺10%)']:g}｜停損 {r['停損參考(−20%)']:g}")
open("research/dip_list.txt", "w").write("\n".join(L))
print("\n".join(L))
