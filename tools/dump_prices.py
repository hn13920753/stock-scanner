"""核對用：輸出指定股票某段期間的 Yahoo 原始日K（未還原）與還原價。"""
import sys
import yfinance as yf
code, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
h = yf.Ticker(code + ".TW").history(start=start, end=end, auto_adjust=False, actions=False)
h["MA5"] = h["Close"].rolling(5).mean()
h["量(張)"] = (h["Volume"] / 1000).round(0)
h["VOL5"] = h["量(張)"].rolling(5).mean().round(0)
pd_out = h[["Open", "High", "Low", "Close", "Adj Close", "MA5", "量(張)", "VOL5"]].round(2)
pd_out.index = pd_out.index.strftime("%Y-%m-%d")
txt = pd_out.to_string()
open(f"research/dump_{code}.txt", "w").write(txt)
print(txt)
