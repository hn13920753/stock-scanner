import pandas as pd, yfinance as yf
d = yf.Ticker("0050.TW").history(period="max", auto_adjust=True, actions=False)
d.index = pd.to_datetime(d.index).tz_localize(None)
print("rows", len(d), d.index[0], d.index[-1])
print("Open<=0:", (d.Open <= 0).sum(), "NaN open:", d.Open.isna().sum())
r = d.Close.pct_change()
print("big daily moves:\n", r[abs(r) > 0.15])
o = d.Open / d.Close.shift(1) - 1
print("big open gaps:\n", o[abs(o) > 0.15])
print(d.loc["2025-06-05":"2025-06-25", ["Open", "Close"]])
print(d.resample("YE").last()[["Close"]])
d2 = yf.download("0050.TW", period="max", auto_adjust=True, progress=False)
print("download rows", len(d2), d2.index[0])
