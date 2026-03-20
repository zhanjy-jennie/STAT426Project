import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time

START = "2018-01-01"
END_EXCL = "2026-01-01"

ASSETS = {
    "Cryptocurrencies": [
        "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD",
    ],
    "TraditionalStocks": [
        "BRK-B", "JNJ", "JPM", "PG", "V",
    ],
    "TechStocks": [
        "AAPL", "AMZN", "GOOGL", "META", "MSFT",
    ],
    "Forex": [
        "AUDUSD=X", "CADUSD=X", "EURUSD=X", "GBPUSD=X", "JPYUSD=X",
    ],
    "Commodities": [
        "GC=F", "HG=F", "KC=F",
    ],
    "Indices": ["^GSPC"],
}

TICKER2TYPE = {tk: tp for tp, lst in ASSETS.items() for tk in lst}
TICKERS = list(TICKER2TYPE.keys())

OUTDIR = Path.home() / "Desktop" / "Assets_Data"
OUTDIR.mkdir(parents=True, exist_ok=True)

def normalize_columns(df: pd.DataFrame, tk: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if tk in df.columns.get_level_values(-1):
            df = df.xs(tk, axis=1, level=-1)
        else:
            df = df.copy()
            df.columns = ["_".join([str(x) for x in c if str(x) != ""]).strip() for c in df.columns]
    return df

def download_one(tk: str) -> pd.DataFrame:
    df = yf.download(tk, start=START, end=END_EXCL, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = normalize_columns(df, tk).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.reset_index()  # Date column
    df["Date"] = pd.to_datetime(df["Date"])

    price_col = None
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        price_col = "Adj Close"
    elif "Close" in df.columns and df["Close"].notna().any():
        price_col = "Close"
    if price_col is None:
        return pd.DataFrame()

    df = df.sort_values("Date")
    df["Close_used"] = df[price_col].astype(float)
    df["Ticker"] = tk
    df["AssetType"] = TICKER2TYPE[tk]
    return df

print(f"[{datetime.now():%H:%M:%S}] Downloading AAPL for calendar ...")
cal_df = download_one("AAPL")
if cal_df.empty:
    raise RuntimeError("AAPL download failed; cannot build 5-day calendar.")

calendar = pd.DatetimeIndex(cal_df["Date"]).sort_values()

raw = {}
date_sets = []
missing = []

for tk in TICKERS:
    print(f"[{datetime.now():%H:%M:%S}] Downloading {tk}")
    df = download_one(tk)
    if df.empty:
        missing.append(tk)
        continue

    df = df[df["Date"].isin(calendar)].copy()
    df = df.sort_values("Date")
    raw[tk] = df
    date_sets.append(set(df["Date"].dropna().values))
    time.sleep(0.12)

if missing:
    raise RuntimeError(f"No data for: {missing}")

common_dates = sorted(set.intersection(*date_sets))
if len(common_dates) == 0:
    raise RuntimeError("Common dates are empty. Try a shorter period or check tickers.")

rows = []
for tk, df in raw.items():
    df2 = df[df["Date"].isin(common_dates)].copy()
    df2 = df2.sort_values("Date")
    df2["Return"] = df2["Close_used"].pct_change()

    keep = ["Date", "Ticker", "AssetType", "Close_used", "Return", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    keep = [c for c in keep if c in df2.columns]
    df2 = df2[keep]

    df2["Date"] = df2["Date"].dt.strftime("%Y-%m-%d")
    rows.append(df2)

panel = pd.concat(rows, ignore_index=True)

out_file = OUTDIR / "Aligned_Input_2018_2025.xlsx"
with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    panel.to_excel(writer, index=False, sheet_name="data")

print("Saved:", out_file)
print("Tickers:", panel["Ticker"].nunique(), "Dates:", panel["Date"].nunique(), "Rows:", len(panel))