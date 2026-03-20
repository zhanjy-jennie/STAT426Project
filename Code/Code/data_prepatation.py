import pandas as pd
import numpy as np
from pathlib import Path

INFILE  = Path.home() / "Desktop" / "Assets_Data" / "Aligned_Input_2018_2025.xlsx"
OUTFILE = Path.home() / "Desktop" / "Assets_Data" / "ML_Input_Preprocessed.xlsx"

LAGS = 20
H    = 21


# helper
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def compute_atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


df = pd.read_excel(INFILE, sheet_name="data")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
df = df[df["Return"].notna()].copy()


# --------------------------------------------------
# 1. first take out SP500
# --------------------------------------------------
sp500 = df[df["Ticker"].astype(str).isin(["^GSPC", "GSPC"])].copy()

if sp500.empty:
    raise RuntimeError("SP500 (^GSPC or GSPC) not found in input file.")

sp500 = sp500.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# 2. build SP500 market features first
# --------------------------------------------------
sp500["sp500_ret"] = sp500["Return"]
sp500["sp500_mean5"] = sp500["Return"].rolling(5).mean()
sp500["sp500_vol20"] = sp500["Return"].rolling(20).std(ddof=0)
sp500["sp500_mom20"] = (
    sp500["Return"].clip(lower=-0.999999).add(1.0).rolling(20).apply(np.prod, raw=True) - 1.0
)

sp500_keep = ["Date", "sp500_ret", "sp500_mean5", "sp500_vol20", "sp500_mom20"]
sp500_feat = sp500[sp500_keep].copy()


# --------------------------------------------------
# 3. merge SP500 features back to all assets
# --------------------------------------------------
df = df.merge(sp500_feat, on="Date", how="left")


# --------------------------------------------------
# 4. remove SP500 itself from prediction targets
# --------------------------------------------------
df = df[~df["Ticker"].astype(str).isin(["^GSPC", "GSPC"])].copy()


df["y_next"] = (
    df.groupby("Ticker")["Close_used"].shift(-H) / df["Close_used"] - 1
)

lag_cols = [f"lag{k}" for k in range(1, LAGS + 1)]
for k in range(1, LAGS + 1):
    df[f"lag{k}"] = df.groupby("Ticker")["Return"].shift(k - 1)


df["mean5"]   = df[[f"lag{k}" for k in range(1, 6)]].mean(axis=1)
df["vol5"]    = df[[f"lag{k}" for k in range(1, 6)]].std(axis=1, ddof=0)
df["mean20"]  = df[lag_cols].mean(axis=1)
df["vol20"]   = df[lag_cols].std(axis=1, ddof=0)
df["mom20"]   = df[lag_cols].clip(lower=-0.999999).add(1.0).prod(axis=1) - 1.0
df["range20"] = df[lag_cols].max(axis=1) - df[lag_cols].min(axis=1)


# add new feature
def add_features(g):
    g = g.copy()
    g["rsi14"] = compute_rsi(g["Return"], 14)
    atr = compute_atr(g["High"], g["Low"], g["Close_used"], 14)
    g["atr14_norm"] = atr / g["Close_used"]

    g["vol_ratio"] = g["vol5"] / (g["vol20"] + 1e-10)
    g["skew20"] = g[lag_cols].skew(axis=1)
    roll_high = g["Close_used"].shift(1).rolling(20).max()
    roll_low  = g["Close_used"].shift(1).rolling(20).min()
    denom     = (roll_high - roll_low).replace(0, np.nan)
    g["price_pos20"] = (g["Close_used"] - roll_low) / denom
    ema20 = g["Close_used"].ewm(span=20, adjust=False).mean()
    g["ema_dev20"] = (g["Close_used"] - ema20) / (ema20 + 1e-10)

    if "Volume" in g.columns:
        vol_ma5  = g["Volume"].rolling(5).mean()
        vol_ma20 = g["Volume"].rolling(20).mean()
        g["vol_surge"] = g["Volume"] / (vol_ma5 + 1e-10)
        g["vol_trend"] = vol_ma5 / (vol_ma20 + 1e-10)
        no_vol = (g["Volume"] == 0) | g["Volume"].isna()
        g.loc[no_vol, ["vol_surge", "vol_trend"]] = np.nan
    else:
        g["vol_surge"] = np.nan
        g["vol_trend"] = np.nan

    return g

df = df.groupby("Ticker", group_keys=False).apply(add_features)


avg_sector_ret = (
    df.groupby(["Date", "AssetType"])["lag1"]
    .transform("mean")
)
df["sector_mom"] = avg_sector_ret
df["asset_type_code"] = df["AssetType"].astype("category").cat.codes
df["month"]       = df["Date"].dt.month
df["day_of_week"] = df["Date"].dt.dayofweek


base_cols = ["mean5", "vol5", "mean20", "vol20", "mom20", "range20"]
new_cols  = [
    "rsi14", "atr14_norm", "vol_ratio", "skew20",
    "price_pos20", "ema_dev20", "vol_surge", "vol_trend",
    "sector_mom", "asset_type_code", "month", "day_of_week",
]
market_cols = ["sp500_ret", "sp500_mean5", "sp500_vol20", "sp500_mom20"]

feature_cols = lag_cols + base_cols + new_cols + market_cols


keep = ["Date", "Ticker", "AssetType", "y_next"] + feature_cols
ml   = df[keep].dropna(subset=lag_cols + base_cols + ["y_next"]).copy()

# new features: fill remaining NaN with 0
ml[new_cols + market_cols] = ml[new_cols + market_cols].fillna(0)

ml = ml.sort_values(["Ticker", "Date"]).reset_index(drop=True)

check = ml.groupby("Ticker")["Date"].agg(["min", "max", "count"]).reset_index()
print(check.head())
print("Tickers:", ml["Ticker"].nunique(), "Rows:", len(ml))
print("Features:", len(feature_cols))
print("SP500 in final data:", ml["Ticker"].astype(str).isin(["^GSPC", "GSPC"]).any())

out = ml.copy()
out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

with pd.ExcelWriter(OUTFILE, engine="openpyxl") as w:
    out.to_excel(w, index=False, sheet_name="ml_input")

print("Saved:", OUTFILE)