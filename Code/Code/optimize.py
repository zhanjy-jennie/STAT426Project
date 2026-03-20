import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

# =========================
# 1) Config
# =========================
IN_PRED = Path.home() / "Desktop" / "Assets_Data" / "XGB_Predictions_AnnualRolling.xlsx"
IN_ALIGNED = Path.home() / "Desktop" / "Assets_Data" / "Aligned_Input_2018_2025.xlsx"
OUTFILE = Path.home() / "Desktop" / "Assets_Data" / "MAD_Optimization_AvgWeights.xlsx"

LOOKBACK_DAYS = 252
W_MIN = 0.02
LAMBDA = 0.10

CRYPTO_MAX = 0.25  # cap for the single crypto in each (All + crypto) portfolio

BACKTEST_START = "2022-01-01"
BACKTEST_END = "2025-12-31"

ASSETS = {
    "Cryptocurrencies": ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD"],
    "TraditionalStocks": ["BRK-B", "JNJ", "JPM", "PG", "V"],
    "TechStocks": ["AAPL", "AMZN", "GOOGL", "META", "MSFT"],
    "Forex": ["AUDUSD=X", "CADUSD=X", "EURUSD=X", "GBPUSD=X", "JPYUSD=X"],
    "Commodities": ["GC=F", "HG=F", "KC=F"],
}

CRYPTO_SET = set(ASSETS["Cryptocurrencies"])

NONCRYPTO_ALL = (
    ASSETS["TraditionalStocks"]
    + ASSETS["TechStocks"]
    + ASSETS["Forex"]
    + ASSETS["Commodities"]
)

PORTFOLIOS = [("NonCrypto_All", NONCRYPTO_ALL)]
for c in ASSETS["Cryptocurrencies"]:
    PORTFOLIOS.append((f"All+{c}", NONCRYPTO_ALL + [c]))


# =========================
# 2) Load inputs
# =========================
mu_hat = pd.read_excel(IN_PRED, sheet_name="mu_hat_matrix")
date_col = "Date" if "Date" in mu_hat.columns else ("RebalanceDate" if "RebalanceDate" in mu_hat.columns else mu_hat.columns[0])
mu_hat[date_col] = pd.to_datetime(mu_hat[date_col])
mu_hat = mu_hat.set_index(date_col).sort_index()

raw = pd.read_excel(IN_ALIGNED, sheet_name="data")
raw["Date"] = pd.to_datetime(raw["Date"])
raw = raw.dropna(subset=["Return"]).copy()
ret_daily = raw.pivot(index="Date", columns="Ticker", values="Return").sort_index()


# =========================
# 3) Mean–MAD solver (LP)
# =========================
def solve_mean_mad(mu_pred: np.ndarray, R_hist: np.ndarray, lam: float, w_min: float, upper: list[float]) -> np.ndarray:
    L, n = R_hist.shape

    if n * w_min > 1.0:
        w_min = max(0.0, 1.0 / n - 1e-6)

    A = R_hist - R_hist.mean(axis=0, keepdims=True)
    c = np.concatenate([-mu_pred, (lam / L) * np.ones(L)])

    I = np.eye(L)
    A_ub = np.vstack([
        np.hstack([ A, -I]),
        np.hstack([-A, -I]),
    ])
    b_ub = np.zeros(2 * L)

    A_eq = np.zeros((1, n + L))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    upper = [min(1.0, float(u)) for u in upper]
    bounds = [(w_min, u) for u in upper] + [(0.0, None) for _ in range(L)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        w = np.full(n, 1.0 / n)
        w = np.maximum(w, w_min)
        w = w / w.sum()
        return w

    return res.x[:n]


# =========================
# 4) Utilities
# =========================
def month_end_dates(dts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dts)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().sort_values().to_numpy())

def safe_sheet_name(s: str) -> str:
    bad = ["=", "^", "/", "\\", ":", "*", "?", "[", "]"]
    for ch in bad:
        s = s.replace(ch, "_")
    return s[:31]


# =========================
# 5) Backtest (metrics + avg weights)
# =========================
def run_portfolio(pname: str, tickers: list[str]) -> dict:
    miss_mu = [t for t in tickers if t not in mu_hat.columns]
    miss_ret = [t for t in tickers if t not in ret_daily.columns]
    if miss_mu:
        return {"Portfolio": pname, "Status": f"Missing mu_hat cols: {miss_mu[:5]}"}
    if miss_ret:
        return {"Portfolio": pname, "Status": f"Missing return cols: {miss_ret[:5]}"}

    mu_p = mu_hat[tickers]
    ret_p = ret_daily[tickers]

    reb = mu_p.dropna().index.intersection(ret_p.index)
    reb = reb[(reb >= pd.to_datetime(BACKTEST_START)) & (reb <= pd.to_datetime(BACKTEST_END))]
    if len(reb) < 2:
        return {"Portfolio": pname, "Status": "Not enough rebalance dates in range"}

    reb = month_end_dates(pd.DatetimeIndex(reb))

    w_rows = []
    daily_parts = []
    used_periods = 0

    upper = [CRYPTO_MAX if tk in CRYPTO_SET else 1.0 for tk in tickers]

    for t0, t1 in zip(reb[:-1], reb[1:]):
        hist = ret_p.loc[ret_p.index <= t0].tail(LOOKBACK_DAYS).dropna(how="any")
        if len(hist) < LOOKBACK_DAYS:
            continue

        hold = ret_p.loc[(ret_p.index > t0) & (ret_p.index <= t1)].dropna(how="any")
        if hold.empty:
            continue

        mu_vec = mu_p.loc[t0].to_numpy(dtype=float)
        w = solve_mean_mad(mu_vec, hist.to_numpy(dtype=float), LAMBDA, W_MIN, upper)

        daily_parts.append(pd.Series(hold.to_numpy(dtype=float) @ w, index=hold.index))
        used_periods += 1

        for tk, ww in zip(tickers, w):
            w_rows.append({"Portfolio": pname, "RebalanceDate": t0, "Ticker": tk, "Weight": float(ww)})

    if not daily_parts:
        return {"Portfolio": pname, "Status": "No valid periods (lookback/NaNs)"}

    daily = pd.concat(daily_parts).sort_index()
    n_days = int(len(daily))

    ann_return = float(np.prod(1.0 + daily.values) ** (252.0 / n_days) - 1.0)
    ann_vol = float(daily.std(ddof=1) * np.sqrt(252.0))
    sharpe = float((daily.mean() / daily.std(ddof=1)) * np.sqrt(252.0)) if daily.std(ddof=1) > 0 else np.nan

    w_df = pd.DataFrame(w_rows)
    avg_w = (
        w_df.groupby(["Portfolio", "Ticker"], as_index=False)["Weight"]
            .mean()
            .rename(columns={"Weight": "AvgWeight"})
            .sort_values(["AvgWeight", "Ticker"], ascending=[False, True])
            .reset_index(drop=True)
    )

    return {
        "Portfolio": pname,
        "Status": "OK",
        "NumAssets": len(tickers),
        "RebalancePeriodsUsed": used_periods,
        "StartDate": daily.index.min(),
        "EndDate": daily.index.max(),
        "TradingDays": n_days,
        "AnnualReturn": ann_return,
        "AnnualVol": ann_vol,
        "Sharpe": sharpe,
        "avg_weights": avg_w,
    }


# =========================
# 6) Run all & save (summary + per-portfolio avg weights)
# =========================
results = []
weights_sheets = {}

for pname, tickers in PORTFOLIOS:
    r = run_portfolio(pname, tickers)
    results.append({k: r.get(k) for k in [
        "Portfolio", "Status", "NumAssets", "RebalancePeriodsUsed",
        "StartDate", "EndDate", "TradingDays", "AnnualReturn", "AnnualVol", "Sharpe"
    ]})
    if r.get("Status") == "OK":
        weights_sheets[pname] = r["avg_weights"]

summary = pd.DataFrame(results)

base = summary.loc[summary["Portfolio"] == "NonCrypto_All"]
if len(base) == 1 and base.iloc[0]["Status"] == "OK":
    b = base.iloc[0]
    summary["DeltaSharpe_vs_Base"] = summary["Sharpe"] - b["Sharpe"]
    summary["DeltaReturn_vs_Base"] = summary["AnnualReturn"] - b["AnnualReturn"]
    summary["DeltaVol_vs_Base"] = summary["AnnualVol"] - b["AnnualVol"]

with pd.ExcelWriter(OUTFILE, engine="openpyxl") as w:
    summary.to_excel(w, index=False, sheet_name="summary")
    for pname, dfw in weights_sheets.items():
        dfw.to_excel(w, index=False, sheet_name=safe_sheet_name(pname))

print("Saved:", OUTFILE)
print(summary[["Portfolio", "Status", "Sharpe", "DeltaSharpe_vs_Base"]])