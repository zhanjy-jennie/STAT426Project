import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time

INFILE      = Path.home() / "Desktop" / "Assets_Data" / "ML_Input_Preprocessed.xlsx"
OUT_PRED    = Path.home() / "Desktop" / "Assets_Data" / "RF_Predictions_AnnualRolling.xlsx"
OUT_METRICS = Path.home() / "Desktop" / "Assets_Data" / "RF_Metrics_AnnualRolling.xlsx"
OUT_REPORT  = Path.home() / "Desktop" / "Assets_Data" / "RF_Training_Report.xlsx"

LAGS        = 20
H_FWD       = 21
TRAIN_YEARS = 4
VAL_YEARS   = 1
TEST_YEARS  = [2022, 2023, 2024, 2025]

TARGET_RAW  = "y_next"
TARGET      = f"ret_fwd_{H_FWD}d"

LAG_COLS    = [f"lag{k}" for k in range(1, LAGS + 1)]
EXTRA_COLS  = [
    "mean5", "vol5", "mean20", "vol20", "mom20", "range20",
    "rsi14", "atr14_norm", "vol_ratio", "skew20",
    "price_pos20", "ema_dev20", "vol_surge", "vol_trend",
    "sector_mom", "asset_type_code", "month", "day_of_week",
    "sp500_ret", "sp500_mean5", "sp500_vol20", "sp500_mom20",
]
FEATURES    = LAG_COLS + EXTRA_COLS

def make_rf():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5,
        max_features=0.4,
        max_samples=0.8,
    )

def directional_accuracy(y_true, y_pred):
    return float(np.mean(np.sign(np.asarray(y_true)) == np.sign(np.asarray(y_pred))))

def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":    float(mean_absolute_error(y_true, y_pred)),
        "RMSE":   float(np.sqrt(mse)),
        "DirAcc": directional_accuracy(y_true, y_pred),
    }

def split_train_val_test(df_one, test_year):
    train_start = test_year - (TRAIN_YEARS + VAL_YEARS)
    train_end   = test_year - 2
    val_year    = test_year - 1

    train = df_one[(df_one["Date"].dt.year >= train_start) & (df_one["Date"].dt.year <= train_end)].copy()
    val   = df_one[df_one["Date"].dt.year == val_year].copy()
    test  = df_one[df_one["Date"].dt.year == test_year].copy()

    if train.empty or val.empty or test.empty:
        return None

    return (
        train, val, test,
        train[FEATURES].to_numpy(), train[TARGET].to_numpy(),
        val[FEATURES].to_numpy(),   val[TARGET].to_numpy(),
        test[FEATURES].to_numpy(),  test[TARGET].to_numpy()
    )

df = pd.read_excel(INFILE, sheet_name="ml_input")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

if TARGET not in df.columns:
    if TARGET_RAW in df.columns:
        df = df.rename(columns={TARGET_RAW: TARGET})
    else:
        raise ValueError(f"Target column not found: '{TARGET_RAW}' or '{TARGET}'")

for c in EXTRA_COLS:
    if c in df.columns:
        df[c] = df[c].fillna(0)
    else:
        df[c] = 0.0

need_cols = ["Date", "Ticker", "AssetType"] + FEATURES + [TARGET]
missing   = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=LAG_COLS + [TARGET]).copy()

universe = (
    df[["Ticker", "AssetType"]].drop_duplicates()
    .sort_values(["AssetType", "Ticker"]).reset_index(drop=True)
)

pred_parts  = []
metric_rows = []
time_rows   = []

overall_start = time.perf_counter()

for tk in sorted(df["Ticker"].unique()):
    df_one     = df[df["Ticker"] == tk].copy()
    asset_type = df_one["AssetType"].iloc[0]

    for test_year in TEST_YEARS:
        pack = split_train_val_test(df_one, test_year)
        if pack is None:
            continue

        train, val, test, X_tr, y_tr, X_val, y_val, X_te, y_te = pack

        model = make_rf()

        fit_start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_end = time.perf_counter()
        fit_seconds = fit_end - fit_start

        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_te)

        pred_parts.append(pd.DataFrame({
            "Date":            test["Date"].values,
            "Ticker":          tk,
            "AssetType":       asset_type,
            "TestYear":        test_year,
            f"{TARGET}_true":  y_te,
            f"{TARGET}_pred":  test_pred,
        }))

        val_m  = eval_metrics(y_val, val_pred)
        test_m = eval_metrics(y_te,  test_pred)
        train_start = test_year - (TRAIN_YEARS + VAL_YEARS)
        train_end   = test_year - 2

        metric_rows.append({
            "Ticker": tk, "AssetType": asset_type, "TestYear": test_year,
            "Target": TARGET, "TrainWindow": f"{train_start}-{train_end}",
            "ValYear": test_year - 1,
            "N_train": len(train), "N_val": len(val), "N_test": len(test),
            "Val_MAE": val_m["MAE"], "Val_RMSE": val_m["RMSE"], "Val_DirAcc": val_m["DirAcc"],
            "Test_MAE": test_m["MAE"], "Test_RMSE": test_m["RMSE"], "Test_DirAcc": test_m["DirAcc"],
        })

        time_rows.append({
            "Ticker": tk,
            "AssetType": asset_type,
            "TestYear": test_year,
            "TrainWindow": f"{train_start}-{train_end}",
            "ValYear": test_year - 1,
            "TrainSeconds": fit_seconds,
            "TrainMinutes": fit_seconds / 60,
            "N_train": len(train),
            "N_val": len(val),
            "N_test": len(test),
            "N_features": len(FEATURES),
        })

overall_end = time.perf_counter()
total_seconds = overall_end - overall_start

if not pred_parts:
    raise ValueError("No predictions generated.")

pred_daily = pd.concat(pred_parts, ignore_index=True).sort_values(["Ticker", "Date"]).reset_index(drop=True)
metrics    = pd.DataFrame(metric_rows).sort_values(["Ticker", "TestYear"]).reset_index(drop=True)
time_detail = pd.DataFrame(time_rows).sort_values(["Ticker", "TestYear"]).reset_index(drop=True)

eom_idx  = pred_daily.groupby(["Ticker", pred_daily["Date"].dt.to_period("M")])["Date"].idxmax()
pred_eom = pred_daily.loc[eom_idx].copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)
pred_eom["YearMonth"]     = pred_eom["Date"].dt.to_period("M").astype(str)
pred_eom["RebalanceDate"] = pred_eom["Date"]
pred_eom["mu_hat"]        = pred_eom[f"{TARGET}_pred"]

true_col = f"{TARGET}_true"

mu_hat_matrix   = pred_eom.pivot(index="RebalanceDate", columns="Ticker", values="mu_hat").sort_index()
ret_true_matrix = pred_eom.pivot(index="RebalanceDate", columns="Ticker", values=true_col).sort_index()

def ic_one_date(g):
    tmp = g[["mu_hat", true_col]].dropna()
    n   = len(tmp)
    if n < 3:
        return pd.Series({"IC": np.nan, "N": n})
    return pd.Series({"IC": float(tmp["mu_hat"].corr(tmp[true_col], method="spearman")), "N": n})

ic_df = (
    pred_eom.groupby("RebalanceDate", as_index=False).apply(ic_one_date)
    .reset_index(drop=True).sort_values("RebalanceDate")
)
ic_df["IC_rolling3"] = ic_df["IC"].rolling(3, min_periods=1).mean()

ic_valid = ic_df["IC"].dropna()
ic_mean  = float(ic_valid.mean())
ic_std   = float(ic_valid.std(ddof=1))
ic_t     = ic_mean / (ic_std / np.sqrt(len(ic_valid))) if ic_std > 0 else np.nan
ic_summary = f"IC mean={ic_mean:.4f} | std={ic_std:.4f} | t≈{ic_t:.2f} | months={len(ic_valid)}"

plt.figure(figsize=(11, 4.5))
plt.plot(ic_df["RebalanceDate"], ic_df["IC"], marker="o", markersize=3.5, linewidth=1.2, alpha=0.9, label="Monthly IC")
plt.plot(ic_df["RebalanceDate"], ic_df["IC_rolling3"], linewidth=2.2, alpha=0.95, label="3M Rolling IC")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title(f"RF — Monthly IC (Spearman)\n{ic_summary}", pad=10)
plt.xlabel("RebalanceDate")
plt.ylabel("IC")
plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
print(ic_summary)

avg_fit_seconds = float(time_detail["TrainSeconds"].mean()) if not time_detail.empty else np.nan
avg_fit_minutes = avg_fit_seconds / 60 if pd.notna(avg_fit_seconds) else np.nan
total_training_seconds = float(time_detail["TrainSeconds"].sum()) if not time_detail.empty else np.nan
total_training_minutes = total_training_seconds / 60 if pd.notna(total_training_seconds) else np.nan
estimated_cpu_hours = total_training_seconds / 3600 if pd.notna(total_training_seconds) else np.nan

compute_time_summary = pd.DataFrame({
    "Item": [
        "Average training time per fit (seconds)",
        "Average training time per fit (minutes)",
        "Total training time for final models (seconds)",
        "Total training time for final models (minutes)",
        "Estimated CPU hours for current run",
        "Number of model fits",
        "Number of tickers",
        "Number of test years",
        "Total wall-clock time of whole script (seconds)",
        "Total wall-clock time of whole script (minutes)",
    ],
    "Value": [
        avg_fit_seconds,
        avg_fit_minutes,
        total_training_seconds,
        total_training_minutes,
        estimated_cpu_hours,
        len(time_detail),
        df["Ticker"].nunique(),
        len(TEST_YEARS),
        total_seconds,
        total_seconds / 60,
    ]
})

rf_params = make_rf().get_params()
hyperparameters = pd.DataFrame({
    "Hyperparameter": list(rf_params.keys()),
    "Value": list(rf_params.values())
}).sort_values("Hyperparameter").reset_index(drop=True)

def fmt_date(s):
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")

pred_daily_save = pred_daily.copy()
pred_daily_save["Date"] = fmt_date(pred_daily_save["Date"])

pred_eom_save = pred_eom.copy()
pred_eom_save["Date"] = fmt_date(pred_eom_save["Date"])
pred_eom_save["RebalanceDate"] = fmt_date(pred_eom_save["RebalanceDate"])

mu_save = mu_hat_matrix.copy()
mu_save.index = pd.to_datetime(mu_save.index).strftime("%Y-%m-%d")
mu_save = mu_save.reset_index().rename(columns={"RebalanceDate": "Date"})

ret_save = ret_true_matrix.copy()
ret_save.index = pd.to_datetime(ret_save.index).strftime("%Y-%m-%d")
ret_save = ret_save.reset_index().rename(columns={"RebalanceDate": "Date"})

with pd.ExcelWriter(OUT_PRED, engine="openpyxl") as w:
    universe.to_excel(w, index=False, sheet_name="universe")
    pred_daily_save.to_excel(w, index=False, sheet_name="pred_daily")
    pred_eom_save.to_excel(w, index=False, sheet_name="pred_eom")
    mu_save.to_excel(w, index=False, sheet_name="mu_hat_matrix")
    ret_save.to_excel(w, index=False, sheet_name="ret_true_matrix")

with pd.ExcelWriter(OUT_METRICS, engine="openpyxl") as w:
    metrics.to_excel(w, index=False, sheet_name="metrics")

with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as w:
    compute_time_summary.to_excel(w, index=False, sheet_name="compute_time_summary")
    time_detail.to_excel(w, index=False, sheet_name="compute_time_detail")
    hyperparameters.to_excel(w, index=False, sheet_name="hyperparameters")

print("Saved:", OUT_PRED)
print("Saved:", OUT_METRICS)
print("Saved:", OUT_REPORT)
print(f"Rows(pred_daily):{len(pred_daily)} | Rows(pred_eom):{len(pred_eom)} | Rows(metrics):{len(metrics)}")