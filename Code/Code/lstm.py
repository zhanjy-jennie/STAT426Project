import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, backend as K


INFILE = Path.home() / "Desktop" / "Assets_Data" / "ML_Input_Preprocessed.xlsx"
OUT_PRED = Path.home() / "Desktop" / "Assets_Data" / "LSTM_Predictions_AnnualRolling.xlsx"
OUT_METRICS = Path.home() / "Desktop" / "Assets_Data" / "LSTM_Metrics_AnnualRolling.xlsx"

OUT_REPORT = Path.home() / "Desktop" / "Assets_Data" / "LSTM_Training_Report.xlsx"
OUT_LC_PLOT = Path.home() / "Desktop" / "Assets_Data" / "LSTM_Learning_Curve.png"

LAGS = 20
H_FWD = 21
TRAIN_YEARS = 4
VAL_YEARS = 1
TEST_YEARS = [2022, 2023, 2024, 2025]

TARGET_RAW = "y_next"
TARGET = f"ret_fwd_{H_FWD}d"

LAG_COLS = [f"lag{k}" for k in range(1, LAGS + 1)]
EXTRA_COLS = [
    "mean5", "vol5", "mean20", "vol20", "mom20", "range20",
    "rsi14", "atr14_norm", "vol_ratio", "skew20",
    "price_pos20", "ema_dev20", "vol_surge", "vol_trend",
    "sector_mom", "asset_type_code", "month", "day_of_week",
    "sp500_ret", "sp500_mean5", "sp500_vol20", "sp500_mom20",
]
FEATURES = LAG_COLS + EXTRA_COLS

MAX_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 10
SEQ_LSTM_UNITS_1 = 64
SEQ_LSTM_UNITS_2 = 32
STATIC_DENSE_UNITS = 32
FUSION_DENSE_UNITS_1 = 64
FUSION_DENSE_UNITS_2 = 32
DROPOUT_RATE = 0.3

tf.random.set_seed(42)
np.random.seed(42)


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "DirAcc": directional_accuracy(y_true, y_pred),
    }


def split_train_val_test(df_one: pd.DataFrame, test_year: int):
    train_start = test_year - (TRAIN_YEARS + VAL_YEARS)
    train_end = test_year - 2
    val_year = test_year - 1

    train = df_one[(df_one["Date"].dt.year >= train_start) & (df_one["Date"].dt.year <= train_end)].copy()
    val = df_one[df_one["Date"].dt.year == val_year].copy()
    test = df_one[df_one["Date"].dt.year == test_year].copy()

    if train.empty or val.empty or test.empty:
        return None

    seq_cols = list(reversed(LAG_COLS))

    X_train_seq = train[seq_cols].to_numpy().reshape(-1, LAGS, 1)
    y_train = train[TARGET].to_numpy()

    X_val_seq = val[seq_cols].to_numpy().reshape(-1, LAGS, 1)
    y_val = val[TARGET].to_numpy()

    X_test_seq = test[seq_cols].to_numpy().reshape(-1, LAGS, 1)
    y_test = test[TARGET].to_numpy()

    X_train_static = train[EXTRA_COLS].to_numpy()
    X_val_static = val[EXTRA_COLS].to_numpy()
    X_test_static = test[EXTRA_COLS].to_numpy()

    return (
        train, val, test,
        X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test,
        X_train_static, X_val_static, X_test_static
    )


def make_lstm():
    seq_in = layers.Input(shape=(LAGS, 1), name="seq")
    x = layers.LSTM(SEQ_LSTM_UNITS_1, return_sequences=True)(seq_in)
    x = layers.LSTM(SEQ_LSTM_UNITS_2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    static_in = layers.Input(shape=(len(EXTRA_COLS),), name="static")
    s = layers.Dense(STATIC_DENSE_UNITS, activation="relu")(static_in)
    s = layers.BatchNormalization()(s)

    z = layers.Concatenate()([x, s])
    z = layers.Dense(FUSION_DENSE_UNITS_1, activation="relu")(z)
    z = layers.Dropout(DROPOUT_RATE)(z)
    z = layers.Dense(FUSION_DENSE_UNITS_2, activation="relu")(z)
    out = layers.Dense(1, name="out")(z)

    model = models.Model(inputs=[seq_in, static_in], outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


df = pd.read_excel(INFILE, sheet_name="ml_input")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

if TARGET not in df.columns:
    if TARGET_RAW in df.columns:
        df = df.rename(columns={TARGET_RAW: TARGET})
    else:
        raise ValueError(f"Target column not found: '{TARGET_RAW}' or '{TARGET}'")

need_cols = ["Date", "Ticker", "AssetType"] + FEATURES + [TARGET]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input file: {missing}")

df = df.dropna(subset=FEATURES + [TARGET]).copy()

universe = (
    df[["Ticker", "AssetType"]]
    .drop_duplicates()
    .sort_values(["AssetType", "Ticker"])
    .reset_index(drop=True)
)


pred_parts = []
metric_rows = []
time_rows = []
curve_parts = []

overall_start = time.perf_counter()

for tk in sorted(df["Ticker"].unique()):
    df_one = df[df["Ticker"] == tk].copy()
    asset_type = df_one["AssetType"].iloc[0]

    for test_year in TEST_YEARS:
        pack = split_train_val_test(df_one, test_year)
        if pack is None:
            continue

        (
            train, val, test,
            X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test,
            X_train_static, X_val_static, X_test_static
        ) = pack

        seq_scaler = StandardScaler()
        static_scaler = StandardScaler()

        seq_scaler.fit(X_train_seq.reshape(-1, 1))
        X_train_seq = seq_scaler.transform(X_train_seq.reshape(-1, 1)).reshape(-1, LAGS, 1)
        X_val_seq = seq_scaler.transform(X_val_seq.reshape(-1, 1)).reshape(-1, LAGS, 1)
        X_test_seq = seq_scaler.transform(X_test_seq.reshape(-1, 1)).reshape(-1, LAGS, 1)

        static_scaler.fit(X_train_static)
        X_train_static = static_scaler.transform(X_train_static)
        X_val_static = static_scaler.transform(X_val_static)
        X_test_static = static_scaler.transform(X_test_static)

        K.clear_session()
        model = make_lstm()

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )

        fit_start = time.perf_counter()

        history = model.fit(
            {"seq": X_train_seq, "static": X_train_static},
            y_train,
            validation_data=({"seq": X_val_seq, "static": X_val_static}, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[es]
        )

        fit_end = time.perf_counter()
        fit_seconds = fit_end - fit_start
        epochs_trained = len(history.history["loss"])
        avg_epoch_seconds = fit_seconds / epochs_trained if epochs_trained > 0 else np.nan

        val_pred = model.predict({"seq": X_val_seq, "static": X_val_static}, verbose=0).reshape(-1)
        test_pred = model.predict({"seq": X_test_seq, "static": X_test_static}, verbose=0).reshape(-1)

        pred_parts.append(pd.DataFrame({
            "Date": test["Date"].values,
            "Ticker": tk,
            "AssetType": asset_type,
            "TestYear": test_year,
            f"{TARGET}_true": y_test,
            f"{TARGET}_pred": test_pred
        }))

        val_m = eval_metrics(y_val, val_pred)
        test_m = eval_metrics(y_test, test_pred)

        train_start = test_year - (TRAIN_YEARS + VAL_YEARS)
        train_end = test_year - 2

        metric_rows.append({
            "Ticker": tk,
            "AssetType": asset_type,
            "TestYear": test_year,
            "Target": TARGET,
            "TrainWindow": f"{train_start}-{train_end}",
            "ValYear": test_year - 1,
            "N_train": int(len(train)),
            "N_val": int(len(val)),
            "N_test": int(len(test)),
            "Val_MAE": val_m["MAE"],
            "Val_RMSE": val_m["RMSE"],
            "Val_DirAcc": val_m["DirAcc"],
            "Test_MAE": test_m["MAE"],
            "Test_RMSE": test_m["RMSE"],
            "Test_DirAcc": test_m["DirAcc"],
        })

        time_rows.append({
            "Ticker": tk,
            "AssetType": asset_type,
            "TestYear": test_year,
            "TrainWindow": f"{train_start}-{train_end}",
            "ValYear": test_year - 1,
            "TrainSeconds": fit_seconds,
            "TrainMinutes": fit_seconds / 60,
            "EpochsTrained": epochs_trained,
            "AvgSecondsPerEpoch": avg_epoch_seconds,
            "N_train": int(len(train)),
            "N_val": int(len(val)),
            "N_test": int(len(test)),
            "N_features": len(FEATURES),
        })

        curve_parts.append(pd.DataFrame({
            "Ticker": tk,
            "AssetType": asset_type,
            "TestYear": test_year,
            "Epoch": np.arange(1, epochs_trained + 1),
            "Train_Loss": history.history["loss"],
            "Val_Loss": history.history["val_loss"],
        }))

overall_end = time.perf_counter()
total_seconds = overall_end - overall_start

if not pred_parts:
    raise ValueError("No predictions generated. Check year ranges / data coverage / NA dropping.")

pred_daily = (
    pd.concat(pred_parts, ignore_index=True)
    .sort_values(["Ticker", "Date"])
    .reset_index(drop=True)
)

metrics = (
    pd.DataFrame(metric_rows)
    .sort_values(["Ticker", "TestYear"])
    .reset_index(drop=True)
)

time_detail = (
    pd.DataFrame(time_rows)
    .sort_values(["Ticker", "TestYear"])
    .reset_index(drop=True)
)

learning_curve_detail = (
    pd.concat(curve_parts, ignore_index=True)
    .sort_values(["Ticker", "TestYear", "Epoch"])
    .reset_index(drop=True)
)

learning_curve_mean = (
    learning_curve_detail.groupby("Epoch", as_index=False)[["Train_Loss", "Val_Loss"]]
    .mean()
    .reset_index(drop=True)
)


eom_idx = pred_daily.groupby(["Ticker", pred_daily["Date"].dt.to_period("M")])["Date"].idxmax()

pred_eom = (
    pred_daily.loc[eom_idx]
    .copy()
    .sort_values(["Ticker", "Date"])
    .reset_index(drop=True)
)

pred_eom["YearMonth"] = pred_eom["Date"].dt.to_period("M").astype(str)
pred_eom["RebalanceDate"] = pred_eom["Date"]
pred_eom["mu_hat"] = pred_eom[f"{TARGET}_pred"]
pred_eom["mu_hat_horizon"] = f"{H_FWD}d_fwd"

true_col = f"{TARGET}_true"

mu_hat_matrix = (
    pred_eom.pivot(index="RebalanceDate", columns="Ticker", values="mu_hat")
    .sort_index()
)

ret_true_matrix = (
    pred_eom.pivot(index="RebalanceDate", columns="Ticker", values=true_col)
    .sort_index()
)


def ic_one_date(g: pd.DataFrame) -> pd.Series:
    tmp = g[["mu_hat", true_col]].dropna()
    n = int(len(tmp))
    if n < 3:
        return pd.Series({"IC": np.nan, "N": n})
    ic = float(tmp["mu_hat"].corr(tmp[true_col], method="spearman"))
    return pd.Series({"IC": ic, "N": n})

ic_df = (
    pred_eom.groupby("RebalanceDate", as_index=False)
    .apply(ic_one_date)
    .reset_index(drop=True)
    .sort_values("RebalanceDate")
)

ic_df["IC_rolling3"] = ic_df["IC"].rolling(3, min_periods=1).mean()

ic_valid = ic_df["IC"].dropna()
if len(ic_valid) >= 2:
    ic_mean = float(ic_valid.mean())
    ic_std = float(ic_valid.std(ddof=1))
    ic_t = ic_mean / (ic_std / np.sqrt(len(ic_valid))) if ic_std > 0 else np.nan
    ic_summary = f"IC mean={ic_mean:.4f} | std={ic_std:.4f} | t≈{ic_t:.2f} | months={len(ic_valid)}"
else:
    ic_summary = f"Not enough IC points. months={len(ic_valid)}"

plt.figure(figsize=(11, 4.5))

plt.plot(
    ic_df["RebalanceDate"], ic_df["IC"],
    marker="o", markersize=3.5, linewidth=1.2, alpha=0.9,
    label="Monthly IC"
)

plt.plot(
    ic_df["RebalanceDate"], ic_df["IC_rolling3"],
    linewidth=2.2, alpha=0.95,
    label="3M Rolling IC"
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.title(f"Monthly IC (Spearman): mu_hat vs realized {TARGET}\n{ic_summary}", pad=10)
plt.xlabel("RebalanceDate")
plt.ylabel("IC")

plt.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.8)
plt.legend(frameon=False)

plt.tight_layout()
plt.show()

print(ic_summary)


avg_epoch_seconds = float(time_detail["AvgSecondsPerEpoch"].mean()) if not time_detail.empty else np.nan
avg_fit_seconds = float(time_detail["TrainSeconds"].mean()) if not time_detail.empty else np.nan
total_training_seconds = float(time_detail["TrainSeconds"].sum()) if not time_detail.empty else np.nan

compute_time_summary = pd.DataFrame({
    "Item": [
        "Average wall-clock time per training epoch (seconds)",
        "Average wall-clock time per training epoch (minutes)",
        "Average training time per final model fit (seconds)",
        "Average training time per final model fit (minutes)",
        "Total training time for final models (seconds)",
        "Total training time for final models (minutes)",
        "Estimated CPU/GPU hours for current run",
        "Number of model fits",
        "Number of tickers",
        "Number of test years",
        "Total wall-clock time of whole script (seconds)",
        "Total wall-clock time of whole script (minutes)",
    ],
    "Value": [
        avg_epoch_seconds,
        avg_epoch_seconds / 60 if pd.notna(avg_epoch_seconds) else np.nan,
        avg_fit_seconds,
        avg_fit_seconds / 60 if pd.notna(avg_fit_seconds) else np.nan,
        total_training_seconds,
        total_training_seconds / 60 if pd.notna(total_training_seconds) else np.nan,
        total_training_seconds / 3600 if pd.notna(total_training_seconds) else np.nan,
        len(time_detail),
        df["Ticker"].nunique(),
        len(TEST_YEARS),
        total_seconds,
        total_seconds / 60,
    ]
})

hyperparameters = pd.DataFrame({
    "Hyperparameter": [
        "lags",
        "horizon_forward_days",
        "train_years",
        "validation_years",
        "test_years",
        "optimizer",
        "learning_rate",
        "max_epochs",
        "batch_size",
        "early_stopping_patience",
        "lstm_units_layer1",
        "lstm_units_layer2",
        "static_dense_units",
        "fusion_dense_units_1",
        "fusion_dense_units_2",
        "dropout_rate",
        "sequence_length",
        "num_static_features",
        "loss",
    ],
    "Value": [
        LAGS,
        H_FWD,
        TRAIN_YEARS,
        VAL_YEARS,
        ",".join(map(str, TEST_YEARS)),
        "Adam",
        LEARNING_RATE,
        MAX_EPOCHS,
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        SEQ_LSTM_UNITS_1,
        SEQ_LSTM_UNITS_2,
        STATIC_DENSE_UNITS,
        FUSION_DENSE_UNITS_1,
        FUSION_DENSE_UNITS_2,
        DROPOUT_RATE,
        LAGS,
        len(EXTRA_COLS),
        "mse",
    ]
})

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)

ax.plot(
    learning_curve_mean["Epoch"],
    learning_curve_mean["Train_Loss"],
    color="#8ECAE6",
    linewidth=2.2,
    label="Training Loss"
)

ax.plot(
    learning_curve_mean["Epoch"],
    learning_curve_mean["Val_Loss"],
    color="#F4A261",
    linewidth=2.2,
    label="Validation Loss"
)

ax.set_title("LSTM Learning Curve", fontsize=17, fontweight="bold", pad=14)
ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
ax.set_ylabel("Loss (MSE)", fontsize=13, fontweight="bold")

ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig(OUT_LC_PLOT, bbox_inches="tight")
plt.close()

print("Saved:", OUT_LC_PLOT)


pred_daily_save = pred_daily.copy()
pred_daily_save["Date"] = pd.to_datetime(pred_daily_save["Date"]).dt.strftime("%Y-%m-%d")

pred_eom_save = pred_eom.copy()
pred_eom_save["Date"] = pd.to_datetime(pred_eom_save["Date"]).dt.strftime("%Y-%m-%d")
pred_eom_save["RebalanceDate"] = pd.to_datetime(pred_eom_save["RebalanceDate"]).dt.strftime("%Y-%m-%d")

mu_hat_matrix_save = mu_hat_matrix.copy()
mu_hat_matrix_save.index = pd.to_datetime(mu_hat_matrix_save.index).strftime("%Y-%m-%d")
mu_hat_matrix_save = mu_hat_matrix_save.reset_index().rename(columns={"RebalanceDate": "Date"})

ret_true_matrix_save = ret_true_matrix.copy()
ret_true_matrix_save.index = pd.to_datetime(ret_true_matrix_save.index).strftime("%Y-%m-%d")
ret_true_matrix_save = ret_true_matrix_save.reset_index().rename(columns={"RebalanceDate": "Date"})

with pd.ExcelWriter(OUT_PRED, engine="openpyxl") as w:
    universe.to_excel(w, index=False, sheet_name="universe")
    pred_daily_save.to_excel(w, index=False, sheet_name="pred_daily")
    pred_eom_save.to_excel(w, index=False, sheet_name="pred_eom")
    mu_hat_matrix_save.to_excel(w, index=False, sheet_name="mu_hat_matrix")
    ret_true_matrix_save.to_excel(w, index=False, sheet_name="ret_true_matrix")

with pd.ExcelWriter(OUT_METRICS, engine="openpyxl") as w:
    metrics.to_excel(w, index=False, sheet_name="metrics")

with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as w:
    compute_time_summary.to_excel(w, index=False, sheet_name="compute_time_summary")
    time_detail.to_excel(w, index=False, sheet_name="compute_time_detail")
    hyperparameters.to_excel(w, index=False, sheet_name="hyperparameters")
    learning_curve_mean.to_excel(w, index=False, sheet_name="learning_curve_mean")
    learning_curve_detail.to_excel(w, index=False, sheet_name="learning_curve_detail")

print("Saved:", OUT_PRED)
print("Saved:", OUT_METRICS)
print("Saved:", OUT_REPORT)
print("Target:", TARGET, "| Horizon:", f"{H_FWD} trading days forward")
print("Rows(pred_daily):", len(pred_daily), "| Rows(pred_eom):", len(pred_eom), "| Rows(metrics):", len(metrics))
print("Features used:", len(FEATURES))