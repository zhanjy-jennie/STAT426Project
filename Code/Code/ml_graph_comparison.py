import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


folder = Path(r"C:\Users\wuyue\Desktop\Assets_Data")

file_lstm = folder / "LSTM_Metrics_AnnualRolling.xlsx"
file_rf   = folder / "RF_Metrics_AnnualRolling.xlsx"
file_xgb  = folder / "XGB_Metrics_AnnualRolling.xlsx"


df_lstm = pd.read_excel(file_lstm)
df_rf   = pd.read_excel(file_rf)
df_xgb  = pd.read_excel(file_xgb)

df_lstm.columns = df_lstm.columns.str.strip()
df_rf.columns = df_rf.columns.str.strip()
df_xgb.columns = df_xgb.columns.str.strip()


summary = pd.DataFrame({
    "LSTM": [
        df_lstm["Val_MAE"].mean(),
        df_lstm["Test_MAE"].mean(),
        df_lstm["Val_RMSE"].mean(),
        df_lstm["Test_RMSE"].mean(),
        df_lstm["Val_DirAcc"].mean(),
        df_lstm["Test_DirAcc"].mean()
    ],
    "RF": [
        df_rf["Val_MAE"].mean(),
        df_rf["Test_MAE"].mean(),
        df_rf["Val_RMSE"].mean(),
        df_rf["Test_RMSE"].mean(),
        df_rf["Val_DirAcc"].mean(),
        df_rf["Test_DirAcc"].mean()
    ],
    "XGB": [
        df_xgb["Val_MAE"].mean(),
        df_xgb["Test_MAE"].mean(),
        df_xgb["Val_RMSE"].mean(),
        df_xgb["Test_RMSE"].mean(),
        df_xgb["Val_DirAcc"].mean(),
        df_xgb["Test_DirAcc"].mean()
    ]
}, index=[
    "Val_MAE", "Test_MAE",
    "Val_RMSE", "Test_RMSE",
    "Val_DirAcc", "Test_DirAcc"
]).T

print(summary.round(6))


metrics_table = summary.copy().round(4)
print("\nMetrics Table: Comprehensive comparison of all models")
print(metrics_table)


metrics_table.to_excel(folder / "Metrics_Table_Comprehensive.xlsx")
metrics_table.to_csv(folder / "Metrics_Table_Comprehensive.csv")


table_to_plot = metrics_table.reset_index().rename(columns={"index": "Model"})

fig, ax = plt.subplots(figsize=(11, 2.6), dpi=300)
ax.axis("off")

table = ax.table(
    cellText=table_to_plot.values,
    colLabels=table_to_plot.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

header_color = "#DCEEF2"
best_color = "#F7E7A9"
text_color = "#2F2F2F"
edge_color = "#BDBDBD"

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(edge_color)
    cell.set_linewidth(0.8)
    cell.get_text().set_color(text_color)

    if row == 0:
        cell.set_facecolor(header_color)
        cell.get_text().set_weight("bold")
    else:
        cell.set_facecolor("white")


metric_columns = ["Val_MAE", "Test_MAE", "Val_RMSE", "Test_RMSE", "Val_DirAcc", "Test_DirAcc"]

for j, col_name in enumerate(metric_columns, start=1):
    if col_name in ["Val_MAE", "Test_MAE", "Val_RMSE", "Test_RMSE"]:
        best_model = metrics_table[col_name].idxmin()
    else:
        best_model = metrics_table[col_name].idxmax()

    best_row = metrics_table.index.get_loc(best_model) + 1
    table[best_row, j].set_facecolor(best_color)


plt.tight_layout()
plt.savefig(folder / "Metrics_Table_Comprehensive.png", bbox_inches="tight")
plt.close()

print(f"Saved: {folder / 'Metrics_Table_Comprehensive.xlsx'}")
print(f"Saved: {folder / 'Metrics_Table_Comprehensive.csv'}")
print(f"Saved: {folder / 'Metrics_Table_Comprehensive.png'}")


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


models = ["LSTM", "RF", "XGB"]
x = np.arange(len(models))
width = 0.16

val_color = "#B7DDE8"
test_color = "#F4C7A1"
val_edge = "#7FA9B5"
test_edge = "#C79263"
text_color = "#2F2F2F"
grid_color = "#E6E6E6"


def draw_chart(val_col, test_col, ylabel, save_name):
    val_values = summary[val_col].values
    test_values = summary[test_col].values

    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=300)

    bars1 = ax.bar(
        x - width/2,
        val_values,
        width=width,
        color=val_color,
        edgecolor=val_edge,
        linewidth=1,
        label="Validation",
        zorder=3
    )

    bars2 = ax.bar(
        x + width/2,
        test_values,
        width=width,
        color=test_color,
        edgecolor=test_edge,
        linewidth=1,
        label="Test",
        zorder=3
    )

    ax.set_title(f"{ylabel} Comparison Across Models", fontsize=17, fontweight="bold", color=text_color, pad=14)
    ax.set_xlabel("Model", fontsize=13, fontweight="bold", color=text_color, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=text_color, labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold", color=text_color)
    ax.tick_params(axis="y", labelsize=11, colors=text_color)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=grid_color, alpha=0.8, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B5B5B5")
    ax.spines["bottom"].set_color("#B5B5B5")

    ax.legend(frameon=False, fontsize=11, loc="upper right")

    all_values = np.concatenate([val_values, test_values])
    y_min = all_values.min()
    y_max = all_values.max()

    if "DirAcc" in ylabel:
        lower = max(0, y_min - 0.04)
        upper = min(1.0, y_max + 0.06)
        ax.set_ylim(lower, upper)
        offset = (upper - lower) * 0.015
    else:
        ax.set_ylim(0, y_max * 1.22)
        offset = y_max * 0.025

    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + offset,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=text_color
        )

    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + offset,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=text_color
        )

    ax.margins(x=0.14)

    save_path = folder / save_name
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


draw_chart("Val_MAE", "Test_MAE", "MAE", "MAE_Comparison.png")
draw_chart("Val_RMSE", "Test_RMSE", "RMSE", "RMSE_Comparison.png")
draw_chart("Val_DirAcc", "Test_DirAcc", "Direction Accuracy", "DirAcc_Comparison.png")