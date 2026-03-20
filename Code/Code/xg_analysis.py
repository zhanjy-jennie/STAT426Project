import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


folder = Path(r"C:\Users\wuyue\Desktop\Assets_Data")
file_XGB = folder / "XGB_Metrics_AnnualRolling.xlsx"


df = pd.read_excel(file_XGB)
df.columns = df.columns.str.strip()


category_order = [
    "Cryptocurrencies",
    "TraditionalStocks",
    "TechStocks",
    "Forex",
    "Commodities"
]


ticker_avg = (
    df.groupby(["AssetType", "Ticker"])[
        ["Val_MAE", "Test_MAE", "Val_RMSE", "Test_RMSE", "Val_DirAcc", "Test_DirAcc"]
    ]
    .mean()
    .reset_index()
)


category_avg = (
    ticker_avg.groupby("AssetType")[
        ["Val_MAE", "Test_MAE", "Val_RMSE", "Test_RMSE", "Val_DirAcc", "Test_DirAcc"]
    ]
    .mean()
    .reindex(category_order)
)


print("Average performance of each category based on XGB:")
print(category_avg.round(6))


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


x = np.arange(len(category_avg.index))

val_color = "#8FD3E8"
test_color = "#F6BE8A"
val_edge = "#5FA8BF"
test_edge = "#D8904F"
text_color = "#2F2F2F"
grid_color = "#E8E8E8"


def draw_line_chart(val_col, test_col, ylabel, save_name, higher_is_better):
    val_values = category_avg[val_col].values
    test_values = category_avg[test_col].values

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(
        x, val_values,
        marker="o", markersize=8,
        linewidth=2.2,
        color=val_color,
        markeredgecolor=val_edge,
        markerfacecolor=val_color,
        label="Validation",
        zorder=3
    )

    ax.plot(
        x, test_values,
        marker="o", markersize=8,
        linewidth=2.2,
        color=test_color,
        markeredgecolor=test_edge,
        markerfacecolor=test_color,
        label="Test",
        zorder=3
    )

    ax.set_title(f"{ylabel} Across Asset Categories (XGB)", fontsize=17, fontweight="bold", color=text_color, pad=14)
    ax.set_xlabel("Asset Category", fontsize=13, fontweight="bold", color=text_color, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=text_color, labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(category_avg.index, fontsize=11, fontweight="bold", color=text_color)
    ax.tick_params(axis="y", labelsize=11, colors=text_color)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=grid_color, alpha=0.9, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    ax.legend(frameon=False, fontsize=11, loc="best")

    all_values = np.concatenate([val_values, test_values])
    y_min = all_values.min()
    y_max = all_values.max()

    if "DirAcc" in ylabel:
        lower = max(0, y_min - 0.03)
        upper = min(1.0, y_max + 0.04)
        ax.set_ylim(lower, upper)
        offset = (upper - lower) * 0.02
    else:
        lower = max(0, y_min * 0.9)
        upper = y_max * 1.12
        ax.set_ylim(lower, upper)
        offset = (upper - lower) * 0.02

    for i, y in enumerate(val_values):
        ax.text(x[i], y + offset, f"{y:.4f}", ha="center", va="bottom", fontsize=10, color=val_edge)

    for i, y in enumerate(test_values):
        ax.text(x[i], y + offset, f"{y:.4f}", ha="center", va="bottom", fontsize=10, color=test_edge)

    save_path = folder / save_name
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

    if higher_is_better:
        best_val = category_avg[val_col].idxmax()
        best_test = category_avg[test_col].idxmax()
    else:
        best_val = category_avg[val_col].idxmin()
        best_test = category_avg[test_col].idxmin()

    print(f"{ylabel} - Best Validation Category: {best_val}")
    print(f"{ylabel} - Best Test Category: {best_test}")
    print("-" * 50)


draw_line_chart("Val_MAE", "Test_MAE", "MAE", "XGB_Category_MAE_Line.png", higher_is_better=False)
draw_line_chart("Val_RMSE", "Test_RMSE", "RMSE", "XGB_Category_RMSE_Line.png", higher_is_better=False)
draw_line_chart("Val_DirAcc", "Test_DirAcc", "Direction Accuracy", "XGB_Category_DirAcc_Line.png", higher_is_better=True)