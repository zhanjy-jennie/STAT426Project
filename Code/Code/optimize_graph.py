import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import re


folder = Path.home() / "Desktop" / "Assets_Data"
INPUT_FILE = folder / "MAD_Optimization_AvgWeights.xlsx"   # 改成你的文件名


excel_file = pd.ExcelFile(INPUT_FILE)
sheet_names = excel_file.sheet_names

if len(sheet_names) == 0:
    raise ValueError("No sheets found in the workbook.")

summary_sheet = sheet_names[0]
weight_sheets = sheet_names[1:]


plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


text_color = "#2F2F2F"
grid_color = "#E8E8E8"

return_color = "#91C8E4"
vol_color = "#F6C89F"
sharpe_color = "#A8D5BA"

weight_color = "#A8DADC"
crypto_color = "#F4A261"
edge_color = "#7A7A7A"


def clean_name(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))


def add_bar_labels(ax, bars, fmt="{:.3f}", percent=False):
    values = [bar.get_height() for bar in bars]
    top = max(values) if len(values) > 0 else 1
    offset = top * 0.02 if top != 0 else 0.01

    for bar in bars:
        h = bar.get_height()
        label = f"{h:.1%}" if percent else fmt.format(h)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=text_color
        )


def draw_summary_chart():
    df = pd.read_excel(INPUT_FILE, sheet_name=summary_sheet)
    df.columns = df.columns.str.strip()

    needed = ["Portfolio", "AnnualReturn", "AnnualVol", "Sharpe"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in summary sheet: {missing}")

    df = df[needed].copy()

    charts = [
        ("AnnualReturn", "Annual Return", return_color, True, "Summary_Annual_Return.png"),
        ("AnnualVol", "Annual Volatility", vol_color, True, "Summary_Annual_Volatility.png"),
        ("Sharpe", "Sharpe Ratio", sharpe_color, False, "Summary_Sharpe_Ratio.png"),
    ]

    for col, title, color, is_percent, file_name in charts:
        fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)

        bars = ax.bar(
            df["Portfolio"],
            df[col],
            color=color,
            edgecolor=edge_color,
            linewidth=0.9,
            width=0.58,
            zorder=3
        )

        ax.set_title(title, fontsize=17, fontweight="bold", pad=12, color=text_color)
        ax.set_xlabel("Portfolio", fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(title, fontsize=12, fontweight="bold", color=text_color)

        ax.tick_params(axis="x", labelrotation=25, labelsize=10, colors=text_color)
        ax.tick_params(axis="y", labelsize=10, colors=text_color)

        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6, color=grid_color, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if is_percent:
            add_bar_labels(ax, bars, percent=True)
        else:
            add_bar_labels(ax, bars, fmt="{:.3f}", percent=False)

        plt.tight_layout()

        save_path = folder / file_name
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print("Saved:", save_path)


def draw_weight_chart(sheet_name):
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()

    needed = ["Ticker", "AvgWeight"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{sheet_name}': {missing}")

    df = df[needed].copy()
    df["AvgWeight"] = pd.to_numeric(df["AvgWeight"], errors="coerce")
    df = df.dropna(subset=["AvgWeight"])
    df = df.sort_values("AvgWeight", ascending=True).reset_index(drop=True)

    colors = []
    for tk in df["Ticker"]:
        if "-USD" in str(tk) or "BTC" in str(tk) or "ETH" in str(tk) or "XRP" in str(tk) or "LTC" in str(tk) or "BCH" in str(tk):
            colors.append(crypto_color)
        else:
            colors.append(weight_color)

    fig_height = max(5.5, 0.38 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height), dpi=300)

    bars = ax.barh(
        df["Ticker"],
        df["AvgWeight"],
        color=colors,
        edgecolor=edge_color,
        linewidth=0.8,
        zorder=3
    )

    ax.set_title(f"Average Portfolio Weights: {sheet_name}", fontsize=15, fontweight="bold", pad=12, color=text_color)
    ax.set_xlabel("Average Weight", fontsize=12, fontweight="bold", color=text_color)
    ax.set_ylabel("Ticker", fontsize=12, fontweight="bold", color=text_color)

    ax.tick_params(axis="x", labelsize=10, colors=text_color)
    ax.tick_params(axis="y", labelsize=10, colors=text_color)

    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.6, color=grid_color, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = df["AvgWeight"].max()
    ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 0.1)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color=text_color
        )

    plt.tight_layout()

    save_path = folder / f"Weights_{clean_name(sheet_name)}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


def draw_combined_weight_chart():
    if len(weight_sheets) == 0:
        return

    n = len(weight_sheets)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.2 * nrows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for ax, sheet_name in zip(axes, weight_sheets):
        df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        df = df[["Ticker", "AvgWeight"]].copy()
        df["AvgWeight"] = pd.to_numeric(df["AvgWeight"], errors="coerce")
        df = df.dropna(subset=["AvgWeight"])
        df = df.sort_values("AvgWeight", ascending=True).reset_index(drop=True)

        colors = []
        for tk in df["Ticker"]:
            if "-USD" in str(tk) or "BTC" in str(tk) or "ETH" in str(tk) or "XRP" in str(tk) or "LTC" in str(tk) or "BCH" in str(tk):
                colors.append(crypto_color)
            else:
                colors.append(weight_color)

        bars = ax.barh(
            df["Ticker"],
            df["AvgWeight"],
            color=colors,
            edgecolor=edge_color,
            linewidth=0.7,
            zorder=3
        )

        ax.set_title(sheet_name, fontsize=13, fontweight="bold", color=text_color, pad=8)
        ax.set_xlabel("Avg Weight", fontsize=11, fontweight="bold", color=text_color)
        ax.set_ylabel("Ticker", fontsize=11, fontweight="bold", color=text_color)

        ax.tick_params(axis="x", labelsize=9, colors=text_color)
        ax.tick_params(axis="y", labelsize=9, colors=text_color)

        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.55, color=grid_color, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        xmax = df["AvgWeight"].max()
        ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 0.1)

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                color=text_color
            )

    for ax in axes[len(weight_sheets):]:
        ax.axis("off")

    fig.suptitle("Average Portfolio Weights Across Portfolios", fontsize=18, fontweight="bold", y=1.01, color=text_color)
    plt.tight_layout()

    save_path = folder / "All_Portfolio_Weights_Combined.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


draw_summary_chart()

for sheet in weight_sheets:
    draw_weight_chart(sheet)

draw_combined_weight_chart()