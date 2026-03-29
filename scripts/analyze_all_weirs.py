"""
All-Weir Bloom Analysis
========================
16보 전체 pre/post NDCI proxy 비교 분석 + 시각화.
extract_all_weirs.py 실행 완료 후 사용.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from scipy import stats

OUTPUT_DIR = Path("Path(__file__).parent.parent / "output"")
BLOOM_DIR = OUTPUT_DIR / "bloom_data"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# River grouping for visualization
RIVER_GROUPS = {
    "한강": ["이포보", "여주보", "강천보"],
    "낙동강": ["상주보", "낙단보", "구미보", "칠곡보", "강정고령보", "달성보", "합천창녕보", "창녕함안보"],
    "금강": ["세종보", "공주보", "백제보"],
    "영산강": ["승촌보", "죽산보"],
}

RIVER_COLORS = {
    "한강": "#2196F3",
    "낙동강": "#F44336",
    "금강": "#4CAF50",
    "영산강": "#FF9800",
}


def load_all_results():
    """Load all per-weir JSON files."""
    results = {}
    for f in sorted(BLOOM_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        results[data["weir"]] = data
    return results


def compute_summary(results):
    """Compute pre/post summary statistics for each weir."""
    rows = []
    for name, data in results.items():
        pre_medians = [r["median"] for r in data["landsat_pre_weir"] if r["median"] is not None]
        post_medians = [r["median"] for r in data["landsat_post_weir"] if r["median"] is not None]

        if not pre_medians or not post_medians:
            continue

        pre_mean = np.mean(pre_medians)
        post_mean = np.mean(post_medians)
        change = post_mean - pre_mean
        pct_change = change / pre_mean * 100

        # Welch t-test
        t_stat, p_val = stats.ttest_ind(pre_medians, post_medians, equal_var=False)

        # Identify river
        river = "Unknown"
        for r, weirs in RIVER_GROUPS.items():
            if name in weirs:
                river = r
                break

        rows.append({
            "weir": name,
            "river": river,
            "lat": data["lat"],
            "lon": data["lon"],
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "change": change,
            "pct_change": pct_change,
            "t_stat": t_stat,
            "p_value": p_val,
            "n_pre": len(pre_medians),
            "n_post": len(post_medians),
        })

    return pd.DataFrame(rows)


def plot_bar_chart(df):
    """Bar chart: pre/post NDCI for all weirs grouped by river."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Sort by river then by change magnitude
    order = []
    for river in ["한강", "낙동강", "금강", "영산강"]:
        river_weirs = df[df["river"] == river].sort_values("change", ascending=False)
        order.extend(river_weirs["weir"].tolist())

    df_sorted = df.set_index("weir").loc[order].reset_index()
    x = np.arange(len(df_sorted))
    width = 0.35

    bars_pre = ax.bar(x - width/2, df_sorted["pre_mean"], width,
                       label="Pre-weir (2000-2012)", color="#90CAF9", edgecolor="gray")
    bars_post = ax.bar(x + width/2, df_sorted["post_mean"], width,
                        label="Post-weir (2013-2024)", color="#EF9A9A", edgecolor="gray")

    # Add significance markers
    for i, row in df_sorted.iterrows():
        if row["p_value"] < 0.001:
            marker = "***"
        elif row["p_value"] < 0.01:
            marker = "**"
        elif row["p_value"] < 0.05:
            marker = "*"
        else:
            marker = "ns"
        y_max = max(row["pre_mean"], row["post_mean"]) + 0.01
        ax.text(i, y_max, f"{row['pct_change']:+.1f}%\n{marker}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    # River group separators
    river_positions = {}
    pos = 0
    for river in ["한강", "낙동강", "금강", "영산강"]:
        river_df = df_sorted[df_sorted["river"] == river]
        if not river_df.empty:
            start = pos
            end = pos + len(river_df) - 1
            river_positions[river] = (start, end)
            if pos > 0:
                ax.axvline(x=pos - 0.5, color="gray", linestyle=":", alpha=0.5)
            pos += len(river_df)

    # River labels at bottom
    for river, (start, end) in river_positions.items():
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[0] - 0.02, river,
                ha="center", va="top", fontsize=10, fontweight="bold",
                color=RIVER_COLORS.get(river, "black"))

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["weir"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Bloom Season Median NDCI Proxy", fontsize=11)
    ax.set_title("Pre vs Post-Weir Bloom Proxy (NDCI) — 16 Weirs", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "all_weirs_pre_post_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_change_map(df):
    """Simple scatter plot showing geographic distribution of change."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for river, color in RIVER_COLORS.items():
        mask = df["river"] == river
        scatter = ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"],
                            c=df.loc[mask, "pct_change"],
                            s=np.abs(df.loc[mask, "pct_change"]) * 15 + 50,
                            cmap="RdYlGn_r", vmin=0, vmax=25,
                            edgecolors=color, linewidths=2,
                            label=river, alpha=0.8, zorder=5)

    # Label weirs
    for _, row in df.iterrows():
        ax.annotate(row["weir"], (row["lon"], row["lat"]),
                   fontsize=6, ha="left", va="bottom",
                   xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Geographic Distribution of Bloom Change (%)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("NDCI Change (%)", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "all_weirs_change_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_timeseries_grid(results, df):
    """4x4 grid of time series for all 16 weirs."""
    weir_order = []
    for river in ["한강", "낙동강", "금강", "영산강"]:
        for w in RIVER_GROUPS[river]:
            if w in results:
                weir_order.append(w)

    n = len(weir_order)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5), squeeze=False)

    for idx, name in enumerate(weir_order):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        data = results[name]

        pre_years = [e["year"] for e in data["landsat_pre_weir"]]
        pre_vals = [e["median"] for e in data["landsat_pre_weir"]]
        post_years = [e["year"] for e in data["landsat_post_weir"]]
        post_vals = [e["median"] for e in data["landsat_post_weir"]]

        ax.plot(pre_years, pre_vals, "o-", color="#2196F3", markersize=3, linewidth=1)
        ax.plot(post_years, post_vals, "s-", color="#F44336", markersize=3, linewidth=1)
        ax.axvline(x=2012, color="gray", linestyle="--", alpha=0.5)

        # Get change info
        row = df[df["weir"] == name]
        if not row.empty:
            pct = row.iloc[0]["pct_change"]
            pval = row.iloc[0]["p_value"]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"{name} ({pct:+.1f}% {sig})", fontsize=9, fontweight="bold")
        else:
            ax.set_title(name, fontsize=9)

        ax.set_xlim(1999, 2025)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide empty axes
    for idx in range(len(weir_order), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Bloom Proxy (NDCI) Time Series — All 16 Weirs",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "all_weirs_timeseries_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    print("Loading bloom data...")
    results = load_all_results()
    print(f"Loaded {len(results)} weirs\n")

    if len(results) == 0:
        print("No data found. Run extract_all_weirs.py first.")
        return

    # Compute summary
    df = compute_summary(results)
    df = df.sort_values("pct_change", ascending=False)

    # Print summary table
    print("=" * 80)
    print("PRE vs POST-WEIR BLOOM PROXY COMPARISON")
    print("=" * 80)
    print(f"{'Weir':<12} {'River':<6} {'Pre':>8} {'Post':>8} {'Change':>8} {'%':>7} "
          f"{'t-stat':>7} {'p-value':>10} {'Sig':>4}")
    print("-" * 80)

    for _, row in df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 \
              else "*" if row["p_value"] < 0.05 else "ns"
        print(f"{row['weir']:<12} {row['river']:<6} {row['pre_mean']:>8.4f} "
              f"{row['post_mean']:>8.4f} {row['change']:>+8.4f} {row['pct_change']:>+6.1f}% "
              f"{row['t_stat']:>7.2f} {row['p_value']:>10.6f} {sig:>4}")

    print("-" * 80)
    print(f"Mean change: {df['pct_change'].mean():+.1f}%")
    print(f"All significant (p<0.05): {(df['p_value'] < 0.05).all()}")
    print(f"All significant (p<0.01): {(df['p_value'] < 0.01).all()}")
    print("=" * 80)

    # Save summary CSV
    summary_path = OUTPUT_DIR / "weir_bloom_summary_stats.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved: {summary_path}")

    # Plots
    print("\nGenerating figures...")
    plot_bar_chart(df)
    plot_timeseries_grid(results, df)
    plot_change_map(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
