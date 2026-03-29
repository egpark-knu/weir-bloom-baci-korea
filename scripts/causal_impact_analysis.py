"""
BSTS CausalImpact Analysis (Robustness Check)
==============================================
Uses Google's Bayesian Structural Time-Series (BSTS) method
(Brodersen et al. 2015, Annals of Applied Statistics) to estimate
the causal effect of weir construction on bloom proxy.

This provides an alternative identification strategy to the BACI
event-study, using upstream control reaches as covariates in the
synthetic counterfactual.

Precedent: Lee et al. (2024 STOTEN) applied BSTS CausalImpact
to Korean weir effects on water quality.

Outputs:
  - figures/causal_impact_pooled.png
  - figures/causal_impact_by_river.png
  - figures/causal_impact_individual.png (grid of 16 weirs)
  - output/causal_impact_summary.csv
  - output/causal_impact_report.txt
"""

from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from causalimpact import CausalImpact

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
USE_WATER_MASKED = True
if USE_WATER_MASKED:
    BLOOM_DIR = OUTPUT_DIR / "water_masked" / "weirs"
    CONTROL_DIR = OUTPUT_DIR / "water_masked" / "controls"
else:
    BLOOM_DIR = OUTPUT_DIR / "bloom_data"
    CONTROL_DIR = OUTPUT_DIR / "control_data"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RIVER_GROUPS = {
    "Han": ["이포보", "여주보", "강천보"],
    "Nakdong": ["상주보", "낙단보", "구미보", "칠곡보", "강정고령보",
                "달성보", "합천창녕보", "창녕함안보"],
    "Geum": ["세종보", "공주보", "백제보"],
    "Yeongsan": ["승촌보", "죽산보"],
}

RIVER_COLORS = {
    "Han": "#2196F3",
    "Nakdong": "#F44336",
    "Geum": "#4CAF50",
    "Yeongsan": "#FF9800",
}

WEIR_EN = {
    "이포보": "Ipo", "여주보": "Yeoju", "강천보": "Gangcheon",
    "상주보": "Sangju", "낙단보": "Nakdan", "구미보": "Gumi",
    "칠곡보": "Chilgok", "강정고령보": "Gangjeong", "달성보": "Dalseong",
    "합천창녕보": "Hapcheon", "창녕함안보": "Changnyeong",
    "세종보": "Sejong", "공주보": "Gongju", "백제보": "Baekje",
    "승촌보": "Seungchon", "죽산보": "Juksan",
}

WEIR_CONTROL_MAP = {
    "이포보": "Ipo_upstream", "여주보": "Yeoju_upstream",
    "강천보": "Gangcheon_upstream", "상주보": "Sangju_upstream",
    "낙단보": "Nakdan_upstream", "구미보": "Gumi_upstream",
    "칠곡보": "Chilgok_upstream", "강정고령보": "Gangjeong_upstream",
    "달성보": "Dalseong_upstream", "합천창녕보": "Hapcheon_upstream",
    "창녕함안보": "Changnyeong_upstream", "세종보": "Sejong_upstream",
    "공주보": "Gongju_upstream", "백제보": "Baekje_upstream",
    "승촌보": "Seungchon_upstream", "죽산보": "Juksan_upstream",
}

# Intervention year: 2012 = last construction year
# Pre-period: 2000-2011, Post-period: 2013-2025
INTERVENTION_YEAR = 2012
PRE_START = 2000
POST_END = 2025


# ── Data Loading ──

def load_weir_series(weir_name_kr: str) -> dict[int, float]:
    """Load annual bloom proxy (median) for a weir site."""
    data_files = list(BLOOM_DIR.glob("*.json"))
    for fp in data_files:
        with open(fp) as f:
            d = json.load(f)
        if d.get("weir") == weir_name_kr or d.get("weir_name_kr") == weir_name_kr:
            series = {}
            for entry in d.get("landsat_pre_weir", []):
                if entry.get("median") is not None:
                    series[entry["year"]] = entry["median"]
            for entry in d.get("landsat_post_weir", []):
                if entry.get("median") is not None:
                    series[entry["year"]] = entry["median"]
            return series
    return {}


def load_control_series(control_name: str) -> dict[int, float]:
    """Load annual bloom proxy (median) for an upstream control reach."""
    fp = CONTROL_DIR / f"{control_name}.json"
    if not fp.exists():
        return {}
    with open(fp) as f:
        d = json.load(f)
    series = {}
    for entry in d.get("landsat_pre_weir", []):
        if entry.get("median") is not None:
            series[entry["year"]] = entry["median"]
    for entry in d.get("landsat_post_weir", []):
        if entry.get("median") is not None:
            series[entry["year"]] = entry["median"]
    return series


def build_paired_df(weir_kr: str) -> pd.DataFrame | None:
    """Build DataFrame with treatment (weir) and control (upstream) series."""
    control_name = WEIR_CONTROL_MAP.get(weir_kr)
    if not control_name:
        return None

    treatment = load_weir_series(weir_kr)
    control = load_control_series(control_name)

    if not treatment or not control:
        return None

    years = sorted(set(treatment.keys()) & set(control.keys()))
    if len(years) < 8:  # need minimum data
        return None

    df = pd.DataFrame({
        "year": years,
        "treatment": [treatment[y] for y in years],
        "control": [control[y] for y in years],
    }).set_index("year")

    return df


# ── CausalImpact Analysis ──

def run_causal_impact(df: pd.DataFrame) -> dict | None:
    """Run CausalImpact on a paired treatment-control DataFrame."""
    # Filter to years with data
    df = df.dropna()

    pre_years = [y for y in df.index if y < INTERVENTION_YEAR]
    post_years = [y for y in df.index if y > INTERVENTION_YEAR]

    if len(pre_years) < 5 or len(post_years) < 3:
        return None

    pre_period = [min(pre_years), max(pre_years)]
    post_period = [min(post_years), max(post_years)]

    # CausalImpact expects: col 0 = response (treatment), col 1+ = covariates (control)
    ci_data = df[["treatment", "control"]]

    try:
        ci = CausalImpact(ci_data, pre_period, post_period)
        sd = ci.summary_data  # rows: actual, predicted, abs_effect, etc. cols: average, cumulative

        result = {
            "pre_period": pre_period,
            "post_period": post_period,
            "avg_actual": float(sd.loc["actual", "average"]),
            "avg_predicted": float(sd.loc["predicted", "average"]),
            "avg_effect": float(sd.loc["abs_effect", "average"]),
            "avg_effect_pct": float(sd.loc["rel_effect", "average"]) * 100,
            "avg_effect_lower": float(sd.loc["abs_effect_lower", "average"]),
            "avg_effect_upper": float(sd.loc["abs_effect_upper", "average"]),
            "p_value": float(ci.p_value),
            "cumulative_effect": float(sd.loc["abs_effect", "cumulative"]),
            "ci_object": ci,
        }
        return result
    except Exception as e:
        print(f"  CausalImpact error: {e}")
        return None


# ── Plotting ──

def plot_pooled_causal_impact(results: dict[str, dict]):
    """Plot pooled CausalImpact results across all weirs."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Panel A: Effect sizes with CIs
    ax = axes[0]
    names = []
    effects = []
    lowers = []
    uppers = []
    colors = []

    for river, weir_list in RIVER_GROUPS.items():
        for w_kr in weir_list:
            if w_kr in results and results[w_kr] is not None:
                en = WEIR_EN.get(w_kr, w_kr)
                names.append(en)
                r = results[w_kr]
                effects.append(r["avg_effect_pct"])
                lowers.append(r["avg_effect_pct"] -
                              r["avg_effect_lower"] / r["avg_predicted"] * 100
                              if r["avg_predicted"] != 0 else 0)
                uppers.append(r["avg_effect_upper"] / r["avg_predicted"] * 100 -
                              r["avg_effect_pct"]
                              if r["avg_predicted"] != 0 else 0)
                colors.append(RIVER_COLORS[river])

    y_pos = np.arange(len(names))
    ax.barh(y_pos, effects, color=colors, alpha=0.7, edgecolor="white")
    # Ensure non-negative xerr values
    lowers = [max(0, l) for l in lowers]
    uppers = [max(0, u) for u in uppers]
    ax.errorbar(effects, y_pos, xerr=[lowers, uppers], fmt="none",
                ecolor="black", capsize=3, linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Causal Effect on Bloom Proxy (%)", fontsize=11)
    ax.set_title("(a) BSTS CausalImpact: Per-Weir Causal Effect Estimates", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Panel B: P-value distribution
    ax2 = axes[1]
    p_vals = [results[w_kr]["p_value"] for river, wl in RIVER_GROUPS.items()
              for w_kr in wl if w_kr in results and results[w_kr] is not None]
    ax2.hist(p_vals, bins=20, range=(0, 1), color="#607D8B", alpha=0.7, edgecolor="white")
    ax2.axvline(0.05, color="red", linestyle="--", linewidth=1.5, label="α = 0.05")
    ax2.set_xlabel("Bayesian p-value", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("(b) Distribution of Bayesian Tail-Area Probabilities", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    sig_count = sum(1 for p in p_vals if p < 0.05)
    ax2.text(0.95, 0.85, f"{sig_count}/{len(p_vals)} significant\n(p < 0.05)",
             transform=ax2.transAxes, ha="right", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    outpath = FIGURES_DIR / "causal_impact_pooled.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_individual_causal_impact(results: dict[str, dict]):
    """4x4 grid of individual CausalImpact plots."""
    all_weirs = [w for wl in RIVER_GROUPS.values() for w in wl]
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    for idx, w_kr in enumerate(all_weirs):
        row, col = divmod(idx, 4)
        ax = axes[row][col]
        en = WEIR_EN.get(w_kr, w_kr)

        if w_kr not in results or results[w_kr] is None:
            ax.text(0.5, 0.5, f"{en}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        r = results[w_kr]
        ci = r["ci_object"]

        # Plot observed vs predicted
        inferences = ci.inferences
        # Get original treatment data for observed line
        df_pair = build_paired_df(w_kr)
        if df_pair is not None:
            ax.plot(df_pair.index, df_pair["treatment"],
                    "k-", linewidth=1.5, label="Observed")
        ax.plot(inferences.index, inferences["preds"],
                "b--", linewidth=1, label="Predicted")
        ax.fill_between(inferences.index,
                        inferences["preds_lower"],
                        inferences["preds_upper"],
                        alpha=0.2, color="blue")
        ax.axvline(INTERVENTION_YEAR, color="red", linewidth=1, linestyle=":")

        # River color for title
        for river, wl in RIVER_GROUPS.items():
            if w_kr in wl:
                title_color = RIVER_COLORS[river]
                break

        pval_str = f"p={r['p_value']:.3f}" if r["p_value"] >= 0.001 else "p<0.001"
        ax.set_title(f"{en} ({r['avg_effect_pct']:+.1f}%, {pval_str})",
                     fontsize=9, fontweight="bold", color=title_color)
        ax.set_ylabel("NDVI", fontsize=8)

        if row == 3:
            ax.set_xlabel("Year", fontsize=8)
        ax.tick_params(labelsize=7)

    # Remove unused subplots
    for idx in range(len(all_weirs), 16):
        row, col = divmod(idx, 4)
        axes[row][col].set_visible(False)

    fig.suptitle("BSTS CausalImpact: Observed vs Counterfactual (All 16 Weirs)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    outpath = FIGURES_DIR / "causal_impact_individual.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_by_river(results: dict[str, dict]):
    """River-level average causal effects."""
    fig, ax = plt.subplots(figsize=(8, 5))

    river_effects = {}
    for river, weir_list in RIVER_GROUPS.items():
        effs = [results[w]["avg_effect_pct"] for w in weir_list
                if w in results and results[w] is not None]
        if effs:
            river_effects[river] = {
                "mean": np.mean(effs),
                "std": np.std(effs) if len(effs) > 1 else 0,
                "n": len(effs),
            }

    rivers = list(river_effects.keys())
    means = [river_effects[r]["mean"] for r in rivers]
    stds = [river_effects[r]["std"] for r in rivers]
    colors = [RIVER_COLORS[r] for r in rivers]
    ns = [river_effects[r]["n"] for r in rivers]

    bars = ax.bar(rivers, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.7, edgecolor="white", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Causal Effect on Bloom Proxy (%)", fontsize=11)
    ax.set_title("BSTS CausalImpact: River-Level Average Causal Effects",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "causal_impact_by_river.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ── Report ──

def write_report(results: dict[str, dict]):
    """Write summary CSV and text report."""
    rows = []
    for river, weir_list in RIVER_GROUPS.items():
        for w_kr in weir_list:
            en = WEIR_EN.get(w_kr, w_kr)
            if w_kr in results and results[w_kr] is not None:
                r = results[w_kr]
                rows.append({
                    "weir": en,
                    "river": river,
                    "avg_effect_pct": round(r["avg_effect_pct"], 2),
                    "avg_effect_lower_pct": round(
                        r["avg_effect_lower"] / r["avg_predicted"] * 100
                        if r["avg_predicted"] != 0 else 0, 2),
                    "avg_effect_upper_pct": round(
                        r["avg_effect_upper"] / r["avg_predicted"] * 100
                        if r["avg_predicted"] != 0 else 0, 2),
                    "p_value": round(r["p_value"], 4),
                    "significant": r["p_value"] < 0.05,
                })
            else:
                rows.append({
                    "weir": en, "river": river,
                    "avg_effect_pct": None, "avg_effect_lower_pct": None,
                    "avg_effect_upper_pct": None, "p_value": None,
                    "significant": None,
                })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "causal_impact_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Text report
    valid = df.dropna(subset=["avg_effect_pct"])
    sig = valid[valid["significant"] == True]

    report = []
    report.append("=" * 60)
    report.append("BSTS CausalImpact Analysis Report")
    report.append("=" * 60)
    report.append(f"\nWeirs analyzed: {len(valid)}/{len(df)}")
    report.append(f"Significant (p<0.05): {len(sig)}/{len(valid)}")
    report.append(f"\nMean causal effect: {valid['avg_effect_pct'].mean():.2f}%")
    report.append(f"Median causal effect: {valid['avg_effect_pct'].median():.2f}%")
    report.append(f"Range: {valid['avg_effect_pct'].min():.2f}% to {valid['avg_effect_pct'].max():.2f}%")

    report.append(f"\nMean p-value: {valid['p_value'].mean():.4f}")
    report.append(f"Median p-value: {valid['p_value'].median():.4f}")

    report.append("\n--- Per-River Summary ---")
    for river in RIVER_GROUPS:
        rv = valid[valid["river"] == river]
        if len(rv) > 0:
            report.append(f"  {river}: mean effect = {rv['avg_effect_pct'].mean():.2f}%, "
                          f"n_sig = {rv['significant'].sum()}/{len(rv)}")

    report.append("\n--- Per-Weir Details ---")
    for _, row in valid.iterrows():
        sig_mark = "***" if row["p_value"] < 0.001 else \
                   "**" if row["p_value"] < 0.01 else \
                   "*" if row["p_value"] < 0.05 else ""
        report.append(f"  {row['weir']:15s} ({row['river']:8s}): "
                      f"{row['avg_effect_pct']:+6.2f}% "
                      f"[{row['avg_effect_lower_pct']:.2f}, {row['avg_effect_upper_pct']:.2f}] "
                      f"p={row['p_value']:.4f} {sig_mark}")

    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "causal_impact_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved: {report_path}")
    print(report_text)


# ── Main ──

def main():
    print("=" * 60)
    print("BSTS CausalImpact Analysis")
    print("=" * 60)

    # Check control data availability
    if not CONTROL_DIR.exists():
        print(f"ERROR: Control data directory not found: {CONTROL_DIR}")
        print("Run extract_control_reaches.py first.")
        return

    available = list(CONTROL_DIR.glob("*.json"))
    print(f"Control reach data files: {len(available)}")

    results = {}
    all_weirs = [w for wl in RIVER_GROUPS.values() for w in wl]

    for w_kr in all_weirs:
        en = WEIR_EN.get(w_kr, w_kr)
        ctrl = WEIR_CONTROL_MAP.get(w_kr)
        print(f"\n  {en} (control: {ctrl})")

        df = build_paired_df(w_kr)
        if df is None:
            print(f"    Skipped (insufficient data)")
            results[w_kr] = None
            continue

        print(f"    Data: {len(df)} years, pre={sum(df.index < INTERVENTION_YEAR)}, "
              f"post={sum(df.index > INTERVENTION_YEAR)}")

        r = run_causal_impact(df)
        results[w_kr] = r
        if r:
            sig = "***" if r["p_value"] < 0.001 else \
                  "**" if r["p_value"] < 0.01 else \
                  "*" if r["p_value"] < 0.05 else ""
            print(f"    Effect: {r['avg_effect_pct']:+.2f}%, p={r['p_value']:.4f} {sig}")

    # Generate outputs
    valid_count = sum(1 for v in results.values() if v is not None)
    print(f"\n\nValid results: {valid_count}/{len(all_weirs)}")

    if valid_count > 0:
        plot_pooled_causal_impact(results)
        plot_individual_causal_impact(results)
        plot_by_river(results)
        write_report(results)
    else:
        print("No valid results to plot.")


if __name__ == "__main__":
    main()
