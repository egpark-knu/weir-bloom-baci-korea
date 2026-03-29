"""
BACI (Before-After-Control-Impact) Analysis for HAB Paper
==========================================================
Compares weir sites (treatment) vs upstream control reaches (control)
to estimate the causal effect of weir construction on bloom intensity.

Requires:
  - output/bloom_data/*.json        (treatment: weir sites)
  - output/control_data/*.json      (control: upstream reaches)

Outputs:
  - figures/baci_event_study.png    (Fig. 3: BACI event-study)
  - figures/baci_by_river.png       (Fig. 4: BACI by river system)
  - figures/parallel_trends.png     (Fig. 13: pre-period parallel trends)
  - figures/baci_treatment_control_ts.png (treatment vs control time series)
  - output/baci_did_estimates.csv
  - output/baci_analysis_report.txt
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import defaultdict

# ─── Paths ───
OUTPUT_DIR = Path("Path(__file__).parent.parent / "output"")
# Default: water-masked data (preferred). Fall back to unmasked.
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
    "Nakdong": ["상주보", "낙단보", "구미보", "칠곡보", "강정고령보", "달성보", "합천창녕보", "창녕함안보"],
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
    "이포보": "Ipo", "여주��": "Yeoju", "강천보": "Gangcheon",
    "상주보": "Sangju", "낙단보": "Nakdan", "구미보": "Gumi",
    "칠곡보": "Chilgok", "강정고령보": "Gangjeong", "달성보": "Dalseong",
    "��천창녕보": "Hapcheon", "창녕함안보": "Changnyeong",
    "세종보": "Sejong", "공주보": "Gongju", "백제보": "Baekje",
    "승촌보": "Seungchon", "죽��보": "Juksan",
}

COMPLETION_YEAR = 2012


def get_river(weir_name):
    for r, weirs in RIVER_GROUPS.items():
        if weir_name in weirs:
            return r
    return "Unknown"


def load_treatment_data():
    """Load weir site (treatment) bloom data."""
    results = {}
    for f in sorted(BLOOM_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        # Handle both old format ("weir") and new format ("weir_name_kr")
        key = data.get("weir") or data.get("weir_name_kr") or data.get("name", "")
        if key:
            results[key] = data
    return results


def load_control_data():
    """Load upstream control reach bloom data."""
    results = {}
    for f in sorted(CONTROL_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        key = data.get("weir_name_kr", "")
        if key:
            results[key] = data
    return results


def extract_yearly_series(data, key_pre="landsat_pre_weir", key_post="landsat_post_weir"):
    """Extract year -> median dict from a treatment or control JSON."""
    yearly = {}
    for rec in data.get(key_pre, []):
        if rec["median"] is not None:
            yearly[rec["year"]] = rec["median"]
    for rec in data.get(key_post, []):
        if rec["median"] is not None:
            yearly[rec["year"]] = rec["median"]
    return yearly


# ═══════════════════════════════════════════════
# 1. BACI EVENT-STUDY
# ═══════════════════════════════════════════════

def baci_event_study(treatment, control):
    """
    Compute BACI event-study coefficients.

    For each weir-control pair:
      BACI_t = (Treatment_t - Treatment_ref) - (Control_t - Control_ref)

    This differences out common trends (climate, region-wide eutrophication).
    """
    all_years = range(2000, 2026)
    baci_by_year = defaultdict(list)

    for weir_name, t_data in treatment.items():
        if weir_name not in control:
            continue

        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        # Reference: 2011 (last full pre-weir year)
        t_ref = t_series.get(2011)
        c_ref = c_series.get(2011)

        if t_ref is None or c_ref is None:
            # Fallback to pre-period mean
            t_pre = [v for y, v in t_series.items() if y <= 2011]
            c_pre = [v for y, v in c_series.items() if y <= 2011]
            t_ref = np.mean(t_pre) if t_pre else None
            c_ref = np.mean(c_pre) if c_pre else None

        if t_ref is None or c_ref is None:
            continue

        for yr in all_years:
            if yr in t_series and yr in c_series:
                baci_coeff = (t_series[yr] - t_ref) - (c_series[yr] - c_ref)
                baci_by_year[yr].append(baci_coeff)

    # Average BACI coefficients
    rows = []
    for yr in sorted(baci_by_year.keys()):
        vals = baci_by_year[yr]
        mean_coeff = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        rows.append({
            "year": yr,
            "relative_year": yr - COMPLETION_YEAR,
            "baci_coeff": mean_coeff,
            "se": se,
            "ci95_lo": mean_coeff - 1.96 * se,
            "ci95_hi": mean_coeff + 1.96 * se,
            "n_pairs": len(vals),
        })

    return pd.DataFrame(rows)


def baci_by_river(treatment, control):
    """BACI event-study per river system."""
    river_baci = {}
    for river, weir_list in RIVER_GROUPS.items():
        t_sub = {k: v for k, v in treatment.items() if k in weir_list}
        c_sub = {k: v for k, v in control.items() if k in weir_list}
        if t_sub and c_sub:
            df = baci_event_study(t_sub, c_sub)
            if not df.empty:
                river_baci[river] = df
    return river_baci


# ═══���═══════════════════════════════════════════
# 2. PARALLEL TRENDS TEST
# ═══════════════════════════════════════════════

def parallel_trends_test(treatment, control):
    """
    Test whether treatment and control had parallel trends pre-weir.

    For each pair, compute (treatment - control) annually in pre-period.
    If trends are parallel, the difference should have slope ≈ 0.
    """
    diff_slopes = []

    for weir_name, t_data in treatment.items():
        if weir_name not in control:
            continue

        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        # Pre-period differences
        pre_diffs = []
        for yr in range(2000, 2012):
            if yr in t_series and yr in c_series:
                pre_diffs.append((yr, t_series[yr] - c_series[yr]))

        if len(pre_diffs) < 5:
            continue

        years = np.array([d[0] for d in pre_diffs])
        diffs = np.array([d[1] for d in pre_diffs])

        slope, intercept, r_val, p_val, std_err = stats.linregress(years, diffs)
        diff_slopes.append({
            "weir": weir_name,
            "river": get_river(weir_name),
            "slope_per_year": slope,
            "r_value": r_val,
            "p_value": p_val,
            "n_years": len(pre_diffs),
        })

    df = pd.DataFrame(diff_slopes)
    if df.empty:
        return df, None, None

    # Test H0: mean slope = 0
    t_stat, p_val = stats.ttest_1samp(df["slope_per_year"], 0)
    return df, t_stat, p_val


# ═══════════════════════════════════════════════
# 3. BACI DID ESTIMATES
# ══════��═══════════════════════════════���════════

def baci_did_estimates(treatment, control):
    """
    Compute BACI DID estimates per weir and per river.

    DID = (Treatment_post - Treatment_pre) - (Control_post - Control_pre)
    """
    rows = []

    for weir_name, t_data in treatment.items():
        if weir_name not in control:
            continue

        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        t_pre = [v for y, v in t_series.items() if y <= 2011]
        t_post = [v for y, v in t_series.items() if y >= 2013]
        c_pre = [v for y, v in c_series.items() if y <= 2011]
        c_post = [v for y, v in c_series.items() if y >= 2013]

        if len(t_pre) < 3 or len(t_post) < 3 or len(c_pre) < 3 or len(c_post) < 3:
            continue

        t_pre_mean, t_post_mean = np.mean(t_pre), np.mean(t_post)
        c_pre_mean, c_post_mean = np.mean(c_pre), np.mean(c_post)

        # DID estimate
        did = (t_post_mean - t_pre_mean) - (c_post_mean - c_pre_mean)

        # Significance: Welch t-test on treatment-control differences
        pre_diffs = []
        post_diffs = []
        for yr in range(2000, 2012):
            if yr in t_series and yr in c_series:
                pre_diffs.append(t_series[yr] - c_series[yr])
        for yr in range(2013, 2026):
            if yr in t_series and yr in c_series:
                post_diffs.append(t_series[yr] - c_series[yr])

        if len(pre_diffs) >= 3 and len(post_diffs) >= 3:
            t_stat, p_val = stats.ttest_ind(post_diffs, pre_diffs, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan

        rows.append({
            "weir": weir_name,
            "weir_en": WEIR_EN.get(weir_name, weir_name),
            "river": get_river(weir_name),
            "treatment_pre": t_pre_mean,
            "treatment_post": t_post_mean,
            "control_pre": c_pre_mean,
            "control_post": c_post_mean,
            "did_estimate": did,
            "did_pct": did / t_pre_mean * 100 if t_pre_mean != 0 else np.nan,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05 if not np.isnan(p_val) else False,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════
# 4. PLOTTING
# ════════════════════��══════════════════════════

def plot_baci_event_study(df, outpath):
    """BACI event-study figure."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axvspan(df["relative_year"].min() - 0.5, -0.5, alpha=0.05, color="blue", label="Pre-weir")
    ax.axvspan(0.5, df["relative_year"].max() + 0.5, alpha=0.05, color="red", label="Post-weir")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Weir completion (2012)")

    ax.fill_between(df["relative_year"], df["ci95_lo"], df["ci95_hi"],
                     alpha=0.2, color="#333333")
    ax.plot(df["relative_year"], df["baci_coeff"], "o-", color="#333333",
            markersize=5, linewidth=1.5, zorder=5)

    ax.set_xlabel("Years Relative to Weir Completion", fontsize=11)
    ax.set_ylabel("BACI Coefficient (Treatment - Control Deviation)", fontsize=11)
    ax.set_title("BACI Event-Study: Bloom Proxy Change at Weir Sites vs Upstream Controls",
                fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df["relative_year"].min() - 0.5, df["relative_year"].max() + 0.5)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def plot_baci_by_river(river_baci, outpath):
    """BACI event-study by river system."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvspan(-12.5, -0.5, alpha=0.03, color="blue")
    ax.axvspan(0.5, 13.5, alpha=0.03, color="red")

    for river, df in river_baci.items():
        color = RIVER_COLORS[river]
        ax.fill_between(df["relative_year"], df["ci95_lo"], df["ci95_hi"],
                         alpha=0.1, color=color)
        ax.plot(df["relative_year"], df["baci_coeff"], "o-", color=color,
                markersize=4, linewidth=1.5, label=f"{river} (n={len(RIVER_GROUPS[river])})")

    ax.set_xlabel("Years Relative to Weir Completion", fontsize=11)
    ax.set_ylabel("BACI Coefficient", fontsize=11)
    ax.set_title("BACI Event-Study by River System", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def plot_parallel_trends(pt_df, t_stat, p_val, outpath):
    """Visualize parallel trends test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Slopes histogram
    ax = axes[0]
    colors = [RIVER_COLORS[get_river(w)] for w in pt_df["weir"]]
    ax.barh(range(len(pt_df)), pt_df["slope_per_year"].values, color=colors, alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_yticks(range(len(pt_df)))
    ax.set_yticklabels([WEIR_EN.get(w, w) for w in pt_df["weir"]], fontsize=8)
    ax.set_xlabel("Pre-period Slope of (Treatment - Control)", fontsize=10)
    ax.set_title(f"(a) Per-Weir Pre-Trend Slopes\nt={t_stat:.2f}, p={p_val:.3f}",
                fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel B: Scatter of slopes
    ax2 = axes[1]
    for river, color in RIVER_COLORS.items():
        mask = pt_df["river"] == river
        ax2.scatter(pt_df.loc[mask, "slope_per_year"],
                   pt_df.loc[mask, "p_value"],
                   c=color, label=river, s=60, edgecolors="gray")
    ax2.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="p=0.05")
    ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Pre-period Slope", fontsize=10)
    ax2.set_ylabel("p-value", fontsize=10)
    ax2.set_title("(b) Individual Pre-Trend Significance", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def plot_treatment_vs_control_ts(treatment, control, outpath):
    """Overlay time series: treatment (solid) vs control (dashed) for all rivers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (river, weir_list) in enumerate(RIVER_GROUPS.items()):
        ax = axes[idx]
        color = RIVER_COLORS[river]

        # Aggregate treatment
        t_yearly = defaultdict(list)
        c_yearly = defaultdict(list)
        for wn in weir_list:
            if wn in treatment:
                for yr, val in extract_yearly_series(treatment[wn]).items():
                    t_yearly[yr].append(val)
            if wn in control:
                for yr, val in extract_yearly_series(control[wn]).items():
                    c_yearly[yr].append(val)

        t_years = sorted(t_yearly.keys())
        t_means = [np.mean(t_yearly[y]) for y in t_years]
        c_years = sorted(c_yearly.keys())
        c_means = [np.mean(c_yearly[y]) for y in c_years]

        ax.plot(t_years, t_means, "o-", color=color, markersize=4,
                linewidth=1.5, label="Weir Sites (Treatment)")
        ax.plot(c_years, c_means, "s--", color=color, markersize=4,
                linewidth=1.5, alpha=0.6, label="Upstream (Control)")

        ax.axvline(x=2012, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"{river} River (n={len(weir_list)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Bloom Proxy (NDVI)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Treatment vs Control: Bloom Proxy Time Series by River",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


# ═══════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════

def main():
    print("Loading treatment (weir) data...")
    treatment = load_treatment_data()
    print(f"  Loaded {len(treatment)} weir sites.")

    print("Loading control (upstream) data...")
    control = load_control_data()
    print(f"  Loaded {len(control)} control reaches.")

    if not control:
        print("\nERROR: No control data found. Run extract_control_reaches.py first.")
        print("       Output expected in: output/control_data/*.json")
        return

    # Match treatment-control pairs
    paired = set(treatment.keys()) & set(control.keys())
    print(f"\n  Matched pairs: {len(paired)}/{len(treatment)}")

    report = []
    report.append("BACI ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Treatment sites: {len(treatment)}")
    report.append(f"Control reaches: {len(control)}")
    report.append(f"Matched pairs: {len(paired)}")
    report.append("")

    # 1. BACI Event-Study
    print("\n[1/4] BACI Event-Study...")
    baci_df = baci_event_study(treatment, control)
    if not baci_df.empty:
        plot_baci_event_study(baci_df, FIGURES_DIR / "baci_event_study.png")
        pre_coeffs = baci_df[baci_df["relative_year"] < 0]["baci_coeff"]
        post_coeffs = baci_df[baci_df["relative_year"] > 0]["baci_coeff"]
        report.append("=== 1. BACI EVENT-STUDY ===")
        report.append(f"Pre-period mean BACI: {pre_coeffs.mean():.4f} (should be ~0)")
        report.append(f"Post-period mean BACI: {post_coeffs.mean():.4f}")
        report.append(f"Structural break (post-pre): {post_coeffs.mean() - pre_coeffs.mean():.4f}")
        report.append("")

    # 2. BACI by River
    print("[2/4] BACI Event-Study by River...")
    river_baci = baci_by_river(treatment, control)
    if river_baci:
        plot_baci_by_river(river_baci, FIGURES_DIR / "baci_by_river.png")

    # 3. Parallel Trends Test
    print("[3/4] Parallel Trends Test...")
    pt_df, pt_t, pt_p = parallel_trends_test(treatment, control)
    if pt_df is not None and not pt_df.empty:
        plot_parallel_trends(pt_df, pt_t, pt_p, FIGURES_DIR / "parallel_trends.png")
        report.append("=== 2. PARALLEL TRENDS TEST ===")
        report.append(f"Mean pre-period slope of (treatment-control): {pt_df['slope_per_year'].mean():.6f}/year")
        report.append(f"t-test (H0: slope=0): t={pt_t:.3f}, p={pt_p:.4f}")
        if pt_p > 0.05:
            report.append("=> PASS: Parallel trends hold (p > 0.05)")
        else:
            report.append("=> WARNING: Parallel trends violated")
        sig = pt_df[pt_df["p_value"] < 0.05]
        report.append(f"Weirs with significant individual divergence: {len(sig)}/{len(pt_df)}")
        report.append("")

    # 4. BACI DID Estimates
    print("[4/4] BACI DID Estimates...")
    did_df = baci_did_estimates(treatment, control)
    if not did_df.empty:
        did_df.to_csv(OUTPUT_DIR / "baci_did_estimates.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'baci_did_estimates.csv'}")

        report.append("=== 3. BACI DID ESTIMATES ===")
        report.append(f"Mean DID estimate: {did_df['did_estimate'].mean():.4f}")
        report.append(f"Mean DID (%): {did_df['did_pct'].mean():.1f}%")
        sig_count = did_df["significant"].sum()
        report.append(f"Significant at p<0.05: {sig_count}/{len(did_df)}")
        report.append("")

        for river in RIVER_GROUPS:
            r_df = did_df[did_df["river"] == river]
            if not r_df.empty:
                report.append(f"  {river}: DID = {r_df['did_estimate'].mean():.4f} "
                             f"({r_df['did_pct'].mean():+.1f}%), "
                             f"sig: {r_df['significant'].sum()}/{len(r_df)}")
        report.append("")

    # Treatment vs Control time series
    print("\nPlotting treatment vs control time series...")
    plot_treatment_vs_control_ts(treatment, control,
                                 FIGURES_DIR / "baci_treatment_control_ts.png")

    # Save report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "baci_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {report_path}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
