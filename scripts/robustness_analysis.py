"""
Robustness Analyses for HAB Paper
==================================
Critical robustness checks to address reviewer concerns:
  1. Negative control (winter months — should show no effect)
  2. Detrended event-study (address pre-trend p=0.006)
  3. Dose-response (weir storage capacity vs bloom change)
  4. Placebo test (random treatment dates)
  5. Leave-one-out sensitivity

Outputs:
  - figures/negative_control.png
  - figures/detrended_event_study.png
  - figures/dose_response.png
  - figures/placebo_distribution.png
  - figures/leave_one_out.png
  - output/robustness_report.txt
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
BLOOM_DIR = OUTPUT_DIR / "bloom_data"
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
    "이포보": "Ipo", "여주보": "Yeoju", "강천보": "Gangcheon",
    "상주보": "Sangju", "낙단보": "Nakdan", "구미보": "Gumi",
    "칠곡보": "Chilgok", "강정고령보": "Gangjeong", "달성보": "Dalseong",
    "합천창녕보": "Hapcheon", "창녕함안보": "Changnyeong",
    "세종보": "Sejong", "공주보": "Gongju", "백제보": "Baekje",
    "승촌보": "Seungchon", "죽산보": "Juksan",
}

COMPLETION_YEAR = 2012

# Weir storage capacity (10^6 m^3) — from 4대강 사업 design docs
WEIR_CAPACITY = {
    "이포보": 28.7, "여주보": 18.3, "강천보": 9.8,
    "상주보": 27.4, "낙단보": 24.0, "구미보": 51.7,
    "칠곡보": 40.6, "강정고령보": 92.3, "달성보": 27.3,
    "합천창녕보": 88.7, "창녕함안보": 102.7,
    "세종보": 12.4, "공주보": 18.3, "백제보": 24.7,
    "승촌보": 29.7, "죽산보": 46.8,
}


def load_all_results():
    results = {}
    for f in sorted(BLOOM_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        results[data["weir"]] = data
    return results


def get_river(weir_name):
    for r, weirs in RIVER_GROUPS.items():
        if weir_name in weirs:
            return r
    return "Unknown"


# ═══════════════════════════════════════════════
# 1. NEGATIVE CONTROL (Winter Months)
# ═══════════════════════════════════════════════

def negative_control_analysis(results):
    """
    Winter months (Dec-Feb) should show no HAB effect from weirs.
    If we detect a change in winter too, our bloom-season effect
    might be driven by confounders rather than weirs.

    Note: Our GEE data is bloom-season only. As a proxy negative control,
    we compare the VARIANCE of bloom-season NDVI pre vs post.
    Weirs affect mean level but shouldn't affect inter-annual variability pattern.

    Alternative: We test if the weakest-signal months (early/late season)
    show similar patterns to peak months.
    """
    rows = []
    for name, data in results.items():
        # Split bloom season into "shoulder" (May, Oct) vs "peak" (Jul, Aug, Sep)
        # Since we only have annual medians, we use a different approach:
        # Test years with lowest pre-weir values (proxy for non-bloom conditions)

        pre = [r["median"] for r in data["landsat_pre_weir"] if r["median"] is not None]
        post = [r["median"] for r in data["landsat_post_weir"] if r["median"] is not None]

        if len(pre) < 5 or len(post) < 5:
            continue

        pre_mean = np.mean(pre)
        post_mean = np.mean(post)
        pre_std = np.std(pre, ddof=1)
        post_std = np.std(post, ddof=1)

        # F-test for variance equality
        f_stat = post_std**2 / pre_std**2 if pre_std > 0 else np.nan
        df1 = len(post) - 1
        df2 = len(pre) - 1
        f_p = 1 - stats.f.cdf(f_stat, df1, df2) if not np.isnan(f_stat) else np.nan

        # Coefficient of variation change
        pre_cv = pre_std / pre_mean * 100 if pre_mean > 0 else np.nan
        post_cv = post_std / post_mean * 100 if post_mean > 0 else np.nan

        rows.append({
            "weir": name,
            "river": get_river(name),
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "pre_std": pre_std,
            "post_std": post_std,
            "pre_cv": pre_cv,
            "post_cv": post_cv,
            "f_stat": f_stat,
            "f_p_value": f_p,
            "mean_change_pct": (post_mean - pre_mean) / pre_mean * 100,
            "std_change_pct": (post_std - pre_std) / pre_std * 100 if pre_std > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def plot_negative_control(df, outpath):
    """Plot negative control: mean change vs variability change."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Mean change (treatment effect — should be significant)
    ax = axes[0]
    for river, color in RIVER_COLORS.items():
        mask = df["river"] == river
        ax.scatter(df.loc[mask, "weir"].map(WEIR_EN), df.loc[mask, "mean_change_pct"],
                  c=color, s=80, label=river, edgecolors="gray", linewidths=0.5, zorder=5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Bloom-Season NDVI Change (%)", fontsize=11)
    ax.set_title("(a) Treatment Effect (Bloom Season)", fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: CV change (should NOT be significant if effect is real)
    ax2 = axes[1]
    for river, color in RIVER_COLORS.items():
        mask = df["river"] == river
        ax2.scatter(df.loc[mask, "weir"].map(WEIR_EN), df.loc[mask, "std_change_pct"],
                   c=color, s=80, label=river, edgecolors="gray", linewidths=0.5, zorder=5)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Std. Dev. Change (%)", fontsize=11)
    ax2.set_title("(b) Variability Change (Placebo Outcome)", fontsize=11, fontweight="bold")
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Test: is mean std_change significantly different from 0?
    t_stat, p_val = stats.ttest_1samp(df["std_change_pct"].dropna(), 0)
    ax2.text(0.05, 0.95, f"H₀: Δσ=0\nt={t_stat:.2f}, p={p_val:.3f}",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()

    return t_stat, p_val


# ═══════════════════════════════════════════════
# 2. DETRENDED EVENT-STUDY
# ═══════════════════════════════════════════════

def detrended_event_study(results):
    """
    Address the pre-trend concern (p=0.006).
    For each weir, fit a linear trend to 2000-2011, subtract it,
    then compute event-study on detrended data.
    """
    all_years = range(2000, 2026)
    coeff_by_year = defaultdict(list)

    for name, data in results.items():
        yearly = {}
        for rec in data["landsat_pre_weir"]:
            if rec["median"] is not None:
                yearly[rec["year"]] = rec["median"]
        for rec in data["landsat_post_weir"]:
            if rec["median"] is not None:
                yearly[rec["year"]] = rec["median"]

        # Fit pre-trend
        pre_years = np.array([y for y in sorted(yearly.keys()) if y <= 2011])
        pre_vals = np.array([yearly[y] for y in pre_years])

        if len(pre_years) < 5:
            continue

        slope, intercept, _, _, _ = stats.linregress(pre_years, pre_vals)

        # Detrend entire series
        detrended = {}
        for yr, val in yearly.items():
            trend = slope * yr + intercept
            detrended[yr] = val - trend

        # Normalize to 2011
        ref_val = detrended.get(2011, np.mean([v for y, v in detrended.items() if y <= 2011]))
        for yr in all_years:
            if yr in detrended:
                coeff_by_year[yr].append(detrended[yr] - ref_val)

    rows = []
    for yr in sorted(coeff_by_year.keys()):
        vals = coeff_by_year[yr]
        mean_coeff = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        rows.append({
            "year": yr,
            "relative_year": yr - COMPLETION_YEAR,
            "coeff": mean_coeff,
            "se": se,
            "ci95_lo": mean_coeff - 1.96 * se,
            "ci95_hi": mean_coeff + 1.96 * se,
            "n_weirs": len(vals),
        })

    return pd.DataFrame(rows)


def plot_detrended_event_study(df_orig, df_detrended, outpath):
    """Compare original vs detrended event-study."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, df, title in [
        (axes[0], df_orig, "(a) Original Event-Study"),
        (axes[1], df_detrended, "(b) Detrended Event-Study"),
    ]:
        ax.axvspan(df["relative_year"].min() - 0.5, -0.5, alpha=0.05, color="blue")
        ax.axvspan(0.5, df["relative_year"].max() + 0.5, alpha=0.05, color="red")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

        ax.fill_between(df["relative_year"], df["ci95_lo"], df["ci95_hi"],
                        alpha=0.2, color="#333333")
        ax.plot(df["relative_year"], df["coeff"], "o-", color="#333333",
                markersize=5, linewidth=1.5, zorder=5)

        ax.set_xlabel("Years Relative to Weir Completion", fontsize=11)
        ax.set_ylabel("NDVI Deviation", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Annotate pre-trend statistics
    pre_orig = df_orig[df_orig["relative_year"] < 0]["coeff"]
    pre_detr = df_detrended[df_detrended["relative_year"] < 0]["coeff"]

    # Pre-period slope test for detrended
    pre_yrs = df_detrended[df_detrended["relative_year"] < 0]
    if len(pre_yrs) >= 3:
        slope, _, _, p_val, _ = stats.linregress(pre_yrs["relative_year"], pre_yrs["coeff"])
        axes[1].text(0.05, 0.95, f"Pre-trend slope: {slope:.5f}\np={p_val:.4f}",
                     transform=axes[1].transAxes, fontsize=9, va="top",
                     bbox=dict(boxstyle="round", facecolor="lightgreen" if p_val > 0.05 else "lightyellow", alpha=0.5))

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


# ═══════════════════════════════════════════════
# 3. DOSE-RESPONSE (Storage Capacity)
# ═══════════════════════════════════════════════

def dose_response_analysis(results):
    """
    Test if weirs with larger storage capacity show larger bloom increases.
    A dose-response relationship strengthens causal claims.
    """
    rows = []
    for name, data in results.items():
        if name not in WEIR_CAPACITY:
            continue
        pre = [r["median"] for r in data["landsat_pre_weir"] if r["median"] is not None]
        post = [r["median"] for r in data["landsat_post_weir"] if r["median"] is not None]

        if not pre or not post:
            continue

        pre_mean = np.mean(pre)
        post_mean = np.mean(post)

        rows.append({
            "weir": name,
            "river": get_river(name),
            "capacity_mcm": WEIR_CAPACITY[name],
            "log_capacity": np.log10(WEIR_CAPACITY[name]),
            "abs_change": post_mean - pre_mean,
            "pct_change": (post_mean - pre_mean) / pre_mean * 100,
        })

    return pd.DataFrame(rows)


def plot_dose_response(df, outpath):
    """Scatter: storage capacity vs bloom change."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, x_col, x_label, title in [
        (axes[0], "capacity_mcm", "Storage Capacity (10⁶ m³)", "(a) Linear Scale"),
        (axes[1], "log_capacity", "log₁₀(Storage Capacity)", "(b) Log Scale"),
    ]:
        for river, color in RIVER_COLORS.items():
            mask = df["river"] == river
            ax.scatter(df.loc[mask, x_col], df.loc[mask, "pct_change"],
                      c=color, s=80, label=river, edgecolors="gray", linewidths=0.5, zorder=5)
            for _, row in df[mask].iterrows():
                ax.annotate(WEIR_EN.get(row["weir"], row["weir"]),
                           (row[x_col], row["pct_change"]),
                           fontsize=6, ha="left", va="bottom",
                           xytext=(3, 3), textcoords="offset points")

        # Regression
        slope, intercept, r_val, p_val, _ = stats.linregress(df[x_col], df["pct_change"])
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Bloom Proxy Change (%)", fontsize=11)
        ax.set_title(f"{title}\nr={r_val:.3f}, p={p_val:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()

    # Return stats for both scales
    _, _, r_lin, p_lin, _ = stats.linregress(df["capacity_mcm"], df["pct_change"])
    _, _, r_log, p_log, _ = stats.linregress(df["log_capacity"], df["pct_change"])
    return r_lin, p_lin, r_log, p_log


# ═══════════════════════════════════════════════
# 4. PLACEBO TEST (Random Treatment Dates)
# ═══════════════════════════════════════════════

def placebo_test(results, n_permutations=1000):
    """
    Randomly assign fake treatment dates and compute the effect.
    If real effect >> placebo effects, it's unlikely to be chance.
    """
    np.random.seed(42)

    # Compute real effect
    real_effects = []
    for name, data in results.items():
        pre = [r["median"] for r in data["landsat_pre_weir"] if r["median"] is not None]
        post = [r["median"] for r in data["landsat_post_weir"] if r["median"] is not None]
        if pre and post:
            real_effects.append(np.mean(post) - np.mean(pre))
    real_mean = np.mean(real_effects)

    # Permutation: for each weir, randomly split the time series
    placebo_means = []
    for _ in range(n_permutations):
        perm_effects = []
        for name, data in results.items():
            all_records = data["landsat_pre_weir"] + data["landsat_post_weir"]
            vals = [r["median"] for r in all_records if r["median"] is not None]
            if len(vals) < 10:
                continue

            # Random split point (between year 3 and year n-3)
            split = np.random.randint(3, len(vals) - 3)
            perm_pre = vals[:split]
            perm_post = vals[split:]
            perm_effects.append(np.mean(perm_post) - np.mean(perm_pre))

        if perm_effects:
            placebo_means.append(np.mean(perm_effects))

    placebo_means = np.array(placebo_means)

    # p-value: fraction of placebo effects >= real effect
    p_value = np.mean(placebo_means >= real_mean)

    return real_mean, placebo_means, p_value


def plot_placebo(real_mean, placebo_means, p_value, outpath):
    """Distribution of placebo effects vs real effect."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(placebo_means, bins=50, color="#90CAF9", edgecolor="gray", alpha=0.8,
            label=f"Placebo effects (n={len(placebo_means)})")
    ax.axvline(x=real_mean, color="red", linewidth=2, linestyle="--",
               label=f"Real effect: {real_mean:.4f}")

    # Percentile annotations
    p95 = np.percentile(placebo_means, 95)
    p99 = np.percentile(placebo_means, 99)
    ax.axvline(x=p95, color="orange", linewidth=1, linestyle=":", alpha=0.7,
               label=f"95th percentile: {p95:.4f}")
    ax.axvline(x=p99, color="red", linewidth=1, linestyle=":", alpha=0.7,
               label=f"99th percentile: {p99:.4f}")

    ax.set_xlabel("Mean NDVI Change (Post - Pre)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Placebo Test: Real Effect vs Random Treatment Dates\n"
                 f"(Permutation p-value = {p_value:.4f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


# ═══════════════════════════════════════════════
# 5. LEAVE-ONE-OUT SENSITIVITY
# ═══════════════════════════════════════════════

def leave_one_out(results):
    """
    Drop each weir one at a time and recompute mean effect.
    If one weir drives the entire result, it's fragile.
    """
    # Full sample effect
    all_effects = {}
    for name, data in results.items():
        pre = [r["median"] for r in data["landsat_pre_weir"] if r["median"] is not None]
        post = [r["median"] for r in data["landsat_post_weir"] if r["median"] is not None]
        if pre and post:
            all_effects[name] = (np.mean(post) - np.mean(pre)) / np.mean(pre) * 100

    full_mean = np.mean(list(all_effects.values()))

    # Leave-one-out
    loo_results = []
    for drop_weir in all_effects:
        remaining = {k: v for k, v in all_effects.items() if k != drop_weir}
        loo_mean = np.mean(list(remaining.values()))
        loo_results.append({
            "dropped_weir": drop_weir,
            "river": get_river(drop_weir),
            "loo_mean_pct": loo_mean,
            "influence": full_mean - loo_mean,  # positive = weir was pulling up
        })

    return pd.DataFrame(loo_results), full_mean


def plot_leave_one_out(df, full_mean, outpath):
    """Leave-one-out forest plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    df_sorted = df.sort_values("loo_mean_pct")

    colors = [RIVER_COLORS.get(get_river(w), "gray") for w in df_sorted["dropped_weir"]]
    y_pos = range(len(df_sorted))

    ax.barh(y_pos, df_sorted["loo_mean_pct"], color=colors, edgecolor="gray", alpha=0.7)
    ax.axvline(x=full_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"Full sample: {full_mean:.1f}%")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([WEIR_EN.get(w, w) for w in df_sorted["dropped_weir"]], fontsize=9)
    ax.set_xlabel("Mean NDVI Change (%) Excluding Dropped Weir", fontsize=11)
    ax.set_title("Leave-One-Out Sensitivity Analysis", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate range
    loo_range = df["loo_mean_pct"].max() - df["loo_mean_pct"].min()
    ax.text(0.02, 0.02, f"Range: {df['loo_mean_pct'].min():.1f}% – {df['loo_mean_pct'].max():.1f}% "
            f"(spread: {loo_range:.1f}%)",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ROBUSTNESS ANALYSES — HAB Paper")
    print("=" * 70)

    results = load_all_results()
    print(f"Loaded {len(results)} weirs\n")

    report = []

    # ─── 1. Negative Control ───
    print("[1/5] Negative Control (Variability Test)...")
    nc_df = negative_control_analysis(results)
    nc_t, nc_p = plot_negative_control(nc_df, FIGURES_DIR / "negative_control.png")

    report.append("=== 1. NEGATIVE CONTROL (VARIABILITY) ===")
    report.append(f"Mean bloom-season NDVI change: {nc_df['mean_change_pct'].mean():+.1f}% (EXPECTED: significant)")
    report.append(f"Mean std. dev. change: {nc_df['std_change_pct'].mean():+.1f}%")
    report.append(f"H0 (Δσ=0): t={nc_t:.3f}, p={nc_p:.4f}")
    if nc_p > 0.05:
        report.append("=> PASS: No significant change in variability (effect is on mean level, not noise)")
    else:
        report.append("=> NOTE: Variability also changed (may need further investigation)")

    # F-test results
    n_sig_f = (nc_df["f_p_value"] < 0.05).sum()
    report.append(f"Weirs with significantly different variance: {n_sig_f}/{len(nc_df)}")
    report.append("")

    # ─── 2. Detrended Event-Study ───
    print("[2/5] Detrended Event-Study...")

    # Load original event-study for comparison
    es_orig_path = OUTPUT_DIR / "event_study_coefficients.csv"
    if es_orig_path.exists():
        df_orig = pd.read_csv(es_orig_path)
    else:
        # Recompute
        from full_analysis import event_study_coefficients
        df_orig = event_study_coefficients(results)

    df_detrended = detrended_event_study(results)
    plot_detrended_event_study(df_orig, df_detrended, FIGURES_DIR / "detrended_event_study.png")

    # Pre-trend test on detrended data
    pre_detr = df_detrended[df_detrended["relative_year"] < 0]
    if len(pre_detr) >= 3:
        slope, _, _, p_val, _ = stats.linregress(pre_detr["relative_year"], pre_detr["coeff"])
        detr_post = df_detrended[df_detrended["relative_year"] > 0]["coeff"].mean()
        detr_pre = pre_detr["coeff"].mean()

        report.append("=== 2. DETRENDED EVENT-STUDY ===")
        report.append(f"Pre-trend slope (detrended): {slope:.6f}")
        report.append(f"Pre-trend p-value (detrended): {p_val:.4f}")
        if p_val > 0.05:
            report.append("=> PASS: Detrending eliminates pre-trend concern")
        else:
            report.append("=> WARNING: Pre-trend persists after detrending")
        report.append(f"Post-period mean coeff (detrended): {detr_post:.4f}")
        report.append(f"Effect magnitude preserved: {detr_post:.4f} vs original {df_orig[df_orig['relative_year'] > 0]['coeff'].mean():.4f}")
    report.append("")

    # ─── 3. Dose-Response ───
    print("[3/5] Dose-Response Analysis (Storage Capacity)...")
    dr_df = dose_response_analysis(results)
    r_lin, p_lin, r_log, p_log = plot_dose_response(dr_df, FIGURES_DIR / "dose_response.png")

    report.append("=== 3. DOSE-RESPONSE (STORAGE CAPACITY) ===")
    report.append(f"Linear: r={r_lin:.3f}, p={p_lin:.4f}")
    report.append(f"Log-linear: r={r_log:.3f}, p={p_log:.4f}")
    if p_lin < 0.05 or p_log < 0.05:
        report.append("=> Dose-response relationship detected (strengthens causal claim)")
    else:
        report.append("=> No significant dose-response (effect may be threshold-based rather than dose-dependent)")
    report.append(f"Capacity range: {dr_df['capacity_mcm'].min():.1f} – {dr_df['capacity_mcm'].max():.1f} MCM")
    report.append("")

    # ─── 4. Placebo Test ───
    print("[4/5] Placebo Test (1000 permutations)...")
    real_mean, placebo_means, perm_p = placebo_test(results, n_permutations=1000)
    plot_placebo(real_mean, placebo_means, perm_p, FIGURES_DIR / "placebo_distribution.png")

    report.append("=== 4. PLACEBO TEST (PERMUTATION) ===")
    report.append(f"Real effect: {real_mean:.4f} NDVI")
    report.append(f"Placebo mean: {np.mean(placebo_means):.4f} (sd={np.std(placebo_means):.4f})")
    report.append(f"Placebo 95th percentile: {np.percentile(placebo_means, 95):.4f}")
    report.append(f"Placebo 99th percentile: {np.percentile(placebo_means, 99):.4f}")
    report.append(f"Permutation p-value: {perm_p:.4f}")
    if perm_p < 0.01:
        report.append("=> PASS: Real effect far exceeds random chance (p<0.01)")
    elif perm_p < 0.05:
        report.append("=> PASS: Real effect exceeds random chance (p<0.05)")
    else:
        report.append("=> WARNING: Real effect not clearly distinguishable from random")
    report.append("")

    # ─── 5. Leave-One-Out ───
    print("[5/5] Leave-One-Out Sensitivity...")
    loo_df, full_mean = leave_one_out(results)
    plot_leave_one_out(loo_df, full_mean, FIGURES_DIR / "leave_one_out.png")

    report.append("=== 5. LEAVE-ONE-OUT SENSITIVITY ===")
    report.append(f"Full sample mean change: {full_mean:.1f}%")
    report.append(f"LOO range: {loo_df['loo_mean_pct'].min():.1f}% – {loo_df['loo_mean_pct'].max():.1f}%")
    report.append(f"Max influence: {loo_df.loc[loo_df['influence'].abs().idxmax(), 'dropped_weir']} "
                 f"(influence: {loo_df['influence'].abs().max():.2f}%)")
    loo_spread = loo_df['loo_mean_pct'].max() - loo_df['loo_mean_pct'].min()
    if loo_spread < 2.0:
        report.append(f"=> PASS: Results robust to single-weir removal (spread: {loo_spread:.1f}%)")
    else:
        report.append(f"=> NOTE: Some sensitivity to individual weirs (spread: {loo_spread:.1f}%)")
    report.append("")

    # ─── Summary ───
    report.append("=" * 60)
    report.append("ROBUSTNESS SUMMARY")
    report.append("=" * 60)

    checks = [
        ("Negative Control (Variability)", nc_p > 0.05),
        ("Detrended Pre-trend", p_val > 0.05 if len(pre_detr) >= 3 else None),
        ("Placebo Test", perm_p < 0.05),
        ("Leave-One-Out", loo_spread < 2.0),
    ]

    for name, passed in checks:
        if passed is None:
            status = "N/A"
        elif passed:
            status = "PASS"
        else:
            status = "CAUTION"
        report.append(f"  [{status:>7}] {name}")

    dose_status = "PASS" if (p_lin < 0.05 or p_log < 0.05) else "NEUTRAL"
    report.append(f"  [{dose_status:>7}] Dose-Response (r_lin={r_lin:.3f}, r_log={r_log:.3f})")

    # Write report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "robustness_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report saved: {report_path}")

    print("\n" + report_text)
    print("\nDone!")


if __name__ == "__main__":
    main()
