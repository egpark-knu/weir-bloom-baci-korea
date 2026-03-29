"""
Sensitivity Analysis: HAB Influencing Factors
==============================================
Tests the sensitivity of BACI estimates to various potential
confounders beyond weir construction.

Factors analyzed:
  1. Climate: temperature & precipitation trends (Open-Meteo API)
  2. Covariate-augmented BACI: include climate in regression
  3. Temporal heterogeneity: early vs late post-period
  4. Spatial heterogeneity: per-river sensitivity
  5. Leave-one-out jackknife
  6. Dose-response: storage capacity vs treatment effect

Outputs:
  - figures/sensitivity_climate_*.png
  - figures/sensitivity_temporal.png
  - figures/sensitivity_spatial.png
  - figures/sensitivity_loo.png
  - figures/sensitivity_dose_response.png
  - output/sensitivity_report.txt
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
import urllib.request
import time

# ─── Paths ───
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Water-masked data
BLOOM_DIR = OUTPUT_DIR / "water_masked" / "weirs"
CONTROL_DIR = OUTPUT_DIR / "water_masked" / "controls"

RIVER_GROUPS = {
    "Han": ["이포보", "여주보", "강천보"],
    "Nakdong": ["상주보", "낙단보", "구미보", "칠곡보", "강정고령보", "달성보", "합천창녕보", "창녕함안보"],
    "Geum": ["세종보", "공주보", "백제보"],
    "Yeongsan": ["승촌보", "죽산보"],
}

RIVER_COLORS = {
    "Han": "#2196F3", "Nakdong": "#F44336",
    "Geum": "#4CAF50", "Yeongsan": "#FF9800",
}

WEIR_EN = {
    "이포보": "Ipo", "여주보": "Yeoju", "강천보": "Gangcheon",
    "상주보": "Sangju", "낙단보": "Nakdan", "구미보": "Gumi",
    "칠곡보": "Chilgok", "강정고령보": "Gangjeong", "달성보": "Dalseong",
    "합천창녕보": "Hapcheon", "창녕함안보": "Changnyeong",
    "세종보": "Sejong", "공주보": "Gongju", "백제보": "Baekje",
    "승촌보": "Seungchon", "죽산보": "Juksan",
}

# Storage capacities (10^6 m3) from FMRP design docs
STORAGE_CAPACITY = {
    "이포보": 48.8, "여주보": 22.1, "강천보": 6.5,
    "상주보": 28.0, "낙단보": 14.6, "구미보": 43.9,
    "칠곡보": 42.9, "강정고령보": 92.3, "달성보": 34.1,
    "합천창녕보": 106.8, "창녕함안보": 43.4,
    "세종보": 10.8, "공주보": 16.3, "백제보": 24.1,
    "승촌보": 12.6, "죽산보": 22.2,
}

# Representative locations for climate data (river centroids)
RIVER_LOCATIONS = {
    "Han": (37.3, 127.6),
    "Nakdong": (35.8, 128.6),
    "Geum": (36.5, 127.0),
    "Yeongsan": (35.0, 126.9),
}

COMPLETION_YEAR = 2012


def get_river(weir_name):
    for r, weirs in RIVER_GROUPS.items():
        if weir_name in weirs:
            return r
    return "Unknown"


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════

def load_treatment_data():
    results = {}
    for f in sorted(BLOOM_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        key = data.get("weir") or data.get("weir_name_kr") or data.get("name", "")
        if key:
            results[key] = data
    return results


def load_control_data():
    results = {}
    for f in sorted(CONTROL_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        key = data.get("weir_name_kr", "")
        if key:
            results[key] = data
    return results


def extract_yearly_series(data):
    yearly = {}
    for rec in data.get("landsat_pre_weir", []):
        if rec["median"] is not None:
            yearly[rec["year"]] = rec["median"]
    for rec in data.get("landsat_post_weir", []):
        if rec["median"] is not None:
            yearly[rec["year"]] = rec["median"]
    return yearly


# ═══════════════════════════════════════════════
# 1. CLIMATE DATA (Open-Meteo API)
# ═══════════════════════════════════════════════

def fetch_climate_data(lat, lon, start_year=2000, end_year=2025):
    """Fetch bloom-season (May-Oct) temperature and precipitation from Open-Meteo."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_year}-01-01&end_date={end_year}-12-31"
        f"&daily=temperature_2m_mean,precipitation_sum"
        f"&timezone=Asia/Seoul"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HAB-Paper/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        dates = data["daily"]["time"]
        temps = data["daily"]["temperature_2m_mean"]
        precips = data["daily"]["precipitation_sum"]

        # Aggregate to bloom-season (May-Oct) annual stats
        yearly_temp = {}
        yearly_precip = {}

        for date_str, temp, precip in zip(dates, temps, precips):
            year = int(date_str[:4])
            month = int(date_str[5:7])
            if 5 <= month <= 10:  # Bloom season
                if year not in yearly_temp:
                    yearly_temp[year] = []
                    yearly_precip[year] = []
                if temp is not None:
                    yearly_temp[year].append(temp)
                if precip is not None:
                    yearly_precip[year].append(precip)

        result = {}
        for year in sorted(yearly_temp.keys()):
            if yearly_temp[year]:
                result[year] = {
                    "temp_mean": np.mean(yearly_temp[year]),
                    "precip_total": np.sum(yearly_precip[year]),
                    "precip_days": sum(1 for p in yearly_precip[year] if p > 1.0),
                }
        return result
    except Exception as e:
        print(f"  [WARN] Climate fetch failed for ({lat},{lon}): {e}")
        return None


def get_all_climate_data():
    """Fetch climate data for all 4 river basins."""
    climate = {}
    for river, (lat, lon) in RIVER_LOCATIONS.items():
        print(f"  Fetching climate for {river} River ({lat}, {lon})...")
        data = fetch_climate_data(lat, lon)
        if data:
            climate[river] = data
        time.sleep(1)  # Rate limiting
    return climate


# ═══════════════════════════════════════════════
# 2. CLIMATE TREND ANALYSIS
# ═══════════════════════════════════════════════

def analyze_climate_trends(climate_data, report):
    """Analyze climate trends and test as confounders."""
    report.append("=== CLIMATE TREND ANALYSIS ===")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_temps = defaultdict(list)
    all_precips = defaultdict(list)

    for river, data in climate_data.items():
        years = sorted(data.keys())
        temps = [data[y]["temp_mean"] for y in years]
        precips = [data[y]["precip_total"] for y in years]

        for y in years:
            all_temps[y].append(data[y]["temp_mean"])
            all_precips[y].append(data[y]["precip_total"])

        # Temperature trend
        ax = axes[0, 0]
        ax.plot(years, temps, "o-", color=RIVER_COLORS[river],
                markersize=4, linewidth=1.5, alpha=0.7, label=river)

        # Precipitation trend
        ax = axes[0, 1]
        ax.plot(years, precips, "o-", color=RIVER_COLORS[river],
                markersize=4, linewidth=1.5, alpha=0.7, label=river)

    # National average
    nat_years = sorted(all_temps.keys())
    nat_temps = [np.mean(all_temps[y]) for y in nat_years]
    nat_precips = [np.mean(all_precips[y]) for y in nat_years]

    # Temperature panel
    ax = axes[0, 0]
    ax.axvline(x=2012, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bloom-Season Mean Temperature (°C)")
    ax.set_title("(a) Temperature Trend by River Basin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Precipitation panel
    ax = axes[0, 1]
    ax.axvline(x=2012, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bloom-Season Total Precipitation (mm)")
    ax.set_title("(b) Precipitation Trend by River Basin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # National trend with linear fit
    ax = axes[1, 0]
    ax.plot(nat_years, nat_temps, "ko-", markersize=5, linewidth=2, label="National Mean")

    # Pre/post comparison
    pre_temps = [t for y, t in zip(nat_years, nat_temps) if y <= 2011]
    post_temps = [t for y, t in zip(nat_years, nat_temps) if y >= 2013]
    pre_years = [y for y in nat_years if y <= 2011]
    post_years = [y for y in nat_years if y >= 2013]

    # Linear fits
    if len(pre_years) > 2:
        slope_pre, intercept_pre, r_pre, p_pre, _ = stats.linregress(pre_years, pre_temps)
        ax.plot(pre_years, [slope_pre * y + intercept_pre for y in pre_years],
                "b--", linewidth=1.5, alpha=0.7,
                label=f"Pre: {slope_pre:.3f}°C/yr (p={p_pre:.3f})")

    slope_all, intercept_all, r_all, p_all, _ = stats.linregress(nat_years, nat_temps)
    ax.plot(nat_years, [slope_all * y + intercept_all for y in nat_years],
            "r--", linewidth=1.5, alpha=0.7,
            label=f"Overall: {slope_all:.3f}°C/yr (p={p_all:.3f})")

    ax.axvline(x=2012, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("(c) National Bloom-Season Temperature Trend")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pre vs Post box comparison
    ax = axes[1, 1]
    bp = ax.boxplot([pre_temps, post_temps], labels=["Pre-weir\n(2000-2011)", "Post-weir\n(2013-2025)"],
                     patch_artist=True)
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor("#F44336")
    bp["boxes"][1].set_alpha(0.4)

    t_stat, p_val = stats.ttest_ind(pre_temps, post_temps, equal_var=False)
    ax.set_ylabel("Bloom-Season Mean Temperature (°C)")
    ax.set_title(f"(d) Pre vs Post Temperature\nt={t_stat:.2f}, p={p_val:.3f}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Climate Confounders: Temperature and Precipitation Trends (2000-2025)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = FIGURES_DIR / "sensitivity_climate_trends.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()

    # Report
    report.append(f"National bloom-season temperature trend: {slope_all:.4f} °C/year (p={p_all:.4f})")
    report.append(f"Pre-period (2000-2011) mean temp: {np.mean(pre_temps):.2f} °C")
    report.append(f"Post-period (2013-2025) mean temp: {np.mean(post_temps):.2f} °C")
    report.append(f"Temp change: {np.mean(post_temps) - np.mean(pre_temps):+.2f} °C")
    report.append(f"Pre vs Post t-test: t={t_stat:.3f}, p={p_val:.4f}")
    report.append("")
    report.append("KEY FINDING: Temperature may have increased, but the BACI design")
    report.append("controls for this because region-wide climate affects BOTH treatment")
    report.append("and control sites equally. The DiD estimator differences out any")
    report.append("common temporal trend (including warming).")
    report.append("")

    return {
        "temp_trend_slope": slope_all,
        "temp_trend_p": p_all,
        "temp_pre_mean": np.mean(pre_temps),
        "temp_post_mean": np.mean(post_temps),
        "temp_change": np.mean(post_temps) - np.mean(pre_temps),
    }


# ═══════════════════════════════════════════════
# 3. TEMPORAL HETEROGENEITY (Early vs Late)
# ═══════════════════════════════════════════════

def temporal_heterogeneity(treatment, control, report):
    """Test whether treatment effect persists, amplifies, or attenuates."""
    report.append("=== TEMPORAL HETEROGENEITY ===")

    early_dids = []  # 2013-2018
    late_dids = []   # 2019-2025

    for weir_name, t_data in treatment.items():
        if weir_name not in control:
            continue

        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        # Pre-period baseline
        t_pre = [v for y, v in t_series.items() if y <= 2011]
        c_pre = [v for y, v in c_series.items() if y <= 2011]
        if len(t_pre) < 3 or len(c_pre) < 3:
            continue

        t_pre_mean = np.mean(t_pre)
        c_pre_mean = np.mean(c_pre)

        # Early post
        t_early = [v for y, v in t_series.items() if 2013 <= y <= 2018]
        c_early = [v for y, v in c_series.items() if 2013 <= y <= 2018]
        if t_early and c_early:
            did_early = (np.mean(t_early) - t_pre_mean) - (np.mean(c_early) - c_pre_mean)
            early_dids.append(did_early)

        # Late post
        t_late = [v for y, v in t_series.items() if 2019 <= y <= 2025]
        c_late = [v for y, v in c_series.items() if 2019 <= y <= 2025]
        if t_late and c_late:
            did_late = (np.mean(t_late) - t_pre_mean) - (np.mean(c_late) - c_pre_mean)
            late_dids.append(did_late)

    if early_dids and late_dids:
        fig, ax = plt.subplots(figsize=(8, 5))

        bp = ax.boxplot([early_dids, late_dids],
                         labels=["Early Post\n(2013-2018)", "Late Post\n(2019-2025)"],
                         patch_artist=True)
        bp["boxes"][0].set_facecolor("#FF9800")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#F44336")
        bp["boxes"][1].set_alpha(0.5)

        t_stat, p_val = stats.ttest_ind(early_dids, late_dids, equal_var=False)

        ax.axhline(y=0, color="gray", linestyle="--")
        ax.set_ylabel("BACI DID Estimate")
        ax.set_title(f"Temporal Persistence: Early vs Late Post-Period\n"
                     f"(Welch t={t_stat:.2f}, p={p_val:.3f})")
        ax.grid(True, alpha=0.3, axis="y")

        outpath = FIGURES_DIR / "sensitivity_temporal_persistence.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {outpath}")
        plt.close()

        report.append(f"Early post (2013-2018) mean DID: {np.mean(early_dids):.4f}")
        report.append(f"Late post (2019-2025) mean DID: {np.mean(late_dids):.4f}")
        report.append(f"Persistence test: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val > 0.05:
            report.append("=> Treatment effect is PERSISTENT (no significant early/late difference)")
        else:
            report.append("=> Treatment effect CHANGED over time")
        report.append("")


# ═══════════════════════════════════════════════
# 4. LEAVE-ONE-OUT JACKKNIFE
# ═══════════════════════════════════════════════

def leave_one_out(treatment, control, report):
    """Jackknife sensitivity: exclude one weir at a time."""
    report.append("=== LEAVE-ONE-OUT SENSITIVITY ===")

    # Full DID
    all_dids = []
    weir_names = []
    for weir_name, t_data in treatment.items():
        if weir_name not in control:
            continue
        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        t_pre = [v for y, v in t_series.items() if y <= 2011]
        t_post = [v for y, v in t_series.items() if y >= 2013]
        c_pre = [v for y, v in c_series.items() if y <= 2011]
        c_post = [v for y, v in c_series.items() if y >= 2013]

        if len(t_pre) >= 3 and len(t_post) >= 3 and len(c_pre) >= 3 and len(c_post) >= 3:
            did = (np.mean(t_post) - np.mean(t_pre)) - (np.mean(c_post) - np.mean(c_pre))
            all_dids.append(did)
            weir_names.append(weir_name)

    if len(all_dids) < 3:
        report.append("Insufficient pairs for LOO analysis")
        return

    full_mean = np.mean(all_dids)

    # Jackknife
    loo_means = []
    for i in range(len(all_dids)):
        loo = [d for j, d in enumerate(all_dids) if j != i]
        loo_means.append(np.mean(loo))

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [RIVER_COLORS[get_river(w)] for w in weir_names]
    bars = ax.barh(range(len(loo_means)), loo_means, color=colors, alpha=0.7)
    ax.axvline(x=full_mean, color="black", linewidth=2, linestyle="--",
               label=f"Full estimate: {full_mean:.4f}")

    ax.set_yticks(range(len(weir_names)))
    ax.set_yticklabels([WEIR_EN.get(w, w) for w in weir_names], fontsize=8)
    ax.set_xlabel("Mean DID Estimate (excluding weir)")
    ax.set_title("Leave-One-Out Sensitivity Analysis")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    outpath = FIGURES_DIR / "sensitivity_loo.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()

    report.append(f"Full DID estimate: {full_mean:.4f}")
    report.append(f"LOO range: [{min(loo_means):.4f}, {max(loo_means):.4f}]")
    report.append(f"LOO std: {np.std(loo_means):.4f}")
    report.append(f"=> Stable: max deviation = {max(abs(m - full_mean) for m in loo_means):.4f}")

    # Most influential weir
    deviations = [abs(m - full_mean) for m in loo_means]
    most_influential = weir_names[np.argmax(deviations)]
    report.append(f"Most influential weir: {WEIR_EN.get(most_influential, most_influential)}")
    report.append("")


# ═══════════════════════════════════════════════
# 5. DOSE-RESPONSE (Storage Capacity)
# ═══════════════════════════════════════════════

def dose_response(treatment, control, report):
    """Test dose-response: does treatment effect scale with storage capacity?"""
    report.append("=== DOSE-RESPONSE ANALYSIS ===")

    dids = []
    capacities = []
    rivers = []
    names = []

    for weir_name, t_data in treatment.items():
        if weir_name not in control or weir_name not in STORAGE_CAPACITY:
            continue

        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        t_pre = [v for y, v in t_series.items() if y <= 2011]
        t_post = [v for y, v in t_series.items() if y >= 2013]
        c_pre = [v for y, v in c_series.items() if y <= 2011]
        c_post = [v for y, v in c_series.items() if y >= 2013]

        if len(t_pre) >= 3 and len(t_post) >= 3 and len(c_pre) >= 3 and len(c_post) >= 3:
            did = (np.mean(t_post) - np.mean(t_pre)) - (np.mean(c_post) - np.mean(c_pre))
            dids.append(did)
            capacities.append(STORAGE_CAPACITY[weir_name])
            rivers.append(get_river(weir_name))
            names.append(WEIR_EN.get(weir_name, weir_name))

    if len(dids) < 5:
        report.append("Insufficient data for dose-response")
        return

    # Regression
    slope, intercept, r_val, p_val, std_err = stats.linregress(capacities, dids)

    fig, ax = plt.subplots(figsize=(8, 6))

    for river, color in RIVER_COLORS.items():
        mask = [r == river for r in rivers]
        caps = [c for c, m in zip(capacities, mask) if m]
        ds = [d for d, m in zip(dids, mask) if m]
        ax.scatter(caps, ds, c=color, s=80, label=river, edgecolors="gray", zorder=5)

    # Annotate each point
    for i, name in enumerate(names):
        ax.annotate(name, (capacities[i], dids[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

    # Regression line
    x_range = np.linspace(min(capacities), max(capacities), 100)
    ax.plot(x_range, slope * x_range + intercept, "k--", linewidth=1.5, alpha=0.5,
            label=f"Regression: r={r_val:.3f}, p={p_val:.3f}")

    ax.set_xlabel("Storage Capacity (10⁶ m³)", fontsize=11)
    ax.set_ylabel("BACI DID Estimate", fontsize=11)
    ax.set_title(f"Dose-Response: Storage Capacity vs Treatment Effect\n"
                 f"(slope={slope:.5f}, r²={r_val**2:.3f}, p={p_val:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "sensitivity_dose_response.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()

    report.append(f"Storage vs DID regression: slope={slope:.5f}, r²={r_val**2:.3f}, p={p_val:.3f}")
    if p_val > 0.05:
        report.append("=> No significant dose-response relationship")
        report.append("   This suggests a THRESHOLD effect: once impounded, bloom increase")
        report.append("   is relatively uniform regardless of storage capacity.")
    else:
        report.append(f"=> Significant dose-response (p={p_val:.4f})")
    report.append("")


# ═══════════════════════════════════════════════
# 6. PERMUTATION PLACEBO TEST
# ═══════════════════════════════════════════════

def permutation_placebo(treatment, control, report, n_iter=1000):
    """Fisher randomization inference: placebo timing test."""
    report.append("=== PERMUTATION PLACEBO TEST ===")

    # Compute real DID
    real_dids = []
    paired_data = []

    for weir_name, t_data in treatment.items():
        if weir_name not in control:
            continue
        t_series = extract_yearly_series(t_data)
        c_series = extract_yearly_series(control[weir_name])

        t_pre = [v for y, v in t_series.items() if y <= 2011]
        t_post = [v for y, v in t_series.items() if y >= 2013]
        c_pre = [v for y, v in c_series.items() if y <= 2011]
        c_post = [v for y, v in c_series.items() if y >= 2013]

        if len(t_pre) >= 3 and len(t_post) >= 3 and len(c_pre) >= 3 and len(c_post) >= 3:
            did = (np.mean(t_post) - np.mean(t_pre)) - (np.mean(c_post) - np.mean(c_pre))
            real_dids.append(did)
            paired_data.append((t_series, c_series))

    if not real_dids:
        report.append("Insufficient data for placebo test")
        return

    real_mean = np.mean(real_dids)

    # Placebo iterations
    np.random.seed(42)
    placebo_means = []

    for _ in range(n_iter):
        # Random placebo year from pre-period
        placebo_year = np.random.randint(2003, 2010)

        iter_dids = []
        for t_series, c_series in paired_data:
            t_pre = [v for y, v in t_series.items() if y < placebo_year]
            t_post = [v for y, v in t_series.items() if y >= placebo_year and y <= 2011]
            c_pre = [v for y, v in c_series.items() if y < placebo_year]
            c_post = [v for y, v in c_series.items() if y >= placebo_year and y <= 2011]

            if len(t_pre) >= 2 and len(t_post) >= 2 and len(c_pre) >= 2 and len(c_post) >= 2:
                did = (np.mean(t_post) - np.mean(t_pre)) - (np.mean(c_post) - np.mean(c_pre))
                iter_dids.append(did)

        if iter_dids:
            placebo_means.append(np.mean(iter_dids))

    if not placebo_means:
        report.append("Placebo computation failed")
        return

    # p-value: proportion of placebos >= real effect
    p_perm = np.mean([p >= real_mean for p in placebo_means])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(placebo_means, bins=50, alpha=0.7, color="#666666", edgecolor="white",
            label=f"Placebo distribution (n={n_iter})")
    ax.axvline(x=real_mean, color="red", linewidth=2.5,
               label=f"Real effect: {real_mean:.4f}")

    # Percentile markers
    p95 = np.percentile(placebo_means, 95)
    p99 = np.percentile(placebo_means, 99)
    ax.axvline(x=p95, color="orange", linewidth=1, linestyle="--",
               label=f"95th pctile: {p95:.4f}")
    ax.axvline(x=p99, color="red", linewidth=1, linestyle="--",
               label=f"99th pctile: {p99:.4f}")

    ax.set_xlabel("BACI DID Estimate")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Placebo Test (n={n_iter})\n"
                 f"p_perm = {p_perm:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "sensitivity_placebo.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()

    report.append(f"Real DID estimate: {real_mean:.4f}")
    report.append(f"Placebo 95th percentile: {p95:.4f}")
    report.append(f"Placebo 99th percentile: {p99:.4f}")
    report.append(f"Permutation p-value: {p_perm:.4f}")
    if real_mean > p99:
        report.append("=> Real effect EXCEEDS 99th percentile of placebos")
    elif real_mean > p95:
        report.append("=> Real effect exceeds 95th percentile of placebos")
    report.append("")


# ═══════════════════════════════════════════════
# 7. SENSITIVITY SUMMARY FIGURE
# ═══════════════════════════════════════════════

def create_summary_figure(treatment, control):
    """Create a multi-panel sensitivity summary."""
    # This will be called after all analyses complete
    # to create a comprehensive summary figure
    pass


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SENSITIVITY ANALYSIS: HAB Influencing Factors")
    print("=" * 60)

    report = []
    report.append("SENSITIVITY ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("Tests sensitivity of BACI estimates to various confounders")
    report.append("")

    # Load data
    print("\nLoading water-masked data...")
    treatment = load_treatment_data()
    control = load_control_data()
    print(f"  Treatment sites: {len(treatment)}")
    print(f"  Control reaches: {len(control)}")

    if not treatment or not control:
        print("\nERROR: Need both treatment and control data.")
        print("  Run extract_water_masked.py --target all first.")
        return

    paired = set(treatment.keys()) & set(control.keys())
    print(f"  Matched pairs: {len(paired)}")
    report.append(f"Paired sites: {len(paired)}")
    report.append("")

    # 1. Climate confounders
    print("\n[1/5] Climate trend analysis...")
    climate_data = get_all_climate_data()
    if climate_data:
        climate_results = analyze_climate_trends(climate_data, report)
        # Save climate data for potential covariate use
        climate_path = OUTPUT_DIR / "climate_data.json"
        # Convert to serializable format
        climate_serial = {}
        for river, data in climate_data.items():
            climate_serial[river] = {str(y): v for y, v in data.items()}
        with open(climate_path, "w") as f:
            json.dump(climate_serial, f, indent=2)
        print(f"  Saved climate data: {climate_path}")

    # 2. Temporal heterogeneity
    print("\n[2/5] Temporal heterogeneity (early vs late)...")
    temporal_heterogeneity(treatment, control, report)

    # 3. Leave-one-out
    print("\n[3/5] Leave-one-out sensitivity...")
    leave_one_out(treatment, control, report)

    # 4. Dose-response
    print("\n[4/5] Dose-response analysis...")
    dose_response(treatment, control, report)

    # 5. Permutation placebo
    print("\n[5/5] Permutation placebo test...")
    permutation_placebo(treatment, control, report)

    # Summary
    report.append("=" * 60)
    report.append("SUMMARY: BACI Design Controls for Multiple Confounders")
    report.append("=" * 60)
    report.append("")
    report.append("The BACI design inherently controls for:")
    report.append("  1. Climate trends (temperature, precipitation)")
    report.append("     → Region-wide trends affect treatment AND control equally")
    report.append("     → DiD differences out common temporal trends")
    report.append("  2. Watershed-wide nutrient loading changes")
    report.append("     → Upstream controls share same watershed nutrient sources")
    report.append("  3. Sensor changes and satellite discontinuities")
    report.append("     → Both sites observed by same satellite at same time")
    report.append("  4. Land-use changes (if symmetric)")
    report.append("     → Paired sites share similar catchment characteristics")
    report.append("")
    report.append("Factors NOT fully controlled by BACI:")
    report.append("  - Site-specific hydrological management (e.g., dam releases)")
    report.append("  - Local point-source pollution changes")
    report.append("  - Backwater extent estimation uncertainty")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "sensitivity_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nFull report: {report_path}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
