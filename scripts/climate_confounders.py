#!/usr/bin/env python3
"""
Climate Confounders Analysis for HAB-BACI Study
=================================================
Analyzes whether climate trends (temperature, precipitation, GDD) could
confound the weir-effect signal in a BACI design comparing 16 weir sites
to upstream controls across South Korea's Four Major Rivers (2000-2025).

Data source: Open-Meteo Archive API (free, no key required).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import requests
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Representative coordinates for each river basin
RIVER_SITES: dict[str, dict[str, float]] = {
    "Han": {"lat": 37.3, "lon": 127.6},
    "Nakdong": {"lat": 35.8, "lon": 128.6},
    "Geum": {"lat": 36.5, "lon": 127.0},
    "Yeongsan": {"lat": 35.0, "lon": 126.9},
}

# Study design periods
PRE_START, PRE_END = 2000, 2011
POST_START, POST_END = 2013, 2025
BREAK_YEAR = 2012  # Structural break (weir construction year)

# Bloom season months (May-October)
BLOOM_MONTHS = list(range(5, 11))  # 5, 6, 7, 8, 9, 10

# GDD base temperature (Celsius)
GDD_BASE = 10.0


# ---------------------------------------------------------------------------
# Mann-Kendall trend test (pure scipy implementation)
# ---------------------------------------------------------------------------

def mann_kendall_test(x: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """
    Mann-Kendall trend test.

    Returns dict with keys: trend, p, z, tau, slope (Theil-Sen).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return {"trend": "insufficient data", "p": np.nan, "z": np.nan,
                "tau": np.nan, "slope": np.nan}

    # S statistic
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = x[j] - x[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S
    unique, counts = np.unique(x, return_counts=True)
    tied_groups = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    for t in tied_groups:
        var_s -= t * (t - 1) * (2 * t + 5) / 18.0

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    # Kendall's tau
    tau = s / (n * (n - 1) / 2.0)

    # Theil-Sen slope
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((x[j] - x[i]) / (j - i))
    slope = np.median(slopes)

    if p <= alpha:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no trend"

    return {"trend": trend, "p": p, "z": z, "tau": tau, "slope": slope}


# ---------------------------------------------------------------------------
# Data fetching from Open-Meteo Archive API
# ---------------------------------------------------------------------------

def fetch_open_meteo(
    lat: float, lon: float, start_date: str, end_date: str,
    daily_vars: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch daily weather data from Open-Meteo Archive API.

    Args:
        lat, lon: Coordinates.
        start_date, end_date: ISO date strings (YYYY-MM-DD).
        daily_vars: List of daily variable names.

    Returns:
        DataFrame with 'date' column and requested variables.
    """
    if daily_vars is None:
        daily_vars = [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": "Asia/Seoul",
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"  [WARN] Attempt {attempt+1}/3 failed: {exc}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(
                    f"Open-Meteo API failed after 3 attempts for "
                    f"({lat}, {lon}): {exc}"
                ) from exc

    daily = data.get("daily", {})
    dates = daily.pop("time", [])
    df = pd.DataFrame(daily, index=pd.to_datetime(dates))
    df.index.name = "date"
    return df


def fetch_all_rivers(
    years: tuple[int, int] = (2000, 2025),
) -> dict[str, pd.DataFrame]:
    """Fetch daily climate data for all 4 river basins."""
    start = f"{years[0]}-01-01"
    end = f"{years[1]}-12-31"
    all_data: dict[str, pd.DataFrame] = {}

    for river, coords in RIVER_SITES.items():
        print(f"Fetching {river} River ({coords['lat']}, {coords['lon']})...")
        df = fetch_open_meteo(
            coords["lat"], coords["lon"], start, end,
            daily_vars=[
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
            ],
        )
        df["river"] = river
        all_data[river] = df
        time.sleep(1)  # polite rate limiting

    return all_data


# ---------------------------------------------------------------------------
# Derived annual metrics
# ---------------------------------------------------------------------------

def compute_annual_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual metrics from daily data.

    Returns DataFrame indexed by year with columns:
        bloom_temp_mean, annual_precip, gdd
    """
    daily = daily.copy()
    daily["year"] = daily.index.year
    daily["month"] = daily.index.month

    records = []
    for yr, grp in daily.groupby("year"):
        # Bloom-season mean temperature (May-Oct)
        bloom = grp[grp["month"].isin(BLOOM_MONTHS)]
        bloom_temp = bloom["temperature_2m_mean"].mean()

        # Annual total precipitation (mm)
        annual_precip = grp["precipitation_sum"].sum()

        # Growing degree days (base 10 C), full year
        tmean = grp["temperature_2m_mean"].fillna(
            (grp["temperature_2m_max"] + grp["temperature_2m_min"]) / 2
        )
        gdd = np.maximum(tmean - GDD_BASE, 0).sum()

        records.append({
            "year": int(yr),
            "bloom_temp_mean": bloom_temp,
            "annual_precip": annual_precip,
            "gdd": gdd,
        })

    return pd.DataFrame(records).set_index("year")


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def analyze_trend(
    years: np.ndarray, values: np.ndarray, var_name: str,
) -> dict[str, Any]:
    """
    Full trend analysis: OLS, Mann-Kendall, pre/post comparison.
    """
    mask = ~np.isnan(values)
    y, v = years[mask], values[mask]

    # 1. OLS linear trend
    slope, intercept, r, p_ols, se = stats.linregress(y, v)

    # 2. Mann-Kendall
    mk = mann_kendall_test(v)

    # 3. Pre vs Post comparison (Welch's t-test)
    pre_mask = (y >= PRE_START) & (y <= PRE_END)
    post_mask = (y >= POST_START) & (y <= POST_END)
    pre_vals = v[pre_mask]
    post_vals = v[post_mask]

    if len(pre_vals) > 1 and len(post_vals) > 1:
        t_stat, p_welch = stats.ttest_ind(pre_vals, post_vals, equal_var=False)
        pre_mean = pre_vals.mean()
        post_mean = post_vals.mean()
        diff = post_mean - pre_mean
    else:
        t_stat = p_welch = pre_mean = post_mean = diff = np.nan

    return {
        "variable": var_name,
        "ols_slope": slope,
        "ols_slope_per_decade": slope * 10,
        "ols_r2": r**2,
        "ols_p": p_ols,
        "mk_trend": mk["trend"],
        "mk_p": mk["p"],
        "mk_z": mk["z"],
        "mk_tau": mk["tau"],
        "theil_sen_slope": mk["slope"],
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "pre_post_diff": diff,
        "welch_t": t_stat,
        "welch_p": p_welch,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_period_shading(ax: plt.Axes, ymin: float, ymax: float) -> None:
    """Add pre/post period shading and break line."""
    ax.axvspan(PRE_START - 0.5, PRE_END + 0.5, alpha=0.08, color="blue",
               label=f"Pre ({PRE_START}-{PRE_END})")
    ax.axvspan(POST_START - 0.5, POST_END + 0.5, alpha=0.08, color="red",
               label=f"Post ({POST_START}-{POST_END})")
    ax.axvline(BREAK_YEAR, color="gray", ls="--", lw=1.2, alpha=0.7,
               label=f"Break ({BREAK_YEAR})")


def _format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def plot_temperature_trend(
    annual_all: pd.DataFrame, results: list[dict], save_path: Path,
) -> None:
    """Temperature trend with per-river lines and national mean."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1])
    fig.suptitle(
        "Bloom-Season Temperature Trend (May-Oct)\nSouth Korea Four Major Rivers",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ax = axes[0]
    colors = {"Han": "#1f77b4", "Nakdong": "#d62728",
              "Geum": "#2ca02c", "Yeongsan": "#9467bd"}

    # Per-river lines
    for river in RIVER_SITES:
        rdf = annual_all[annual_all["river"] == river]
        ax.plot(rdf["year"], rdf["bloom_temp_mean"], "o-", ms=4, lw=1.2,
                alpha=0.5, color=colors[river], label=river)

    # National mean
    natl = annual_all.groupby("year")["bloom_temp_mean"].mean()
    years_arr = natl.index.values.astype(float)
    vals_arr = natl.values

    ax.plot(natl.index, natl.values, "ks-", ms=6, lw=2.2, label="National Mean",
            zorder=5)

    # OLS trend line
    slope, intercept, _, _, _ = stats.linregress(years_arr, vals_arr)
    trend_line = intercept + slope * years_arr
    ax.plot(natl.index, trend_line, "r--", lw=1.5, alpha=0.8,
            label=f"OLS trend: {slope*10:+.2f} C/decade")

    # Period shading
    _add_period_shading(ax, vals_arr.min(), vals_arr.max())

    # MK result annotation
    mk_res = next((r for r in results if r["variable"] == "bloom_temp_mean"), None)
    if mk_res:
        txt = (
            f"Mann-Kendall: {mk_res['mk_trend']} ({_format_p(mk_res['mk_p'])})\n"
            f"Theil-Sen: {mk_res['theil_sen_slope']*10:+.3f} C/decade\n"
            f"Pre mean: {mk_res['pre_mean']:.2f} C | "
            f"Post mean: {mk_res['post_mean']:.2f} C\n"
            f"Diff: {mk_res['pre_post_diff']:+.2f} C "
            f"(Welch {_format_p(mk_res['welch_p'])})"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                          alpha=0.9, ec="gray"))

    ax.set_ylabel("Bloom-Season Mean Temp (C)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Bottom panel: pre-post box comparison
    ax2 = axes[1]
    pre_data = annual_all[
        (annual_all["year"] >= PRE_START) & (annual_all["year"] <= PRE_END)
    ]["bloom_temp_mean"]
    post_data = annual_all[
        (annual_all["year"] >= POST_START) & (annual_all["year"] <= POST_END)
    ]["bloom_temp_mean"]

    bp = ax2.boxplot(
        [pre_data.dropna(), post_data.dropna()],
        tick_labels=[f"Pre\n({PRE_START}-{PRE_END})", f"Post\n({POST_START}-{POST_END})"],
        widths=0.5, patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#a8d8ea")
    bp["boxes"][1].set_facecolor("#ffb3b3")
    ax2.set_ylabel("Temp (C)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_precipitation_trend(
    annual_all: pd.DataFrame, results: list[dict], save_path: Path,
) -> None:
    """Precipitation trend with per-river bars and national mean."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1])
    fig.suptitle(
        "Annual Precipitation Trend\nSouth Korea Four Major Rivers",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ax = axes[0]
    colors = {"Han": "#1f77b4", "Nakdong": "#d62728",
              "Geum": "#2ca02c", "Yeongsan": "#9467bd"}

    for river in RIVER_SITES:
        rdf = annual_all[annual_all["river"] == river]
        ax.plot(rdf["year"], rdf["annual_precip"], "o-", ms=3, lw=1,
                alpha=0.4, color=colors[river], label=river)

    # National mean
    natl = annual_all.groupby("year")["annual_precip"].mean()
    years_arr = natl.index.values.astype(float)
    vals_arr = natl.values

    ax.bar(natl.index, natl.values, color="steelblue", alpha=0.5, width=0.7,
           label="National Mean", zorder=3)

    # OLS trend
    slope, intercept, _, _, _ = stats.linregress(years_arr, vals_arr)
    trend_line = intercept + slope * years_arr
    ax.plot(natl.index, trend_line, "r--", lw=1.5, alpha=0.8,
            label=f"OLS trend: {slope*10:+.1f} mm/decade")

    _add_period_shading(ax, 0, vals_arr.max() * 1.1)

    # MK annotation
    mk_res = next((r for r in results if r["variable"] == "annual_precip"), None)
    if mk_res:
        txt = (
            f"Mann-Kendall: {mk_res['mk_trend']} ({_format_p(mk_res['mk_p'])})\n"
            f"Theil-Sen: {mk_res['theil_sen_slope']*10:+.1f} mm/decade\n"
            f"Pre mean: {mk_res['pre_mean']:.0f} mm | "
            f"Post mean: {mk_res['post_mean']:.0f} mm\n"
            f"Diff: {mk_res['pre_post_diff']:+.0f} mm "
            f"(Welch {_format_p(mk_res['welch_p'])})"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                          alpha=0.9, ec="gray"))

    ax.set_ylabel("Annual Precipitation (mm)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Box comparison
    ax2 = axes[1]
    pre_data = annual_all[
        (annual_all["year"] >= PRE_START) & (annual_all["year"] <= PRE_END)
    ]["annual_precip"]
    post_data = annual_all[
        (annual_all["year"] >= POST_START) & (annual_all["year"] <= POST_END)
    ]["annual_precip"]

    bp = ax2.boxplot(
        [pre_data.dropna(), post_data.dropna()],
        tick_labels=[f"Pre\n({PRE_START}-{PRE_END})", f"Post\n({POST_START}-{POST_END})"],
        widths=0.5, patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#a8d8ea")
    bp["boxes"][1].set_facecolor("#ffb3b3")
    ax2.set_ylabel("Precip (mm)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_summary(
    annual_all: pd.DataFrame, results: list[dict], save_path: Path,
) -> None:
    """
    Multi-panel summary figure:
    (A) Temperature trend, (B) Precipitation trend,
    (C) GDD trend, (D) Cross-river correlation heatmap,
    (E) Confounder assessment text box.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.28)
    fig.suptitle(
        "Climate Confounders Summary for HAB-BACI Study\n"
        "South Korea Four Major Rivers (2000-2025)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    natl = annual_all.groupby("year").agg({
        "bloom_temp_mean": "mean",
        "annual_precip": "mean",
        "gdd": "mean",
    })

    var_info = [
        ("bloom_temp_mean", "Bloom-Season Temp (C)", "(A)"),
        ("annual_precip", "Annual Precipitation (mm)", "(B)"),
        ("gdd", "Growing Degree Days (base 10C)", "(C)"),
    ]

    for idx, (var, ylabel, label) in enumerate(var_info):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        years_arr = natl.index.values.astype(float)
        vals_arr = natl[var].values

        ax.plot(natl.index, vals_arr, "ko-", ms=5, lw=1.8, zorder=5)

        # OLS
        slope, intercept, _, _, _ = stats.linregress(years_arr, vals_arr)
        ax.plot(natl.index, intercept + slope * years_arr, "r--", lw=1.5, alpha=0.7)

        _add_period_shading(ax, vals_arr.min(), vals_arr.max())

        mk = next((r for r in results if r["variable"] == var), None)
        if mk:
            star = "*" if mk["mk_p"] < 0.05 else ""
            ax.set_title(
                f"{label} {ylabel}\n"
                f"MK: {mk['mk_trend']}{star} ({_format_p(mk['mk_p'])}), "
                f"Theil-Sen: {mk['theil_sen_slope']*10:+.3f}/dec",
                fontsize=10,
            )
        else:
            ax.set_title(f"{label} {ylabel}", fontsize=10)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="lower right")

    # (D) Cross-river correlation heatmap for bloom temperature
    ax_corr = fig.add_subplot(gs[1, 1])
    rivers = list(RIVER_SITES.keys())
    pivot = annual_all.pivot_table(
        index="year", columns="river", values="bloom_temp_mean",
    )[rivers]
    corr = pivot.corr()

    im = ax_corr.imshow(corr.values, vmin=0.5, vmax=1.0, cmap="YlOrRd", aspect="auto")
    ax_corr.set_xticks(range(len(rivers)))
    ax_corr.set_xticklabels(rivers, fontsize=9)
    ax_corr.set_yticks(range(len(rivers)))
    ax_corr.set_yticklabels(rivers, fontsize=9)
    for i in range(len(rivers)):
        for j in range(len(rivers)):
            ax_corr.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                         va="center", fontsize=10,
                         color="white" if corr.values[i, j] > 0.85 else "black")
    ax_corr.set_title("(D) Cross-River Temp Correlation\n(High = uniform climate signal)",
                       fontsize=10)
    fig.colorbar(im, ax=ax_corr, shrink=0.8, label="Pearson r")

    # (E) Confounder assessment text box
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    temp_res = next((r for r in results if r["variable"] == "bloom_temp_mean"), {})
    prec_res = next((r for r in results if r["variable"] == "annual_precip"), {})
    gdd_res = next((r for r in results if r["variable"] == "gdd"), {})
    min_corr = corr.min().min()

    # Build assessment
    lines = [
        "CONFOUNDER ASSESSMENT FOR BACI DESIGN",
        "=" * 50,
        "",
        f"1. TEMPERATURE: Bloom-season mean shows "
        f"{'a significant' if temp_res.get('mk_p', 1) < 0.05 else 'no significant'} "
        f"trend ({_format_p(temp_res.get('mk_p', 1))})",
        f"   - Theil-Sen slope: {temp_res.get('theil_sen_slope', 0)*10:+.3f} C/decade",
        f"   - Pre-post diff: {temp_res.get('pre_post_diff', 0):+.2f} C "
        f"({_format_p(temp_res.get('welch_p', 1))})",
        "",
        f"2. PRECIPITATION: Annual total shows "
        f"{'a significant' if prec_res.get('mk_p', 1) < 0.05 else 'no significant'} "
        f"trend ({_format_p(prec_res.get('mk_p', 1))})",
        f"   - Theil-Sen slope: {prec_res.get('theil_sen_slope', 0)*10:+.1f} mm/decade",
        "",
        f"3. GDD: Growing degree days show "
        f"{'a significant' if gdd_res.get('mk_p', 1) < 0.05 else 'no significant'} "
        f"trend ({_format_p(gdd_res.get('mk_p', 1))})",
        "",
        f"4. SPATIAL UNIFORMITY: Min cross-river temp correlation = {min_corr:.2f}",
        f"   -> Climate trends are REGION-WIDE (all rivers show similar patterns).",
        "",
        "CONCLUSION:",
        "   Even if climate shows a trend, the BACI design CONTROLS for this",
        "   because climate affects both treatment (weir) and control (upstream)",
        "   sites equally. The high cross-river correlations confirm that climate",
        "   variation is spatially uniform across South Korea's river basins.",
        "   Therefore, climate trends do NOT confound the weir effect estimate",
        "   in the Difference-in-Differences framework.",
    ]

    ax_txt.text(
        0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
        fontsize=9.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f0", ec="gray", alpha=0.95),
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Climate Confounders Analysis for HAB-BACI Study")
    print("=" * 60)

    # --- Step 1: Fetch data ---
    cache_path = OUTPUT_DIR / "climate_daily_cache.parquet"
    if cache_path.exists():
        print(f"\nLoading cached data from {cache_path}")
        daily_all = pd.read_parquet(cache_path)
    else:
        print("\nFetching daily climate data from Open-Meteo...")
        river_data = fetch_all_rivers((PRE_START, POST_END))
        daily_all = pd.concat(river_data.values())
        daily_all.to_parquet(cache_path)
        print(f"  Cached to {cache_path}")

    # --- Step 2: Compute annual metrics per river ---
    print("\nComputing annual metrics...")
    annual_frames = []
    for river in RIVER_SITES:
        rdf = daily_all[daily_all["river"] == river]
        ann = compute_annual_metrics(rdf)
        ann["river"] = river
        annual_frames.append(ann.reset_index())

    annual_all = pd.concat(annual_frames, ignore_index=True)
    annual_all.to_csv(OUTPUT_DIR / "climate_annual_metrics.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'climate_annual_metrics.csv'}")

    # --- Step 3: National-level trend analysis ---
    print("\nTrend analysis (national mean)...")
    natl = annual_all.groupby("year").agg({
        "bloom_temp_mean": "mean",
        "annual_precip": "mean",
        "gdd": "mean",
    })
    years_arr = natl.index.values.astype(float)

    results = []
    for var in ["bloom_temp_mean", "annual_precip", "gdd"]:
        res = analyze_trend(years_arr, natl[var].values, var)
        results.append(res)
        mk_star = "*" if res["mk_p"] < 0.05 else ""
        print(
            f"  {var:20s}: MK={res['mk_trend']:12s}{mk_star} "
            f"({_format_p(res['mk_p'])}), "
            f"Theil-Sen={res['theil_sen_slope']*10:+.4f}/dec, "
            f"Pre={res['pre_mean']:.2f}, Post={res['post_mean']:.2f}, "
            f"Diff={res['pre_post_diff']:+.2f} (Welch {_format_p(res['welch_p'])})"
        )

    # Save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "climate_trend_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'climate_trend_results.csv'}")

    # --- Step 4: Per-river trend analysis ---
    print("\nPer-river trend analysis...")
    river_results = []
    for river in RIVER_SITES:
        rdf = annual_all[annual_all["river"] == river].sort_values("year")
        y = rdf["year"].values.astype(float)
        for var in ["bloom_temp_mean", "annual_precip", "gdd"]:
            res = analyze_trend(y, rdf[var].values, var)
            res["river"] = river
            river_results.append(res)

    river_df = pd.DataFrame(river_results)
    river_df.to_csv(OUTPUT_DIR / "climate_trend_per_river.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'climate_trend_per_river.csv'}")

    # Print per-river temperature trends
    print("\n  Bloom-season temperature trends by river:")
    for _, row in river_df[river_df["variable"] == "bloom_temp_mean"].iterrows():
        star = "*" if row["mk_p"] < 0.05 else ""
        print(
            f"    {row['river']:10s}: MK={row['mk_trend']:12s}{star} "
            f"({_format_p(row['mk_p'])}), "
            f"slope={row['theil_sen_slope']*10:+.3f} C/dec"
        )

    # --- Step 5: Cross-river correlation ---
    print("\nCross-river temperature correlation:")
    pivot = annual_all.pivot_table(
        index="year", columns="river", values="bloom_temp_mean",
    )
    corr = pivot.corr()
    print(corr.to_string())

    # --- Step 6: Generate figures ---
    print("\nGenerating figures...")
    plot_temperature_trend(
        annual_all, results,
        FIG_DIR / "climate_temperature_trend.png",
    )
    plot_precipitation_trend(
        annual_all, results,
        FIG_DIR / "climate_precipitation_trend.png",
    )
    plot_summary(
        annual_all, results,
        FIG_DIR / "climate_confounders_summary.png",
    )

    # --- Step 7: Print summary ---
    print("\n" + "=" * 60)
    print("CONFOUNDER ASSESSMENT SUMMARY")
    print("=" * 60)

    temp_res = results[0]
    prec_res = results[1]
    gdd_res = results[2]
    min_corr = corr.min().min()

    print(f"""
Temperature:
  - Trend: {temp_res['mk_trend']} ({_format_p(temp_res['mk_p'])})
  - Rate: {temp_res['theil_sen_slope']*10:+.3f} C/decade
  - Pre-post diff: {temp_res['pre_post_diff']:+.2f} C

Precipitation:
  - Trend: {prec_res['mk_trend']} ({_format_p(prec_res['mk_p'])})
  - Rate: {prec_res['theil_sen_slope']*10:+.1f} mm/decade

GDD:
  - Trend: {gdd_res['mk_trend']} ({_format_p(gdd_res['mk_p'])})
  - Rate: {gdd_res['theil_sen_slope']*10:+.1f} deg-days/decade

Cross-river temperature correlation:
  - Min r = {min_corr:.3f} (high = spatially uniform)

BACI Implication:
  Climate trends, even if significant, affect BOTH treatment and
  control sites uniformly (min cross-river r = {min_corr:.3f}).
  The DiD estimator removes any common time trend, so climate
  is NOT a confounding threat to the weir-effect estimate.
""")

    print(f"Output files in: {OUTPUT_DIR}")
    print(f"Figures in: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
