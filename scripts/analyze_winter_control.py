"""
Winter Negative Control Analysis
=================================
Computes BACI DiD for winter (DJF) NDVI to test falsification:
  H0: No weir effect in winter (cyanobacteria inactive)

If the bloom-season DiD reflects genuine impoundment→HAB causation,
the winter DiD should be near zero and non-significant.
"""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
WINTER_DIR = BASE_DIR / "output" / "water_masked" / "winter"
WEIR_DIR = WINTER_DIR / "weirs"
CTRL_DIR = WINTER_DIR / "controls"
WEIR_FILE = BASE_DIR / "data" / "weir_inventory.json"
CONTROL_FILE = BASE_DIR / "data" / "control_reaches.json"


def load_site(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def yearly_medians(data: dict) -> dict[int, float]:
    """Extract {year: median} from pre+post lists, skipping None."""
    out = {}
    for period_key in ["landsat_pre_weir", "landsat_post_weir"]:
        for entry in data.get(period_key, []):
            if entry.get("median") is not None:
                out[entry["year"]] = entry["median"]
    return out


def compute_baci_did(t_series: dict, c_series: dict,
                     pre_range=(2001, 2012), post_range=(2014, 2025)):
    """Compute BACI DiD = (T_post - T_pre) - (C_post - C_pre)."""
    t_pre = [t_series[y] for y in range(pre_range[0], pre_range[1] + 1) if y in t_series]
    t_post = [t_series[y] for y in range(post_range[0], post_range[1] + 1) if y in t_series]
    c_pre = [c_series[y] for y in range(pre_range[0], pre_range[1] + 1) if y in c_series]
    c_post = [c_series[y] for y in range(post_range[0], post_range[1] + 1) if y in c_series]

    if len(t_pre) < 3 or len(t_post) < 3 or len(c_pre) < 3 or len(c_post) < 3:
        return None

    did = (np.mean(t_post) - np.mean(t_pre)) - (np.mean(c_post) - np.mean(c_pre))

    # Welch t-test on year-matched treatment-control differences
    pre_diffs = []
    post_diffs = []
    for y in range(pre_range[0], pre_range[1] + 1):
        if y in t_series and y in c_series:
            pre_diffs.append(t_series[y] - c_series[y])
    for y in range(post_range[0], post_range[1] + 1):
        if y in t_series and y in c_series:
            post_diffs.append(t_series[y] - c_series[y])

    if len(pre_diffs) >= 3 and len(post_diffs) >= 3:
        t_stat, p_val = stats.ttest_ind(post_diffs, pre_diffs, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        "t_pre_mean": np.mean(t_pre),
        "t_post_mean": np.mean(t_post),
        "c_pre_mean": np.mean(c_pre),
        "c_post_mean": np.mean(c_post),
        "did": did,
        "t_stat": t_stat,
        "p_value": p_val,
        "n_pre_diffs": len(pre_diffs),
        "n_post_diffs": len(post_diffs),
    }


def main():
    # Load weir-control mapping
    with open(WEIR_FILE) as f:
        weirs = json.load(f)
    with open(CONTROL_FILE) as f:
        controls = json.load(f)

    # Build control lookup: weir_name_en -> control_name
    # Control file uses short names (e.g. "Ipo"), weir file uses "Ipo Weir"
    # Also handle compound names: "Gangjeong" matches "Gangjeong-Goryeong Weir"
    ctrl_lookup = {}
    for c in controls:
        key = c["weir_name_en"]
        ctrl_lookup[key] = c["control_name"]
        ctrl_lookup[key + " Weir"] = c["control_name"]

    def find_control(weir_name: str) -> str | None:
        if weir_name in ctrl_lookup:
            return ctrl_lookup[weir_name]
        # Try stripping " Weir" and matching first part
        base = weir_name.replace(" Weir", "")
        if base in ctrl_lookup:
            return ctrl_lookup[base]
        # Try first part of compound name (e.g. "Gangjeong-Goryeong" -> "Gangjeong")
        first = base.split("-")[0]
        if first in ctrl_lookup:
            return ctrl_lookup[first]
        return None

    results = []
    all_dids = []

    print(f"\n{'=' * 70}")
    print("WINTER NEGATIVE CONTROL — BACI DiD ANALYSIS")
    print(f"{'=' * 70}\n")

    for w in weirs:
        name = w.get("name_en", w.get("weir_name_en", ""))
        ctrl_name = find_control(name)
        if not ctrl_name:
            print(f"  SKIP {name}: no control site mapped")
            continue

        t_data = load_site(WEIR_DIR / f"{name}_winter.json")
        c_data = load_site(CTRL_DIR / f"{ctrl_name}_winter.json")

        if t_data is None:
            print(f"  SKIP {name}: no winter weir data")
            continue
        if c_data is None:
            print(f"  SKIP {name}: no winter control data ({ctrl_name})")
            continue

        t_series = yearly_medians(t_data)
        c_series = yearly_medians(c_data)

        did_result = compute_baci_did(t_series, c_series)
        if did_result is None:
            print(f"  SKIP {name}: insufficient years")
            continue

        sig = "***" if did_result["p_value"] < 0.001 else \
              "**" if did_result["p_value"] < 0.01 else \
              "*" if did_result["p_value"] < 0.05 else "ns"

        print(f"  {name:30s}  DiD={did_result['did']:+.4f}  "
              f"t={did_result['t_stat']:+.2f}  p={did_result['p_value']:.3f}  {sig}")

        results.append({"weir": name, "control": ctrl_name, **did_result})
        all_dids.append(did_result["did"])

    # Pooled statistics
    print(f"\n{'=' * 70}")
    if len(all_dids) >= 2:
        pooled_mean = np.mean(all_dids)
        pooled_se = np.std(all_dids, ddof=1) / np.sqrt(len(all_dids))
        t_stat_pool, p_pool = stats.ttest_1samp(all_dids, 0)

        print(f"POOLED WINTER DiD (n={len(all_dids)} pairs)")
        print(f"  Mean DiD:  {pooled_mean:+.4f}")
        print(f"  SE:        {pooled_se:.4f}")
        print(f"  t-stat:    {t_stat_pool:+.3f}")
        print(f"  p-value:   {p_pool:.4f}")
        print(f"  Significant (p<0.05): {'YES ⚠️' if p_pool < 0.05 else 'NO ✓'}")
        sig_count = sum(1 for r in results if r["p_value"] < 0.05)
        print(f"  Individual significant: {sig_count}/{len(results)}")
    else:
        print("Insufficient pairs for pooled analysis")
        pooled_mean = p_pool = None

    # Save results
    out_path = WINTER_DIR / "winter_did_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "pairs": results,
            "pooled": {
                "mean_did": float(pooled_mean) if pooled_mean is not None else None,
                "se": float(pooled_se) if pooled_mean is not None else None,
                "t_stat": float(t_stat_pool) if pooled_mean is not None else None,
                "p_value": float(p_pool) if p_pool is not None else None,
                "n_pairs": len(all_dids),
            }
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Also save CSV
    csv_path = WINTER_DIR / "winter_did_summary.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV saved: {csv_path}")

    return results


if __name__ == "__main__":
    main()
