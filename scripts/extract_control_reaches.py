"""
Extract bloom proxy time series for UPSTREAM CONTROL REACHES.
=============================================================
Mirrors extract_all_weirs.py but for 16 upstream reference points
(15-20 km above each weir, beyond backwater influence zone).

These control reaches serve as the "untreated" arm of the
Before-After-Control-Impact (BACI) design.

Sensors:
  - Landsat 5/7 (2000-2012, pre-weir baseline)
  - Landsat 8/9 (2013-2025, post-weir)
  - Sentinel-2   (2017-2025, true NDCI)

Output:
  - Per-control JSON  -> output/control_data/<name>_control.json
  - Combined CSV      -> output/control_reaches_bloom_summary.csv

Usage:
    python extract_control_reaches.py
    python extract_control_reaches.py --test          # first 2 only
    python extract_control_reaches.py --control Ipo_upstream
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import ee

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CONTROL_PATH = BASE_DIR / "control_reaches.json"
OUTPUT_DIR = BASE_DIR / "output" / "control_data"
COMBINED_CSV = BASE_DIR / "output" / "control_reaches_bloom_summary.csv"

# ── GEE Auth ─────��─────────────────────────────────────────────
SERVICE_ACCOUNT = "your-service-account@project.iam.gserviceaccount.com"
KEY_PATH = (
    "your-gee-service-key.json"  # Replace with your GEE service key path
    
)
GEE_PROJECT = "your-gee-project-id"

# ── Constants ──────────────────────────────────────────────────
BUFFER_M = 5000          # Same 5 km buffer as weir sites
BLOOM_MONTHS = (5, 10)   # May-October inclusive
LANDSAT_PRE_RANGE = (2000, 2012)
LANDSAT_POST_RANGE = (2013, 2025)
SENTINEL2_RANGE = (2017, 2025)
SCALE_LANDSAT = 30
SCALE_S2 = 20
MAX_RETRIES = 1


# ── GEE Initialization ────────────────────────────────────────

def init_gee() -> None:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
    ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    print("[GEE] Initialized successfully.")


# ── Cloud Masking ──────────────────────────────────────────────

def mask_landsat(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)


def mask_s2(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask)


# ── Index Computation ──────────────────────────────────────────
# Landsat: NDVI applied to water = (NIR-Red)/(NIR+Red)
# Sentinel-2: true NDCI = (B5-B4)/(B5+B4)
# Both stored as band "BLOOM_PROXY" for unified pipeline

def add_bloom_proxy_landsat57(image: ee.Image) -> ee.Image:
    """Landsat 5/7: NDVI = (NIR - Red) / (NIR + Red)."""
    red = image.select("SR_B3").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndvi)


def add_bloom_proxy_landsat89(image: ee.Image) -> ee.Image:
    """Landsat 8/9: NDVI = (NIR - Red) / (NIR + Red)."""
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndvi)


def add_bloom_proxy_s2(image: ee.Image) -> ee.Image:
    """Sentinel-2: true NDCI = (B5 - B4) / (B5 + B4)."""
    red = image.select("B4").multiply(0.0001)
    re1 = image.select("B5").multiply(0.0001)
    ndci = re1.subtract(red).divide(re1.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndci)


# ── Collection Builders ───────────────────────────────────────

def _bloom_filter(col: ee.ImageCollection, year: int) -> ee.ImageCollection:
    return (col
            .filterDate(f"{year}-05-01", f"{year}-11-01")
            .filter(ee.Filter.calendarRange(BLOOM_MONTHS[0], BLOOM_MONTHS[1], "month")))


def get_landsat57_col(geom: ee.Geometry, year: int) -> ee.ImageCollection:
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat57))
    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat57))
    return _bloom_filter(l5.merge(l7), year)


def get_landsat89_col(geom: ee.Geometry, year: int) -> ee.ImageCollection:
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat89))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat89))
    return _bloom_filter(l8.merge(l9), year)


def get_s2_col(geom: ee.Geometry, year: int) -> ee.ImageCollection:
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(geom)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .map(mask_s2)
           .map(add_bloom_proxy_s2))
    return _bloom_filter(col, year)


# ── Stats Extraction ───���──────────────────────────────────────

def _combined_reducer() -> ee.Reducer:
    return (ee.Reducer.median()
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True))


def extract_stats(col: ee.ImageCollection, geom: ee.Geometry,
                  scale: int) -> dict:
    composite = col.select("BLOOM_PROXY").reduce(_combined_reducer())
    raw = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e8,
    ).getInfo()
    return raw


def parse_stats(raw: dict | None) -> dict:
    if raw is None:
        return {"median": None, "mean": None, "std": None, "count": 0}
    return {
        "median": raw.get("BLOOM_PROXY_median"),
        "mean":   raw.get("BLOOM_PROXY_mean"),
        "std":    raw.get("BLOOM_PROXY_stdDev"),
        "count":  int(raw.get("BLOOM_PROXY_count") or 0),
    }


# ── Per-Control Extraction ─────────────────────────────────────

def extract_one_control(ctrl: dict) -> dict:
    """Extract full time series for a single control reach."""
    name = ctrl["control_name"]
    lat, lon = ctrl["lat"], ctrl["lon"]
    point = ee.Geometry.Point(lon, lat)
    reach_geom = point.buffer(BUFFER_M)

    result = {
        "control_name": name,
        "weir_name_kr": ctrl["weir_name_kr"],
        "weir_name_en": ctrl["weir_name_en"],
        "lat": lat,
        "lon": lon,
        "buffer_m": BUFFER_M,
        "distance_km": ctrl["distance_km"],
        "landsat_pre_weir": [],
        "landsat_post_weir": [],
        "sentinel2": [],
    }

    # ── Landsat 5/7: pre-weir ──
    for year in range(LANDSAT_PRE_RANGE[0], LANDSAT_PRE_RANGE[1] + 1):
        try:
            col = get_landsat57_col(reach_geom, year)
            raw = extract_stats(col, reach_geom, SCALE_LANDSAT)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Landsat 5/7"
            result["landsat_pre_weir"].append(entry)
        except Exception as exc:
            print(f"    WARN {name} L57 {year}: {exc}")
            result["landsat_pre_weir"].append(
                {"year": year, "sensor": "Landsat 5/7",
                 "median": None, "mean": None, "std": None, "count": 0})

    # ── Landsat 8/9: post-weir ──
    for year in range(LANDSAT_POST_RANGE[0], LANDSAT_POST_RANGE[1] + 1):
        try:
            col = get_landsat89_col(reach_geom, year)
            raw = extract_stats(col, reach_geom, SCALE_LANDSAT)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Landsat 8/9"
            result["landsat_post_weir"].append(entry)
        except Exception as exc:
            print(f"    WARN {name} L89 {year}: {exc}")
            result["landsat_post_weir"].append(
                {"year": year, "sensor": "Landsat 8/9",
                 "median": None, "mean": None, "std": None, "count": 0})

    # ── Sentinel-2 ──
    for year in range(SENTINEL2_RANGE[0], SENTINEL2_RANGE[1] + 1):
        try:
            col = get_s2_col(reach_geom, year)
            raw = extract_stats(col, reach_geom, SCALE_S2)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Sentinel-2"
            result["sentinel2"].append(entry)
        except Exception as exc:
            print(f"    WARN {name} S2 {year}: {exc}")
            result["sentinel2"].append(
                {"year": year, "sensor": "Sentinel-2",
                 "median": None, "mean": None, "std": None, "count": 0})

    return result


# ── Main ──────────────────────────────────────────────────────

def main():
    init_gee()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONTROL_PATH) as f:
        controls = json.load(f)

    # CLI filtering
    test_mode = "--test" in sys.argv
    single = None
    if "--control" in sys.argv:
        idx = sys.argv.index("--control")
        single = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    if test_mode:
        controls = controls[:2]
        print(f"[TEST MODE] Processing first {len(controls)} control reaches only.\n")
    elif single:
        controls = [c for c in controls if c["control_name"] == single
                    or c["weir_name_kr"] == single]
        if not controls:
            print(f"Control reach '{single}' not found.")
            sys.exit(1)

    all_results = []
    csv_rows = []

    for i, ctrl in enumerate(controls, 1):
        name = ctrl["control_name"]
        print(f"\n[{i}/{len(controls)}] {name} ({ctrl['weir_name_kr']} upstream) "
              f"({ctrl['lat']:.4f}, {ctrl['lon']:.4f})")

        t0 = time.time()
        try:
            result = extract_one_control(ctrl)
            all_results.append(result)

            # Save per-control JSON
            out_path = OUTPUT_DIR / f"{name}.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_path}")

            # Collect CSV rows
            for period_key in ["landsat_pre_weir", "landsat_post_weir", "sentinel2"]:
                period_label = {
                    "landsat_pre_weir": "pre",
                    "landsat_post_weir": "post",
                    "sentinel2": "s2",
                }[period_key]
                for entry in result[period_key]:
                    csv_rows.append({
                        "control_name": name,
                        "weir_name_kr": ctrl["weir_name_kr"],
                        "year": entry["year"],
                        "sensor": entry["sensor"],
                        "bloom_proxy_median": entry["median"],
                        "bloom_proxy_mean": entry["mean"],
                        "bloom_proxy_std": entry["std"],
                        "count": entry["count"],
                        "period": period_label,
                    })

        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

    # Write combined CSV
    if csv_rows:
        fieldnames = [
            "control_name", "weir_name_kr", "year", "sensor",
            "bloom_proxy_median", "bloom_proxy_mean", "bloom_proxy_std",
            "count", "period",
        ]
        with open(COMBINED_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCombined CSV: {COMBINED_CSV}")

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(all_results)}/{len(controls)} control reaches.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
