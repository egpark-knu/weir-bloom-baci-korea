"""
Water-Masked Bloom Proxy Extraction (v2)
=========================================
Same pipeline as extract_all_weirs.py / extract_control_reaches.py
but with JRC Global Surface Water mask applied.

This ensures NDVI/NDCI is computed ONLY on water pixels,
excluding riparian vegetation that inflates values.

Water mask: JRC Global Surface Water (Pekel et al. 2016, Nature)
  - Collection: "JRC/GSW1_4/GlobalSurfaceWater"
  - Band: "occurrence" (0-100, % of time pixel was water)
  - Threshold: ≥ 50% occurrence = persistent water

Outputs:
  - output/water_masked/weirs/<weir_name>.json
  - output/water_masked/controls/<control_name>.json
  - output/water_masked/weirs_bloom_summary.csv
  - output/water_masked/controls_bloom_summary.csv

Usage:
    python extract_water_masked.py --target weirs      # weir sites only
    python extract_water_masked.py --target controls    # control reaches only
    python extract_water_masked.py --target all         # both (default)
    python extract_water_masked.py --test               # first 2 only
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

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
WEIR_FILE = BASE_DIR / "data" / "weir_inventory.json"
CONTROL_FILE = BASE_DIR / "data" / "control_reaches.json"
OUTPUT_BASE = BASE_DIR / "output" / "water_masked"
WEIR_OUT = OUTPUT_BASE / "weirs"
CTRL_OUT = OUTPUT_BASE / "controls"

# ── GEE Auth ──
SERVICE_ACCOUNT = "your-service-account@project.iam.gserviceaccount.com"
KEY_PATH = (
    "your-gee-service-key.json"  # Replace with your GEE service key path
    
)
GEE_PROJECT = "your-gee-project-id"

# ── Constants ──
BUFFER_M = 5000
BLOOM_MONTHS = (5, 10)
LANDSAT_PRE_RANGE = (2000, 2012)
LANDSAT_POST_RANGE = (2013, 2025)
SENTINEL2_RANGE = (2017, 2025)
SCALE_LANDSAT = 30
SCALE_S2 = 20
WATER_OCCURRENCE_THRESHOLD = 50  # ≥ 50% = persistent water


# ── GEE Init ──

def init_gee() -> None:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
    ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    print("[GEE] Initialized.")


# ── Water Mask ──

def get_water_mask(geom: ee.Geometry) -> ee.Image:
    """JRC Global Surface Water: persistent water pixels (occurrence ≥ threshold)."""
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occurrence = gsw.select("occurrence")
    water_mask = occurrence.gte(WATER_OCCURRENCE_THRESHOLD)
    return water_mask


# ── Cloud Masking ──

def mask_landsat(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)


def mask_s2(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask)


# ── Index + Water Mask ──

def add_bloom_proxy_landsat57(image: ee.Image) -> ee.Image:
    red = image.select("SR_B3").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndvi)


def add_bloom_proxy_landsat89(image: ee.Image) -> ee.Image:
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndvi)


def add_bloom_proxy_s2(image: ee.Image) -> ee.Image:
    red = image.select("B4").multiply(0.0001)
    re1 = image.select("B5").multiply(0.0001)
    ndci = re1.subtract(red).divide(re1.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndci)


# ── Collection Builders (with water mask) ──

def _bloom_filter(col: ee.ImageCollection, year: int) -> ee.ImageCollection:
    return (col
            .filterDate(f"{year}-05-01", f"{year}-11-01")
            .filter(ee.Filter.calendarRange(BLOOM_MONTHS[0], BLOOM_MONTHS[1], "month")))


def _apply_water_mask(col: ee.ImageCollection, water_mask: ee.Image) -> ee.ImageCollection:
    """Apply water mask to every image in the collection."""
    return col.map(lambda img: img.updateMask(water_mask))


def get_landsat57_col(geom: ee.Geometry, year: int, water_mask: ee.Image) -> ee.ImageCollection:
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat57))
    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat57))
    col = _bloom_filter(l5.merge(l7), year)
    return _apply_water_mask(col, water_mask)


def get_landsat89_col(geom: ee.Geometry, year: int, water_mask: ee.Image) -> ee.ImageCollection:
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat89))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_bloom_proxy_landsat89))
    col = _bloom_filter(l8.merge(l9), year)
    return _apply_water_mask(col, water_mask)


def get_s2_col(geom: ee.Geometry, year: int, water_mask: ee.Image) -> ee.ImageCollection:
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(geom)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .map(mask_s2)
           .map(add_bloom_proxy_s2))
    col = _bloom_filter(col, year)
    return _apply_water_mask(col, water_mask)


# ── Stats ──

def _combined_reducer() -> ee.Reducer:
    return (ee.Reducer.median()
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True))


def extract_stats(col: ee.ImageCollection, geom: ee.Geometry, scale: int) -> dict:
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


# ── Extraction ──

def extract_one_site(name: str, lat: float, lon: float, meta: dict) -> dict:
    """Extract water-masked bloom proxy for a single site."""
    point = ee.Geometry.Point(lon, lat)
    geom = point.buffer(BUFFER_M)
    water_mask = get_water_mask(geom)

    # Check water pixel count
    water_count = water_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=SCALE_LANDSAT,
        maxPixels=1e8,
    ).getInfo()
    n_water = int(water_count.get("occurrence", 0))

    total_count = water_mask.Not().add(water_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=SCALE_LANDSAT,
        maxPixels=1e8,
    ).getInfo()
    n_total = int(total_count.get("occurrence", 0))

    water_pct = n_water / n_total * 100 if n_total > 0 else 0
    print(f"  Water pixels: {n_water}/{n_total} ({water_pct:.1f}%)")

    result = {
        **meta,
        "name": name,
        "lat": lat,
        "lon": lon,
        "buffer_m": BUFFER_M,
        "water_mask_threshold": WATER_OCCURRENCE_THRESHOLD,
        "water_pixel_count": n_water,
        "water_pixel_pct": round(water_pct, 1),
        "landsat_pre_weir": [],
        "landsat_post_weir": [],
        "sentinel2": [],
    }

    # Landsat 5/7 pre-weir
    for year in range(LANDSAT_PRE_RANGE[0], LANDSAT_PRE_RANGE[1] + 1):
        try:
            col = get_landsat57_col(geom, year, water_mask)
            raw = extract_stats(col, geom, SCALE_LANDSAT)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Landsat 5/7"
            result["landsat_pre_weir"].append(entry)
        except Exception as exc:
            print(f"    WARN L57 {year}: {exc}")
            result["landsat_pre_weir"].append(
                {"year": year, "sensor": "Landsat 5/7",
                 "median": None, "mean": None, "std": None, "count": 0})

    # Landsat 8/9 post-weir
    for year in range(LANDSAT_POST_RANGE[0], LANDSAT_POST_RANGE[1] + 1):
        try:
            col = get_landsat89_col(geom, year, water_mask)
            raw = extract_stats(col, geom, SCALE_LANDSAT)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Landsat 8/9"
            result["landsat_post_weir"].append(entry)
        except Exception as exc:
            print(f"    WARN L89 {year}: {exc}")
            result["landsat_post_weir"].append(
                {"year": year, "sensor": "Landsat 8/9",
                 "median": None, "mean": None, "std": None, "count": 0})

    # Sentinel-2
    for year in range(SENTINEL2_RANGE[0], SENTINEL2_RANGE[1] + 1):
        try:
            col = get_s2_col(geom, year, water_mask)
            raw = extract_stats(col, geom, SCALE_S2)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Sentinel-2"
            result["sentinel2"].append(entry)
        except Exception as exc:
            print(f"    WARN S2 {year}: {exc}")
            result["sentinel2"].append(
                {"year": year, "sensor": "Sentinel-2",
                 "median": None, "mean": None, "std": None, "count": 0})

    return result


def extract_weirs(test_mode: bool = False):
    """Extract water-masked bloom proxy for all weir sites."""
    WEIR_OUT.mkdir(parents=True, exist_ok=True)

    with open(WEIR_FILE) as f:
        weirs = json.load(f)

    if test_mode:
        weirs = weirs[:2]
        print(f"[TEST] First {len(weirs)} weirs only.\n")

    all_results = []
    csv_rows = []

    for i, w in enumerate(weirs, 1):
        name = w.get("name_en", w.get("weir_name_en", f"weir_{i}"))
        name_kr = w.get("name_kr", w.get("weir_name_kr", ""))
        lat, lon = w["lat"], w["lon"]
        print(f"\n[Weir {i}/{len(weirs)}] {name} ({name_kr}) ({lat:.4f}, {lon:.4f})")

        t0 = time.time()
        try:
            meta = {"site_type": "weir", "weir_name_kr": name_kr, "weir_name_en": name}
            result = extract_one_site(name, lat, lon, meta)
            all_results.append(result)

            out_path = WEIR_OUT / f"{name}_Weir.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_path}")

            for period_key in ["landsat_pre_weir", "landsat_post_weir", "sentinel2"]:
                label = {"landsat_pre_weir": "pre", "landsat_post_weir": "post", "sentinel2": "s2"}[period_key]
                for entry in result[period_key]:
                    csv_rows.append({
                        "site": name, "site_type": "weir", "weir_name_kr": name_kr,
                        "year": entry["year"], "sensor": entry["sensor"],
                        "bloom_proxy_median": entry["median"],
                        "bloom_proxy_mean": entry["mean"],
                        "bloom_proxy_std": entry["std"],
                        "count": entry["count"], "period": label,
                        "water_pixel_pct": result["water_pixel_pct"],
                    })
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()

        print(f"  Time: {time.time() - t0:.1f}s")

    if csv_rows:
        csv_path = OUTPUT_BASE / "weirs_bloom_summary.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV: {csv_path}")

    return all_results


def extract_controls(test_mode: bool = False):
    """Extract water-masked bloom proxy for all upstream control reaches."""
    CTRL_OUT.mkdir(parents=True, exist_ok=True)

    with open(CONTROL_FILE) as f:
        controls = json.load(f)

    if test_mode:
        controls = controls[:2]
        print(f"[TEST] First {len(controls)} controls only.\n")

    all_results = []
    csv_rows = []

    for i, c in enumerate(controls, 1):
        name = c["control_name"]
        lat, lon = c["lat"], c["lon"]
        print(f"\n[Control {i}/{len(controls)}] {name} ({c['weir_name_kr']}) ({lat:.4f}, {lon:.4f})")

        t0 = time.time()
        try:
            meta = {
                "site_type": "control", "control_name": name,
                "weir_name_kr": c["weir_name_kr"], "weir_name_en": c["weir_name_en"],
                "distance_km": c["distance_km"],
            }
            result = extract_one_site(name, lat, lon, meta)
            all_results.append(result)

            out_path = CTRL_OUT / f"{name}.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_path}")

            for period_key in ["landsat_pre_weir", "landsat_post_weir", "sentinel2"]:
                label = {"landsat_pre_weir": "pre", "landsat_post_weir": "post", "sentinel2": "s2"}[period_key]
                for entry in result[period_key]:
                    csv_rows.append({
                        "site": name, "site_type": "control",
                        "weir_name_kr": c["weir_name_kr"],
                        "year": entry["year"], "sensor": entry["sensor"],
                        "bloom_proxy_median": entry["median"],
                        "bloom_proxy_mean": entry["mean"],
                        "bloom_proxy_std": entry["std"],
                        "count": entry["count"], "period": label,
                        "water_pixel_pct": result["water_pixel_pct"],
                    })
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()

        print(f"  Time: {time.time() - t0:.1f}s")

    if csv_rows:
        csv_path = OUTPUT_BASE / "controls_bloom_summary.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV: {csv_path}")

    return all_results


# ── Main ──

def main():
    init_gee()

    target = "all"
    test_mode = "--test" in sys.argv
    if "--target" in sys.argv:
        idx = sys.argv.index("--target")
        target = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "all"

    print(f"Target: {target}, Test: {test_mode}")
    print(f"Water mask threshold: {WATER_OCCURRENCE_THRESHOLD}% occurrence\n")

    if target in ("weirs", "all"):
        print("=" * 60)
        print("EXTRACTING WEIR SITES (water-masked)")
        print("=" * 60)
        extract_weirs(test_mode)

    if target in ("controls", "all"):
        print("\n" + "=" * 60)
        print("EXTRACTING CONTROL REACHES (water-masked)")
        print("=" * 60)
        extract_controls(test_mode)

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
