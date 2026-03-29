"""
Winter Negative Control Extraction (water-masked)
==================================================
Same pipeline as extract_water_masked.py but for Dec-Feb (DJF).
Used as a falsification test: no HAB effect expected in winter.

For "winter year Y", we use Dec(Y-1) through Feb(Y).
This ensures the same year labels as the bloom-season analysis.

Outputs:
  - output/water_masked/winter/weirs/<weir_name>.json
  - output/water_masked/winter/controls/<control_name>.json
  - output/water_masked/winter_summary.csv
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
OUTPUT_BASE = BASE_DIR / "output" / "water_masked" / "winter"
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
SCALE_LANDSAT = 30
WATER_OCCURRENCE_THRESHOLD = 50


def init_gee() -> None:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
    ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    print("[GEE] Initialized.")


def get_water_mask(geom: ee.Geometry) -> ee.Image:
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    return gsw.select("occurrence").gte(WATER_OCCURRENCE_THRESHOLD)


def mask_landsat(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)


def add_ndvi_landsat57(image: ee.Image) -> ee.Image:
    red = image.select("SR_B3").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndvi)


def add_ndvi_landsat89(image: ee.Image) -> ee.Image:
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("BLOOM_PROXY")
    return image.addBands(ndvi)


def _winter_filter(col: ee.ImageCollection, year: int) -> ee.ImageCollection:
    """Filter for DJF: Dec(year-1) through Feb(year)."""
    return col.filterDate(f"{year - 1}-12-01", f"{year}-03-01")


def _apply_water_mask(col: ee.ImageCollection, water_mask: ee.Image) -> ee.ImageCollection:
    return col.map(lambda img: img.updateMask(water_mask))


def get_landsat57_winter(geom: ee.Geometry, year: int, water_mask: ee.Image) -> ee.ImageCollection:
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndvi_landsat57))
    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndvi_landsat57))
    col = _winter_filter(l5.merge(l7), year)
    return _apply_water_mask(col, water_mask)


def get_landsat89_winter(geom: ee.Geometry, year: int, water_mask: ee.Image) -> ee.ImageCollection:
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndvi_landsat89))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndvi_landsat89))
    col = _winter_filter(l8.merge(l9), year)
    return _apply_water_mask(col, water_mask)


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


def extract_one_site_winter(name: str, lat: float, lon: float, meta: dict) -> dict:
    """Extract water-masked winter NDVI for a single site."""
    point = ee.Geometry.Point(lon, lat)
    geom = point.buffer(BUFFER_M)
    water_mask = get_water_mask(geom)

    result = {
        **meta,
        "name": name,
        "lat": lat,
        "lon": lon,
        "season": "winter_DJF",
        "landsat_pre_weir": [],
        "landsat_post_weir": [],
    }

    # Pre-weir: winter 2001-2012 (DJF: Dec(Y-1) to Feb(Y))
    # Start at 2001 because winter 2000 would need Dec 1999
    for year in range(2001, 2013):
        try:
            col = get_landsat57_winter(geom, year, water_mask)
            raw = extract_stats(col, geom, SCALE_LANDSAT)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Landsat 5/7"
            result["landsat_pre_weir"].append(entry)
        except Exception as exc:
            print(f"    WARN L57 winter {year}: {exc}")
            result["landsat_pre_weir"].append(
                {"year": year, "sensor": "Landsat 5/7",
                 "median": None, "mean": None, "std": None, "count": 0})

    # Post-weir: winter 2014-2025
    # Skip 2013 (transition: Dec 2012 is construction year)
    for year in range(2014, 2026):
        try:
            col = get_landsat89_winter(geom, year, water_mask)
            raw = extract_stats(col, geom, SCALE_LANDSAT)
            entry = parse_stats(raw)
            entry["year"] = year
            entry["sensor"] = "Landsat 8/9"
            result["landsat_post_weir"].append(entry)
        except Exception as exc:
            print(f"    WARN L89 winter {year}: {exc}")
            result["landsat_post_weir"].append(
                {"year": year, "sensor": "Landsat 8/9",
                 "median": None, "mean": None, "std": None, "count": 0})

    return result


def extract_all_winter(test_mode: bool = False):
    """Extract winter NDVI for all weirs and controls."""
    WEIR_OUT.mkdir(parents=True, exist_ok=True)
    CTRL_OUT.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    # ── Weirs ──
    with open(WEIR_FILE) as f:
        weirs = json.load(f)
    if test_mode:
        weirs = weirs[:2]

    print(f"\n{'=' * 60}")
    print(f"EXTRACTING WINTER NDVI — {len(weirs)} WEIRS")
    print(f"{'=' * 60}")

    for i, w in enumerate(weirs, 1):
        name = w.get("name_en", w.get("weir_name_en", f"weir_{i}"))
        name_kr = w.get("name_kr", w.get("weir_name_kr", ""))
        lat, lon = w["lat"], w["lon"]
        print(f"\n[Weir {i}/{len(weirs)}] {name} ({name_kr})")

        # Skip already-extracted sites
        out_path = WEIR_OUT / f"{name}_winter.json"
        if out_path.exists() and "--force" not in sys.argv:
            print(f"  SKIP (already exists)")
            with open(out_path) as fh:
                result = json.load(fh)
            for pk in ["landsat_pre_weir", "landsat_post_weir"]:
                period = "pre" if "pre" in pk else "post"
                for entry in result[pk]:
                    csv_rows.append({
                        "site": name, "site_type": "weir",
                        "year": entry["year"], "median": entry["median"],
                        "mean": entry["mean"], "count": entry["count"],
                        "period": period,
                    })
            continue

        t0 = time.time()
        try:
            meta = {"site_type": "weir", "weir_name_kr": name_kr}
            result = extract_one_site_winter(name, lat, lon, meta)
            out_path = WEIR_OUT / f"{name}_winter.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)

            for pk in ["landsat_pre_weir", "landsat_post_weir"]:
                period = "pre" if "pre" in pk else "post"
                for entry in result[pk]:
                    csv_rows.append({
                        "site": name, "site_type": "weir",
                        "year": entry["year"], "median": entry["median"],
                        "mean": entry["mean"], "count": entry["count"],
                        "period": period,
                    })
            print(f"  OK ({time.time() - t0:.1f}s)")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    # ── Controls ──
    with open(CONTROL_FILE) as f:
        controls = json.load(f)
    if test_mode:
        controls = controls[:2]

    print(f"\n{'=' * 60}")
    print(f"EXTRACTING WINTER NDVI — {len(controls)} CONTROLS")
    print(f"{'=' * 60}")

    for i, c in enumerate(controls, 1):
        name = c["control_name"]
        lat, lon = c["lat"], c["lon"]
        print(f"\n[Control {i}/{len(controls)}] {name}")

        # Skip already-extracted sites
        out_path = CTRL_OUT / f"{name}_winter.json"
        if out_path.exists() and "--force" not in sys.argv:
            print(f"  SKIP (already exists)")
            with open(out_path) as fh:
                result = json.load(fh)
            for pk in ["landsat_pre_weir", "landsat_post_weir"]:
                period = "pre" if "pre" in pk else "post"
                for entry in result[pk]:
                    csv_rows.append({
                        "site": name, "site_type": "control",
                        "year": entry["year"], "median": entry["median"],
                        "mean": entry["mean"], "count": entry["count"],
                        "period": period,
                    })
            continue

        t0 = time.time()
        try:
            meta = {"site_type": "control", "weir_name_en": c["weir_name_en"]}
            result = extract_one_site_winter(name, lat, lon, meta)
            out_path = CTRL_OUT / f"{name}_winter.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)

            for pk in ["landsat_pre_weir", "landsat_post_weir"]:
                period = "pre" if "pre" in pk else "post"
                for entry in result[pk]:
                    csv_rows.append({
                        "site": name, "site_type": "control",
                        "year": entry["year"], "median": entry["median"],
                        "mean": entry["mean"], "count": entry["count"],
                        "period": period,
                    })
            print(f"  OK ({time.time() - t0:.1f}s)")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    # ── Summary CSV ──
    if csv_rows:
        csv_path = OUTPUT_BASE / "winter_summary.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV: {csv_path}")

    print(f"\n{'=' * 60}")
    print("Winter extraction done.")
    return csv_rows


if __name__ == "__main__":
    init_gee()
    test_mode = "--test" in sys.argv
    extract_all_winter(test_mode)
