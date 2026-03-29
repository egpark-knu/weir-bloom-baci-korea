"""
GEE Bloom Index Extraction Pipeline
=====================================
HAB 논문: Landsat-Sentinel 연계 bloom proxy time series 추출

센서 3구간:
  - Period 1: Landsat-only baseline (2000-2012)
  - Period 2: Overlap/transition (2013-2017) — Landsat 8 + Sentinel-2A(2015.6~) + HLS
  - Period 3: Sentinel-2 enhanced (2018-2025)

Bloom Indices:
  - NDCI (Normalized Difference Chlorophyll Index): (B5-B4)/(B5+B4) [Sentinel-2]
  - CI_cyano (Cyanobacteria Index): B5 - (B4 + (B6-B4)*(705-665)/(740-665)) [Sentinel-2]
  - FLH (Fluorescence Line Height): B5 - B4 - (B6-B4)*(705-665)/(740-665) [Sentinel-2]

Landsat equivalents use appropriate band mapping.

Usage:
    # After ee.Authenticate() and ee.Initialize()
    python gee_bloom_pipeline.py
"""

import ee
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

GEE_PROJECT = "your-gee-project-id"

# Band mappings for bloom index calculation
BAND_MAP = {
    "landsat5": {
        "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4",
        "qa": "QA_PIXEL", "scale": 0.0000275, "offset": -0.2
    },
    "landsat7": {
        "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4",
        "qa": "QA_PIXEL", "scale": 0.0000275, "offset": -0.2
    },
    "landsat8": {
        "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5",
        "qa": "QA_PIXEL", "scale": 0.0000275, "offset": -0.2
    },
    "landsat9": {
        "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5",
        "qa": "QA_PIXEL", "scale": 0.0000275, "offset": -0.2
    },
    "sentinel2": {
        "green": "B3",      # 560 nm
        "red": "B4",        # 665 nm
        "rededge1": "B5",   # 705 nm
        "rededge2": "B6",   # 740 nm
        "nir": "B8",        # 842 nm
        "qa": "QA60", "scale": 0.0001, "offset": 0
    }
}

# Bloom season: May-October (주 분석 기간)
BLOOM_MONTHS = [5, 6, 7, 8, 9, 10]
# Negative control season: December-February
NEGATIVE_CONTROL_MONTHS = [12, 1, 2]


# ─────────────────────────────────────────────
# Cloud Masking
# ─────────────────────────────────────────────

def mask_landsat_sr(image):
    """Landsat Collection 2 SR cloud/shadow/water masking."""
    qa = image.select("QA_PIXEL")
    # Bit 3: cloud shadow, Bit 4: cloud
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    cloud = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud_shadow).updateMask(cloud)


def mask_sentinel2(image):
    """Sentinel-2 cloud masking using QA60."""
    qa = image.select("QA60")
    # Bit 10: opaque clouds, Bit 11: cirrus
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return image.updateMask(cloud).updateMask(cirrus)


# ─────────────────────────────────────────────
# Bloom Index Computation
# ─────────────────────────────────────────────

def compute_landsat_bloom_indices(image, sensor="landsat8"):
    """Compute bloom proxies from Landsat SR."""
    bm = BAND_MAP[sensor]

    # Apply scaling
    green = image.select(bm["green"]).multiply(bm["scale"]).add(bm["offset"])
    red = image.select(bm["red"]).multiply(bm["scale"]).add(bm["offset"])
    nir = image.select(bm["nir"]).multiply(bm["scale"]).add(bm["offset"])

    # NDCI proxy using NIR and Red (Landsat lacks red-edge)
    # For Landsat, use (NIR - Red) / (NIR + Red) as NDVI-like bloom proxy
    ndci_proxy = nir.subtract(red).divide(nir.add(red)).rename("NDCI_proxy")

    # Green-Red ratio (simple bloom indicator)
    gr_ratio = green.divide(red).rename("GR_ratio")

    return image.addBands([ndci_proxy, gr_ratio])


def compute_sentinel2_bloom_indices(image):
    """Compute bloom proxies from Sentinel-2 L2A."""
    bm = BAND_MAP["sentinel2"]

    red = image.select(bm["red"]).multiply(bm["scale"])       # B4, 665nm
    re1 = image.select(bm["rededge1"]).multiply(bm["scale"])  # B5, 705nm
    re2 = image.select(bm["rededge2"]).multiply(bm["scale"])  # B6, 740nm

    # NDCI: (B5 - B4) / (B5 + B4)
    ndci = re1.subtract(red).divide(re1.add(red)).rename("NDCI")

    # CI_cyano: B5 - [B4 + (B6 - B4) * (705 - 665) / (740 - 665)]
    # Baseline interpolation between B4 and B6 at 705nm wavelength
    baseline = red.add(re2.subtract(red).multiply((705 - 665) / (740 - 665)))
    ci_cyano = re1.subtract(baseline).rename("CI_cyano")

    # FLH (Fluorescence Line Height) — same formula structure
    flh = re1.subtract(baseline).rename("FLH")

    return image.addBands([ndci, ci_cyano, flh])


# ─────────────────────────────────────────────
# Reach Geometry Definition
# ─────────────────────────────────────────────

def create_reach_geometry(lat, lon, influence_km=5.0, width_m=500):
    """Create a buffered reach geometry around a weir location.

    Args:
        lat, lon: Weir center coordinates
        influence_km: Upstream influence distance (km)
        width_m: River corridor width buffer (m)

    Returns:
        ee.Geometry for the reach
    """
    center = ee.Geometry.Point([lon, lat])
    # Buffer by influence distance upstream + width
    reach = center.buffer(influence_km * 1000)
    return reach


def load_weir_inventory(json_path):
    """Load 16-weir inventory from JSON."""
    with open(json_path) as f:
        weirs = json.load(f)
    return weirs


# ─────────────────────────────────────────────
# Time Series Extraction
# ─────────────────────────────────────────────

def extract_landsat_timeseries(reach_geom, start_year, end_year, months):
    """Extract Landsat bloom proxy time series for a reach.

    Combines Landsat 5/7/8/9 Collection 2 SR.
    """
    collections = {
        "landsat5": ("LANDSAT/LT05/C02/T1_L2", 1984, 2012),
        "landsat7": ("LANDSAT/LE07/C02/T1_L2", 1999, 2022),
        "landsat8": ("LANDSAT/LC08/C02/T1_L2", 2013, 2025),
        "landsat9": ("LANDSAT/LC09/C02/T1_L2", 2021, 2025),
    }

    all_images = ee.ImageCollection([])

    for sensor, (collection_id, avail_start, avail_end) in collections.items():
        eff_start = max(start_year, avail_start)
        eff_end = min(end_year, avail_end)
        if eff_start > eff_end:
            continue

        col = (ee.ImageCollection(collection_id)
               .filterBounds(reach_geom)
               .filterDate(f"{eff_start}-01-01", f"{eff_end}-12-31")
               .filter(ee.Filter.calendarRange(months[0], months[-1], "month"))
               .map(mask_landsat_sr)
               .map(lambda img: compute_landsat_bloom_indices(img, sensor)))

        all_images = all_images.merge(col)

    return all_images


def extract_sentinel2_timeseries(reach_geom, start_year, end_year, months):
    """Extract Sentinel-2 bloom index time series for a reach."""
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(reach_geom)
           .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
           .filter(ee.Filter.calendarRange(months[0], months[-1], "month"))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .map(mask_sentinel2)
           .map(compute_sentinel2_bloom_indices))

    return col


def compute_seasonal_composite(collection, reach_geom, index_band, year, months):
    """Compute seasonal (bloom season) median composite for a reach-year.

    Returns dict with: year, median, p25, p75, count, mean
    """
    filtered = (collection
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .filter(ee.Filter.calendarRange(months[0], months[-1], "month"))
                .select(index_band))

    stats = filtered.reduce(
        ee.Reducer.median()
        .combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
        .combine(ee.Reducer.mean(), sharedInputs=True)
    ).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=reach_geom,
        scale=30,
        maxPixels=1e8
    )

    return stats


def extract_reach_annual_series(reach_geom, weir_name, completion_year,
                                 start_year=2000, end_year=2025):
    """Extract full annual bloom time series for one reach.

    Combines Landsat (all periods) + Sentinel-2 (2015+).
    Returns list of dicts: [{year, sensor, ndci_median, ndci_p25, ndci_p75, count}, ...]
    """
    results = []

    for year in range(start_year, end_year + 1):
        # Landsat (all years)
        landsat_col = extract_landsat_timeseries(
            reach_geom, year, year, BLOOM_MONTHS)

        ls_stats = compute_seasonal_composite(
            landsat_col, reach_geom, "NDCI_proxy", year, BLOOM_MONTHS)

        results.append({
            "weir": weir_name,
            "year": year,
            "sensor": "landsat",
            "period": "pre" if year < completion_year else "post",
            "stats": ls_stats.getInfo()
        })

        # Sentinel-2 (2015+)
        if year >= 2016:
            s2_col = extract_sentinel2_timeseries(
                reach_geom, year, year, BLOOM_MONTHS)

            s2_stats = compute_seasonal_composite(
                s2_col, reach_geom, "NDCI", year, BLOOM_MONTHS)

            results.append({
                "weir": weir_name,
                "year": year,
                "sensor": "sentinel2",
                "period": "pre" if year < completion_year else "post",
                "stats": s2_stats.getInfo()
            })

    return results


# ─────────────────────────────────────────────
# HLS (Harmonized Landsat Sentinel) — Overlap Period
# ─────────────────────────────────────────────

def extract_hls_timeseries(reach_geom, start_year=2013, end_year=2025, months=BLOOM_MONTHS):
    """Extract NASA HLS time series (cross-calibrated Landsat-Sentinel).

    HLS is available in GEE as NASA/HLS/HLSL30 (Landsat) and NASA/HLS/HLSS30 (Sentinel).
    """
    # HLS Landsat
    hls_l = (ee.ImageCollection("NASA/HLS/HLSL30/v002")
             .filterBounds(reach_geom)
             .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
             .filter(ee.Filter.calendarRange(months[0], months[-1], "month")))

    # HLS Sentinel
    hls_s = (ee.ImageCollection("NASA/HLS/HLSS30/v002")
             .filterBounds(reach_geom)
             .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
             .filter(ee.Filter.calendarRange(months[0], months[-1], "month")))

    return hls_l.merge(hls_s)


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_full_pipeline(weir_inventory_path, output_dir):
    """Run complete bloom extraction pipeline for all weirs.

    Steps:
    1. Load weir inventory
    2. For each weir: extract annual bloom time series (Landsat + S2)
    3. Save results as JSON per weir + combined CSV
    """
    # Initialize GEE
    ee.Initialize(project=GEE_PROJECT)

    weirs = load_weir_inventory(weir_inventory_path)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for weir in weirs:
        name = weir["name_kr"]
        lat, lon = weir["lat"], weir["lon"]
        comp_year = weir["completion_year"]
        influence = weir.get("influence_km", 5.0)

        print(f"\n--- Processing: {name} ({lat:.4f}, {lon:.4f}) ---")

        # Create reach geometry
        reach = create_reach_geometry(lat, lon, influence_km=influence)

        # Extract time series
        series = extract_reach_annual_series(
            reach, name, comp_year,
            start_year=2000, end_year=2025
        )

        all_results.extend(series)

        # Save per-weir
        out_path = os.path.join(output_dir, f"{weir['name_en'].replace(' ', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(series, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {out_path} ({len(series)} records)")

    # Save combined
    combined_path = os.path.join(output_dir, "all_weirs_bloom_series.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Pipeline complete: {len(all_results)} total records ===")
    print(f"Combined output: {combined_path}")

    return all_results


# ─────────────────────────────────────────────
# Negative Control Extraction
# ─────────────────────────────────────────────

def extract_negative_control_series(reach_geom, weir_name, completion_year,
                                     start_year=2000, end_year=2025):
    """Extract bloom indices during non-bloom season (Dec-Feb) as negative control.

    If weir effect is real, the Dec-Feb period should show NO treatment effect.
    """
    results = []

    for year in range(start_year, end_year + 1):
        landsat_col = extract_landsat_timeseries(
            reach_geom, year, year, NEGATIVE_CONTROL_MONTHS)

        stats = compute_seasonal_composite(
            landsat_col, reach_geom, "NDCI_proxy", year, NEGATIVE_CONTROL_MONTHS)

        results.append({
            "weir": weir_name,
            "year": year,
            "sensor": "landsat",
            "season": "winter_control",
            "period": "pre" if year < completion_year else "post",
            "stats": stats.getInfo()
        })

    return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEE Bloom Index Pipeline")
    parser.add_argument("--inventory", default="weir_inventory.json",
                        help="Path to weir inventory JSON")
    parser.add_argument("--output", default="bloom_data",
                        help="Output directory")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process first 2 weirs only")
    args = parser.parse_args()

    if args.test:
        # Quick test with subset
        ee.Initialize(project=GEE_PROJECT)
        weirs = load_weir_inventory(args.inventory)[:2]
        for w in weirs:
            reach = create_reach_geometry(w["lat"], w["lon"])
            print(f"Testing {w['name_kr']}...")
            # Just check image count
            col = extract_landsat_timeseries(reach, 2010, 2012, BLOOM_MONTHS)
            count = col.size().getInfo()
            print(f"  Landsat images (2010-2012): {count}")
    else:
        run_full_pipeline(args.inventory, args.output)
