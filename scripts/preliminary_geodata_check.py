"""
Preliminary Geodata Check
==========================
GEE 인증 없이 로컬 geodata로 16보 주변 특성을 미리 확인한다.

1. 각 보 좌표를 중심으로 5km 버퍼 생성
2. 버퍼 내 지질, 토양, GW well, 하천 특성 추출
3. Focal weir 선정을 위한 기초 데이터 정리
"""

import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path

# ─── Config ───
GEODATA_DIR = Path("geodata")  # Local geodata directory
INVENTORY_PATH = Path(__file__).parent.parent / "data/weir_inventory.json"
OUTPUT_PATH = Path(__file__).parent.parent / "output"
BUFFER_KM = 5.0


def load_weirs():
    """Load weir inventory and create GeoDataFrame."""
    with open(INVENTORY_PATH) as f:
        data = json.load(f)

    weirs = data["weirs"]
    gdf = gpd.GeoDataFrame(
        weirs,
        geometry=[Point(w["coordinates"]["longitude"], w["coordinates"]["latitude"])
                  for w in weirs],
        crs="EPSG:4326"
    )
    # Project to UTM 52N (Korea)
    gdf = gdf.to_crs("EPSG:5186")  # Korea Central Belt
    return gdf


def create_buffers(gdf, buffer_m=5000):
    """Create buffer zones around each weir."""
    gdf_buf = gdf.copy()
    gdf_buf["geometry"] = gdf_buf.geometry.buffer(buffer_m)
    return gdf_buf


def check_geology(buf_gdf):
    """Check geology within each weir buffer."""
    geo_path = GEODATA_DIR / "geology.gpkg"
    if not geo_path.exists():
        print("⚠️ geology.gpkg not found")
        return None

    print("Loading geology... (391MB, may take a moment)")
    geology = gpd.read_file(geo_path)
    geology = geology.to_crs(buf_gdf.crs)

    results = []
    for idx, weir in buf_gdf.iterrows():
        name = weir["name_kr"]
        buf = weir.geometry

        clipped = gpd.clip(geology, buf)
        if clipped.empty:
            results.append({"weir": name, "geology_features": 0, "alluvial_pct": 0})
            continue

        total_area = clipped.geometry.area.sum()

        # Check for alluvial/quaternary
        alluvial_kw = ["충적", "Qa", "제4기", "quaternary", "alluvium", "미고결"]
        alluvial = clipped[clipped.apply(
            lambda row: any(kw in str(row.values).lower() for kw in alluvial_kw), axis=1
        )]
        alluvial_area = alluvial.geometry.area.sum() if not alluvial.empty else 0

        results.append({
            "weir": name,
            "geology_features": len(clipped),
            "alluvial_pct": round(alluvial_area / total_area * 100, 1) if total_area > 0 else 0,
            "total_area_km2": round(total_area / 1e6, 2),
        })
        print(f"  {name}: {len(clipped)} features, alluvial {results[-1]['alluvial_pct']}%")

    return pd.DataFrame(results)


def check_soil(buf_gdf):
    """Check soil drainage within each weir buffer."""
    soil_path = GEODATA_DIR / "soil_drn.gpkg"
    if not soil_path.exists():
        print("⚠️ soil_drn.gpkg not found")
        return None

    print("Loading soil drainage... (784MB, may take a while)")
    soil = gpd.read_file(soil_path)
    soil = soil.to_crs(buf_gdf.crs)

    results = []
    for idx, weir in buf_gdf.iterrows():
        name = weir["name_kr"]
        buf = weir.geometry

        clipped = gpd.clip(soil, buf)
        if clipped.empty:
            results.append({"weir": name, "soil_features": 0, "permeable_pct": 0})
            continue

        total_area = clipped.geometry.area.sum()
        perm_kw = ["양호", "매우양호", "good", "well", "excessive"]
        permeable = clipped[clipped.apply(
            lambda row: any(kw in str(row.values) for kw in perm_kw), axis=1
        )]
        perm_area = permeable.geometry.area.sum() if not permeable.empty else 0

        results.append({
            "weir": name,
            "soil_features": len(clipped),
            "permeable_pct": round(perm_area / total_area * 100, 1) if total_area > 0 else 0,
        })
        print(f"  {name}: {len(clipped)} features, permeable {results[-1]['permeable_pct']}%")

    return pd.DataFrame(results)


def check_gw_wells(buf_gdf):
    """Check groundwater monitoring wells within each weir buffer."""
    gw_path = GEODATA_DIR / "national_monitoring_wells.gpkg"
    if not gw_path.exists():
        print("⚠️ national_monitoring_wells.gpkg not found")
        return None

    print("Loading GW wells...")
    wells = gpd.read_file(gw_path)
    wells = wells.to_crs(buf_gdf.crs)

    results = []
    for idx, weir in buf_gdf.iterrows():
        name = weir["name_kr"]
        buf = weir.geometry

        wells_in = wells[wells.geometry.within(buf)]

        results.append({
            "weir": name,
            "n_wells": len(wells_in),
            "well_ids": list(wells_in.iloc[:, 0].values[:5]) if not wells_in.empty else [],
        })
        print(f"  {name}: {len(wells_in)} wells nearby")

    return pd.DataFrame(results)


def check_rivers(buf_gdf):
    """Check river network density within each weir buffer."""
    river_path = GEODATA_DIR / "river.gpkg"
    if not river_path.exists():
        print("⚠️ river.gpkg not found")
        return None

    print("Loading river network...")
    rivers = gpd.read_file(river_path)
    rivers = rivers.to_crs(buf_gdf.crs)

    results = []
    for idx, weir in buf_gdf.iterrows():
        name = weir["name_kr"]
        buf = weir.geometry

        clipped = gpd.clip(rivers, buf)
        river_length_km = clipped.geometry.length.sum() / 1000 if not clipped.empty else 0

        results.append({
            "weir": name,
            "river_segments": len(clipped),
            "river_length_km": round(river_length_km, 2),
        })
        print(f"  {name}: {len(clipped)} segments, {river_length_km:.1f} km")

    return pd.DataFrame(results)


def main():
    """Run all geodata checks and save summary."""
    print("=" * 60)
    print("Preliminary Geodata Check for 16 Weirs")
    print("=" * 60)

    # Load weirs
    print("\n1. Loading weir inventory...")
    weirs = load_weirs()
    print(f"   Loaded {len(weirs)} weirs")

    # Create buffers
    print(f"\n2. Creating {BUFFER_KM}km buffers...")
    buffers = create_buffers(weirs, buffer_m=BUFFER_KM * 1000)

    # Check each layer
    print("\n3. Checking GW wells (smallest file first)...")
    gw_df = check_gw_wells(buffers)

    print("\n4. Checking river network...")
    river_df = check_rivers(buffers)

    print("\n5. Checking geology...")
    geo_df = check_geology(buffers)

    print("\n6. Checking soil drainage...")
    soil_df = check_soil(buffers)

    # Merge results
    print("\n7. Merging results...")
    weir_names = [w["name_kr"] for w in json.load(open(INVENTORY_PATH))["weirs"]]
    summary = pd.DataFrame({"weir": weir_names})

    for df in [gw_df, river_df, geo_df, soil_df]:
        if df is not None:
            summary = summary.merge(df, on="weir", how="left")

    # Save
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_PATH / "geodata_check_summary.csv"
    summary.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ Summary saved to: {out_file}")

    # Print table
    print("\n" + "=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)

    return summary


if __name__ == "__main__":
    main()
