"""
Susceptibility Index Construction
==================================
HAB 논문: Source Pressure Index (SPI) + Subsurface Susceptibility Index (SSI)

Inputs (from geodata):
  - geology.gpkg: 충적층/투수성 지질 분류
  - soil_drn.gpkg: 토양 배수등급 (투수성)
  - national_monitoring_wells.gpkg: 지하수 관측정 (수위)
  - dem_90.tif: 경사도
  - standard_watershed.gpkg: 유역 경계
  - EGIS 토지피복도: 축산시설, 농경지, 도시
  - population.gpkg: 인구 밀도

Weight schemes for sensitivity analysis:
  1. Equal weights
  2. Expert-based weights (from research plan v2)
  3. PCA-derived data-driven weights

Output:
  - Per-reach SPI and SSI values
  - 3 weight scheme variants for robustness
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# Weight Schemes (from research plan v2)
# ─────────────────────────────────────────────

WEIGHT_SCHEMES = {
    "expert": {
        "SPI": {
            "livestock_density": 0.40,
            "ag_land_fraction": 0.30,
            "population_density": 0.15,
            "fertilizer_proxy": 0.15,
        },
        "SSI": {
            "alluvial_geology": 0.25,
            "permeable_soil": 0.25,
            "shallow_gw": 0.20,
            "low_slope": 0.15,
            "floodplain_proximity": 0.15,
        }
    },
    "equal": {
        "SPI": {
            "livestock_density": 0.25,
            "ag_land_fraction": 0.25,
            "population_density": 0.25,
            "fertilizer_proxy": 0.25,
        },
        "SSI": {
            "alluvial_geology": 0.20,
            "permeable_soil": 0.20,
            "shallow_gw": 0.20,
            "low_slope": 0.20,
            "floodplain_proximity": 0.20,
        }
    },
    # PCA weights are computed from data
}


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_geodata(geodata_dir):
    """Load all required GIS layers from geodata directory.

    Args:
        geodata_dir: Path to 07_geodata/ directory

    Returns:
        dict of GeoDataFrames
    """
    gdir = Path(geodata_dir)
    data = {}

    layers = {
        "geology": "geology.gpkg",
        "soil": "soil_drn.gpkg",
        "gw_wells": "national_monitoring_wells.gpkg",
        "river": "river.gpkg",
        "watershed": "standard_watershed.gpkg",
        "population": "population.gpkg",
    }

    for name, filename in layers.items():
        path = gdir / filename
        if path.exists():
            data[name] = gpd.read_file(path)
            print(f"  Loaded {name}: {len(data[name])} features")
        else:
            print(f"  ⚠️ Missing: {path}")
            data[name] = None

    # DEM (raster)
    dem_path = gdir / "dem_90.tif"
    if dem_path.exists():
        data["dem_path"] = str(dem_path)
        print(f"  Found DEM: {dem_path}")
    else:
        print(f"  ⚠️ Missing DEM: {dem_path}")
        data["dem_path"] = None

    return data


def load_landuse(egis_path):
    """Load 환경부 EGIS 토지피복도.

    Extract:
    - Livestock facility density
    - Agricultural land fraction
    - Urban/population proxy
    """
    if not Path(egis_path).exists():
        print(f"  ⚠️ EGIS 토지피복도 not found: {egis_path}")
        return None

    landuse = gpd.read_file(egis_path)
    print(f"  Loaded EGIS landuse: {len(landuse)} features")
    return landuse


# ─────────────────────────────────────────────
# SSI Component Computation
# ─────────────────────────────────────────────

def compute_alluvial_fraction(geology_gdf, reach_geom):
    """Fraction of alluvial/quaternary deposits within reach buffer.

    충적층(Qa), 제4기 퇴적층 비율 → 높을수록 투수성 높음
    """
    if geology_gdf is None:
        return np.nan

    clipped = gpd.clip(geology_gdf, reach_geom)
    if clipped.empty:
        return 0.0

    total_area = clipped.geometry.area.sum()
    # Filter for alluvial/quaternary units
    # Column names may vary — adapt to actual schema
    alluvial_keywords = ["충적", "Qa", "alluvium", "제4기", "quaternary"]
    alluvial = clipped[clipped.apply(
        lambda row: any(kw in str(row.values) for kw in alluvial_keywords), axis=1
    )]
    alluvial_area = alluvial.geometry.area.sum() if not alluvial.empty else 0.0

    return alluvial_area / total_area if total_area > 0 else 0.0


def compute_permeable_soil_fraction(soil_gdf, reach_geom):
    """Fraction of high-permeability soil (양호/매우양호 배수등급) within reach."""
    if soil_gdf is None:
        return np.nan

    clipped = gpd.clip(soil_gdf, reach_geom)
    if clipped.empty:
        return 0.0

    total_area = clipped.geometry.area.sum()
    # High permeability keywords — adapt to actual column names
    perm_keywords = ["양호", "매우양호", "good", "excessive", "well"]
    permeable = clipped[clipped.apply(
        lambda row: any(kw in str(row.values) for kw in perm_keywords), axis=1
    )]
    perm_area = permeable.geometry.area.sum() if not permeable.empty else 0.0

    return perm_area / total_area if total_area > 0 else 0.0


def compute_shallow_gw(gw_wells_gdf, reach_geom, threshold_m=10.0):
    """Average groundwater depth within reach. Shallow = more connected.

    Returns: fraction of wells with depth < threshold_m
    """
    if gw_wells_gdf is None:
        return np.nan

    # Find wells within reach
    wells_in = gpd.sjoin(gw_wells_gdf, gpd.GeoDataFrame(
        geometry=[reach_geom], crs=gw_wells_gdf.crs), how="inner")

    if wells_in.empty:
        return np.nan

    # Look for depth column — adapt to actual schema
    depth_cols = [c for c in wells_in.columns if "depth" in c.lower() or "수위" in c]
    if not depth_cols:
        return np.nan

    depths = wells_in[depth_cols[0]].dropna()
    if depths.empty:
        return np.nan

    return (depths < threshold_m).mean()


def compute_slope(dem_path, reach_geom):
    """Mean slope within reach from DEM.

    Low slope → more susceptible to groundwater-surface water exchange
    Returns: normalized inverse slope (higher = lower slope = more susceptible)
    """
    if dem_path is None:
        return np.nan

    try:
        import rasterio
        from rasterio.mask import mask as rio_mask

        with rasterio.open(dem_path) as src:
            # This is simplified — proper slope requires gradient computation
            out_image, _ = rio_mask(src, [reach_geom.__geo_interface__], crop=True)
            elevation = out_image[0]
            elevation = elevation[elevation != src.nodata]

            if len(elevation) < 2:
                return np.nan

            # Compute slope proxy (range of elevation / extent)
            slope_proxy = np.std(elevation) / (np.mean(elevation) + 1e-6)
            # Inverse: low slope = high susceptibility
            return 1.0 / (1.0 + slope_proxy)
    except ImportError:
        print("  ⚠️ rasterio not available for slope computation")
        return np.nan


def compute_floodplain_proximity(river_gdf, reach_geom):
    """Fraction of reach area within floodplain distance of river network.

    Closer to river → more likely GW-SW interaction
    """
    if river_gdf is None:
        return np.nan

    # Simple proxy: density of river network within reach
    clipped = gpd.clip(river_gdf, reach_geom)
    if clipped.empty:
        return 0.0

    river_length = clipped.geometry.length.sum()
    reach_area = reach_geom.area if hasattr(reach_geom, 'area') else 1.0

    # Normalize: river density (m/m²)
    return river_length / reach_area if reach_area > 0 else 0.0


# ─────────────────────────────────────────────
# SPI Component Computation
# ─────────────────────────────────────────────

def compute_livestock_density(landuse_gdf, reach_geom):
    """Livestock facility density within reach watershed."""
    if landuse_gdf is None:
        return np.nan

    clipped = gpd.clip(landuse_gdf, reach_geom)
    if clipped.empty:
        return 0.0

    total_area = clipped.geometry.area.sum()
    livestock_keywords = ["축산", "livestock", "목장", "사육"]
    livestock = clipped[clipped.apply(
        lambda row: any(kw in str(row.values) for kw in livestock_keywords), axis=1
    )]
    livestock_area = livestock.geometry.area.sum() if not livestock.empty else 0.0

    return livestock_area / total_area if total_area > 0 else 0.0


def compute_ag_fraction(landuse_gdf, reach_geom):
    """Agricultural land fraction within reach watershed."""
    if landuse_gdf is None:
        return np.nan

    clipped = gpd.clip(landuse_gdf, reach_geom)
    if clipped.empty:
        return 0.0

    total_area = clipped.geometry.area.sum()
    ag_keywords = ["농경", "논", "밭", "과수", "agricultural", "paddy", "crop"]
    ag = clipped[clipped.apply(
        lambda row: any(kw in str(row.values) for kw in ag_keywords), axis=1
    )]
    ag_area = ag.geometry.area.sum() if not ag.empty else 0.0

    return ag_area / total_area if total_area > 0 else 0.0


def compute_population_density(pop_gdf, reach_geom):
    """Population density within reach."""
    if pop_gdf is None:
        return np.nan

    clipped = gpd.clip(pop_gdf, reach_geom)
    if clipped.empty:
        return 0.0

    pop_cols = [c for c in clipped.columns if "pop" in c.lower() or "인구" in c]
    if not pop_cols:
        return np.nan

    total_pop = clipped[pop_cols[0]].sum()
    area = clipped.geometry.area.sum()
    return total_pop / area if area > 0 else 0.0


# ─────────────────────────────────────────────
# Index Aggregation
# ─────────────────────────────────────────────

def compute_index(components, weights):
    """Compute weighted index from normalized components.

    Args:
        components: dict of {name: raw_value}
        weights: dict of {name: weight}

    Returns:
        Weighted index value (0-1 range after MinMax scaling)
    """
    values = []
    w = []
    for name, weight in weights.items():
        val = components.get(name, np.nan)
        if not np.isnan(val):
            values.append(val)
            w.append(weight)

    if not values:
        return np.nan

    values = np.array(values)
    w = np.array(w)
    w = w / w.sum()  # Re-normalize weights for missing components

    return np.dot(values, w)


def compute_pca_weights(all_components_df, index_type="SSI"):
    """Derive data-driven weights from PCA on all reach components.

    Uses first principal component loadings as weights.
    """
    cols = list(WEIGHT_SCHEMES["expert"][index_type].keys())
    available_cols = [c for c in cols if c in all_components_df.columns]

    if len(available_cols) < 2:
        return {c: 1.0 / len(cols) for c in cols}  # Fall back to equal

    data = all_components_df[available_cols].dropna()
    if len(data) < 3:
        return {c: 1.0 / len(cols) for c in cols}

    # Standardize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=1)
    pca.fit(scaled)

    loadings = np.abs(pca.components_[0])
    weights = loadings / loadings.sum()

    return dict(zip(available_cols, weights))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def compute_indices_for_all_reaches(reaches, geodata_dir, egis_path=None):
    """Compute SPI and SSI for all reaches with 3 weight schemes.

    Args:
        reaches: list of dicts with {name, geometry (shapely), ...}
        geodata_dir: path to geodata directory
        egis_path: path to EGIS land use data

    Returns:
        DataFrame with columns: reach_name, SPI_expert, SSI_expert,
                                 SPI_equal, SSI_equal, SPI_pca, SSI_pca,
                                 + all raw components
    """
    # Load data
    print("Loading geodata...")
    geodata = load_geodata(geodata_dir)
    landuse = load_landuse(egis_path) if egis_path else None

    results = []

    for reach in reaches:
        name = reach["name"]
        geom = reach["geometry"]

        print(f"\nComputing indices for: {name}")

        # SSI components
        ssi_components = {
            "alluvial_geology": compute_alluvial_fraction(geodata["geology"], geom),
            "permeable_soil": compute_permeable_soil_fraction(geodata["soil"], geom),
            "shallow_gw": compute_shallow_gw(geodata["gw_wells"], geom),
            "low_slope": compute_slope(geodata["dem_path"], geom),
            "floodplain_proximity": compute_floodplain_proximity(geodata["river"], geom),
        }

        # SPI components
        spi_components = {
            "livestock_density": compute_livestock_density(landuse, geom),
            "ag_land_fraction": compute_ag_fraction(landuse, geom),
            "population_density": compute_population_density(geodata["population"], geom),
            "fertilizer_proxy": compute_ag_fraction(landuse, geom) * 0.8,  # Proxy
        }

        row = {"reach_name": name}
        row.update({f"raw_{k}": v for k, v in ssi_components.items()})
        row.update({f"raw_{k}": v for k, v in spi_components.items()})

        # Compute with expert and equal weights
        for scheme in ["expert", "equal"]:
            row[f"SSI_{scheme}"] = compute_index(
                ssi_components, WEIGHT_SCHEMES[scheme]["SSI"])
            row[f"SPI_{scheme}"] = compute_index(
                spi_components, WEIGHT_SCHEMES[scheme]["SPI"])

        results.append(row)

    df = pd.DataFrame(results)

    # PCA weights (computed from all reaches)
    pca_ssi_weights = compute_pca_weights(
        df.rename(columns={f"raw_{k}": k for k in WEIGHT_SCHEMES["expert"]["SSI"]}),
        "SSI"
    )
    pca_spi_weights = compute_pca_weights(
        df.rename(columns={f"raw_{k}": k for k in WEIGHT_SCHEMES["expert"]["SPI"]}),
        "SPI"
    )

    print(f"\nPCA-derived SSI weights: {pca_ssi_weights}")
    print(f"PCA-derived SPI weights: {pca_spi_weights}")

    # Recompute with PCA weights
    for i, reach in enumerate(reaches):
        ssi_comps = {k: df.iloc[i][f"raw_{k}"] for k in WEIGHT_SCHEMES["expert"]["SSI"]}
        spi_comps = {k: df.iloc[i][f"raw_{k}"] for k in WEIGHT_SCHEMES["expert"]["SPI"]}
        df.loc[i, "SSI_pca"] = compute_index(ssi_comps, pca_ssi_weights)
        df.loc[i, "SPI_pca"] = compute_index(spi_comps, pca_spi_weights)

    return df


if __name__ == "__main__":
    print("Susceptibility Index module loaded.")
    print("Use compute_indices_for_all_reaches() with geodata directory.")
    print(f"Weight schemes: {list(WEIGHT_SCHEMES.keys())} + PCA")
