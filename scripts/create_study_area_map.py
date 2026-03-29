"""
Study Area Map for HAB Paper (Fig. 1)
======================================
16 weirs across 4 major rivers of South Korea.
Basemap via contextily (OpenStreetMap tiles) showing rivers and terrain.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import contextily as cx

OUTPUT_DIR = Path("Path(__file__).parent.parent / "output"")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Weir inventory
WEIR_FILE = Path(str(Path(__file__).parent.parent / "data/weir_inventory.json"))
CONTROL_FILE = Path(str(Path(__file__).parent.parent / "data/control_reaches.json"))

RIVER_GROUPS = {
    "Han R.": ["이포보", "여주보", "강천보"],
    "Nakdong R.": ["상주보", "낙단보", "구미보", "칠곡보", "강정고령보", "달성보", "합천창녕보", "창녕함안보"],
    "Geum R.": ["세종보", "공주보", "백제보"],
    "Yeongsan R.": ["승촌보", "죽산보"],
}

RIVER_COLORS = {
    "Han R.": "#2196F3",
    "Nakdong R.": "#F44336",
    "Geum R.": "#4CAF50",
    "Yeongsan R.": "#FF9800",
}

WEIR_EN = {
    "이포보": "Ipo", "여주보": "Yeoju", "강천보": "Gangcheon",
    "상주보": "Sangju", "낙단보": "Nakdan", "구미보": "Gumi",
    "칠곡보": "Chilgok", "강정고령보": "Gangjeong", "달성보": "Dalseong",
    "합천창녕보": "Hapcheon", "창녕함안보": "Changnyeong",
    "세종보": "Sejong", "공주보": "Gongju", "백제보": "Baekje",
    "승촌보": "Seungchon", "죽산보": "Juksan",
}


def load_weir_inventory():
    with open(WEIR_FILE) as f:
        return json.load(f)


def load_control_reaches():
    with open(CONTROL_FILE) as f:
        return json.load(f)


def create_study_area_map():
    weirs = load_weir_inventory()
    controls = load_control_reaches()

    fig, ax = plt.subplots(figsize=(10, 12))

    # Use weir + control locations to set bounds
    lons = [w["lon"] for w in weirs]
    lats = [w["lat"] for w in weirs]
    ctrl_lons_all = [c["lon"] for c in controls]
    ctrl_lats_all = [c["lat"] for c in controls]
    all_lons = lons + ctrl_lons_all
    all_lats = lats + ctrl_lats_all

    lon_min, lon_max = min(all_lons) - 0.4, max(all_lons) + 0.4
    lat_min, lat_max = min(all_lats) - 0.4, max(all_lats) + 0.4

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Add basemap (OpenStreetMap tiles showing rivers/terrain)
    try:
        cx.add_basemap(ax, crs="EPSG:4326",
                       source=cx.providers.CartoDB.Voyager,
                       zoom=9, alpha=0.6)
    except Exception as e:
        print(f"[WARN] Basemap failed: {e}. Using plain background.")
        ax.set_facecolor("#f0f4f8")

    # Plot weirs by river group
    for river, weir_names in RIVER_GROUPS.items():
        color = RIVER_COLORS[river]
        river_weirs = [w for w in weirs if w["name_kr"] in weir_names]

        river_lons = [w["lon"] for w in river_weirs]
        river_lats = [w["lat"] for w in river_weirs]

        # Draw river line (connect weirs in order)
        if len(river_weirs) > 1:
            ax.plot(river_lons, river_lats, '-', color=color, linewidth=2, alpha=0.4, zorder=2)

        # Plot weirs
        ax.scatter(river_lons, river_lats, c=color, s=120, zorder=5,
                  edgecolors="white", linewidths=1.5, label=f"{river} (n={len(river_weirs)})")

        # Label weirs
        for w in river_weirs:
            en_name = WEIR_EN.get(w["name_kr"], w["name_kr"])
            # Offset to avoid overlap
            offset_x = 0.03
            offset_y = 0.02
            ax.annotate(en_name,
                       (w["lon"], w["lat"]),
                       xytext=(offset_x, offset_y),
                       textcoords="offset fontsize",
                       fontsize=8, fontweight="bold",
                       color=color,
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                edgecolor=color, alpha=0.8),
                       zorder=6)

    # Plot control reaches (upstream reference points)
    ctrl_lons = ctrl_lons_all
    ctrl_lats = ctrl_lats_all
    ax.scatter(ctrl_lons, ctrl_lats, c="gray", s=60, zorder=4,
              marker="^", edgecolors="white", linewidths=1.0, alpha=0.7,
              label="Upstream Controls (n=16)")

    # Draw dashed lines connecting each weir to its control
    weir_dict = {w["name_kr"]: w for w in weirs}
    for c in controls:
        w = weir_dict.get(c["weir_name_kr"])
        if w:
            ax.plot([w["lon"], c["lon"]], [w["lat"], c["lat"]],
                    '--', color="gray", linewidth=0.8, alpha=0.4, zorder=3)

    # Axis settings (bounds already set before basemap)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Study Area: 16 Weirs and Upstream Control Reaches\nAcross Four Major Rivers of South Korea",
                fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend
    legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.9,
                       edgecolor="gray", title="River System")
    legend.get_title().set_fontweight("bold")

    # Scale bar (approximate)
    scale_lon = lon_min + 0.3
    scale_lat = lat_min + 0.15
    # 1 degree lat ≈ 111 km at this latitude
    km50_deg = 50 / 111
    ax.plot([scale_lon, scale_lon + km50_deg], [scale_lat, scale_lat],
            'k-', linewidth=3)
    ax.text(scale_lon + km50_deg / 2, scale_lat + 0.05, "50 km",
            ha="center", fontsize=9, fontweight="bold")

    # North arrow
    arrow_x = lon_max - 0.3
    arrow_y = lat_max - 0.2
    ax.annotate("N", xy=(arrow_x, arrow_y + 0.15), fontsize=14, fontweight="bold",
               ha="center")
    ax.annotate("", xy=(arrow_x, arrow_y + 0.12), xytext=(arrow_x, arrow_y),
               arrowprops=dict(arrowstyle="->", linewidth=2, color="black"))

    # Inset info box
    info_text = ("Four Major Rivers Project\n"
                 "Construction: 2009-2012\n"
                 "16 weirs (treatment) +\n"
                 "16 upstream controls\n"
                 "BACI design: 2000-2025")
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                     edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    outpath = FIGURES_DIR / "study_area_map.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_study_area_map()
