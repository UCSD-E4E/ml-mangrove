"""
validate.py — Sanity-check Observatory/data/florida_mangroves.geojson

Checks:
  1. File loads and is a valid FeatureCollection
  2. All required properties present on every feature
  3. All geometries are valid (shapely)
  4. All centroids fall within the Florida bounding box
  5. Per-class polygon counts, total area, health_index distribution
  6. Saves a centroid scatter plot → Observatory/data/florida_centroids.png

Usage:
    python Observatory/pipeline/validate.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import shape

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parents[1] / "data"
GEOJSON     = DATA_DIR / "florida_mangroves.geojson"
PLOT_OUTPUT = DATA_DIR / "florida_centroids.png"

# ── Florida bounding box (WGS84) ──────────────────────────────────────────────
FL_MINX, FL_MAXX = -87.7, -79.8
FL_MINY, FL_MAXY =  24.4,  31.1

REQUIRED_PROPS = {"class_idx", "class_name", "health_index", "area_sqm", "year", "tile_id"}

CLASS_NAMES = [
    "Tree Cover", "Grassland", "Cropland", "Built-up",
    "Bare/Sparse Veg", "Water", "Wetland", "Mangrove",
]
CLASS_COLORS = [
    "#2d8a4e", "#c8d96f", "#e3c878", "#c0392b",
    "#d4b483", "#2471a3", "#76d7c4", "#1abc9c",
]

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"


def check(condition: bool, msg: str) -> bool:
    print(f"  {PASS if condition else FAIL}  {msg}")
    return condition


def main():
    all_ok = True

    # ── 1. Load file ──────────────────────────────────────────────────────────
    print("\n── 1. File loading ──────────────────────────────────────────────")
    if not GEOJSON.exists():
        print(f"  {FAIL}  File not found: {GEOJSON}")
        sys.exit(1)

    with open(GEOJSON) as f:
        fc = json.load(f)

    all_ok &= check(fc.get("type") == "FeatureCollection", "type == FeatureCollection")
    features = fc.get("features", [])
    all_ok &= check(len(features) > 0, f"Non-empty: {len(features):,} features")

    # ── 2. Required properties ────────────────────────────────────────────────
    print("\n── 2. Property completeness (all features) ──────────────────────")
    missing_props = [
        f["properties"]
        for f in features
        if not REQUIRED_PROPS.issubset(f.get("properties", {}).keys())
    ]
    all_ok &= check(len(missing_props) == 0,
                    f"All features have required props ({', '.join(sorted(REQUIRED_PROPS))})"
                    + (f"  — {len(missing_props)} missing" if missing_props else ""))

    # ── 3. Geometry validity (random sample of 200) ───────────────────────────
    print("\n── 3. Geometry validity (sample 200) ────────────────────────────")
    sample = random.sample(features, min(200, len(features)))
    invalid = []
    for feat in sample:
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            invalid.append(feat["properties"]["tile_id"])
    all_ok &= check(len(invalid) == 0,
                    f"All sampled geometries valid"
                    + (f"  — {len(invalid)} invalid in sample" if invalid else ""))

    # ── 4. Florida bounding box ───────────────────────────────────────────────
    print("\n── 4. Florida bounding box check (all centroids) ────────────────")
    out_of_bounds = []
    lons, lats = [], []
    for feat in features:
        cx, cy = shape(feat["geometry"]).centroid.coords[0]
        lons.append(cx)
        lats.append(cy)
        if not (FL_MINX <= cx <= FL_MAXX and FL_MINY <= cy <= FL_MAXY):
            out_of_bounds.append((cx, cy))
    all_ok &= check(len(out_of_bounds) == 0,
                    f"All centroids within Florida bbox"
                    + (f"  — {len(out_of_bounds)} outside" if out_of_bounds else ""))

    # ── 5. Statistics ─────────────────────────────────────────────────────────
    print("\n── 5. Per-class statistics ──────────────────────────────────────")
    counts       = Counter()
    total_area   = defaultdict(float)
    health_vals  = defaultdict(list)
    for feat in features:
        p = feat["properties"]
        name = p["class_name"]
        counts[name]      += 1
        total_area[name]  += p["area_sqm"]
        health_vals[name].append(p["health_index"])

    print(f"\n  {'Class':<20} {'Polygons':>9} {'Area (km²)':>12} {'NDVI mean':>10} {'NDVI std':>9}")
    print("  " + "-" * 65)
    for name in CLASS_NAMES:
        if name not in counts:
            continue
        n       = counts[name]
        area_km = total_area[name] / 1e6
        h       = np.array(health_vals[name])
        print(f"  {name:<20} {n:>9,} {area_km:>11.1f} {h.mean():>10.3f} {h.std():>9.3f}")
    print(f"\n  Total polygons : {len(features):,}")
    print(f"  Total area     : {sum(total_area.values()) / 1e6:,.1f} km²")

    # Health index overall
    all_health = [f["properties"]["health_index"] for f in features]
    h = np.array(all_health)
    print(f"\n  Health index   : min={h.min():.3f}  max={h.max():.3f}  "
          f"mean={h.mean():.3f}  median={np.median(h):.3f}")

    # ── 6. Centroid scatter plot ──────────────────────────────────────────────
    print("\n── 6. Generating centroid plot ──────────────────────────────────")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # Plot per class
    class_lons = defaultdict(list)
    class_lats = defaultdict(list)
    for feat, lon, lat in zip(features, lons, lats):
        class_lons[feat["properties"]["class_name"]].append(lon)
        class_lats[feat["properties"]["class_name"]].append(lat)

    for i, name in enumerate(CLASS_NAMES):
        if name not in class_lons:
            continue
        ax.scatter(class_lons[name], class_lats[name],
                   s=1, alpha=0.4, color=CLASS_COLORS[i], label=name, linewidths=0)

    # Florida bbox outline
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((FL_MINX, FL_MINY),
                            FL_MAXX - FL_MINX, FL_MAXY - FL_MINY,
                            linewidth=1, edgecolor="#555", facecolor="none",
                            linestyle="--"))

    ax.set_xlim(FL_MINX - 0.5, FL_MAXX + 0.5)
    ax.set_ylim(FL_MINY - 0.5, FL_MAXY + 0.5)
    ax.set_xlabel("Longitude", color="#aaa")
    ax.set_ylabel("Latitude", color="#aaa")
    ax.set_title("Florida Mangrove Observatory — Polygon Centroids", color="white", fontsize=13)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    legend = ax.legend(loc="lower left", fontsize=8, framealpha=0.3,
                       labelcolor="white", markerscale=6)
    legend.get_frame().set_facecolor("#1a1a2e")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOT_OUTPUT}")

    # ── Result ────────────────────────────────────────────────────────────────
    print(f"\n{'── ALL CHECKS PASSED ──' if all_ok else '── SOME CHECKS FAILED ──'}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
