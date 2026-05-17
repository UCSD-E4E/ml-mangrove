"""
vectorize.py — Convert Florida GEE training GeoTIFFs into a GeoJSON FeatureCollection.

Reads each 69-band tile from Florida_Training_Dataset/, extracts band 69 (ESA WorldCover
labels), vectorizes all 8 active classes, and bakes in health_index (NDVI), area_sqm,
year, class metadata for each polygon.

Usage:
    python Observatory/pipeline/vectorize.py

Output:
    Observatory/data/florida_mangroves.geojson
"""

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import shape
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
TILES_DIR      = REPO_ROOT / "Scale Adaption" / "GEE" / "Florida_Training_Dataset"
OUTPUT_DIR     = Path(__file__).resolve().parents[1] / "data"
OUTPUT_GEOJSON = OUTPUT_DIR / "florida_mangroves.geojson"

# ── Class definitions (must match segformer_training.ipynb exactly) ───────────
# ESA values 20 (Shrubland), 70 (Snow/Ice), 100 (Moss/Lichen) are absent in
# Florida and are excluded (→ IGNORE_INDEX 255 in the model).
ESA_TO_CLASS = {
    10: 0,   # Tree Cover
    30: 1,   # Grassland
    40: 2,   # Cropland
    50: 3,   # Built-up
    60: 4,   # Bare / Sparse Vegetation
    80: 5,   # Water
    90: 6,   # Wetland
    95: 7,   # Mangrove
}
CLASS_NAMES = [
    "Tree Cover",        # 0  ESA 10
    "Grassland",         # 1  ESA 30
    "Cropland",          # 2  ESA 40
    "Built-up",          # 3  ESA 50
    "Bare/Sparse Veg",   # 4  ESA 60
    "Water",             # 5  ESA 80
    "Wetland",           # 6  ESA 90
    "Mangrove",          # 7  ESA 95
]

YEAR        = 2023   # default year for all Florida training tiles
WGS84       = CRS.from_epsg(4326)
LABEL_BAND  = 69     # 1-indexed (rasterio convention)
RED_BAND    = 1      # Sentinel-2 Red
NIR_BAND    = 4      # Sentinel-2 NIR


# ── Per-tile helper ────────────────────────────────────────────────────────────

def _mean_ndvi_in_window(ndvi: np.ndarray,
                         geom_dict: dict,
                         src_transform,
                         win_row_off: int,
                         win_col_off: int) -> float:
    """
    Compute mean NDVI for a polygon using a pre-clipped NDVI window.

    ndvi           — 2-D float32 array for the bounding-box window
    geom_dict      — polygon geometry in source CRS (GeoJSON-like dict)
    src_transform  — affine transform of the *full* tile
    win_row_off    — row offset of the window within the full tile
    win_col_off    — col offset of the window within the full tile
    """
    if ndvi.size == 0:
        return 0.0

    # Build the affine transform that corresponds to this window
    win_transform = src_transform * src_transform.__class__.translation(
        win_col_off, win_row_off
    )
    mask = rasterio.features.geometry_mask(
        [geom_dict],
        out_shape=ndvi.shape,
        transform=win_transform,
        invert=True,       # True inside polygon
    )
    vals = ndvi[mask]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def vectorize_tile(tif_path: Path) -> list:
    """
    Vectorize one 69-band GeoTIFF tile into a list of GeoJSON Feature dicts.

    Each feature carries:
        class_idx    int   model class index (0–7)
        class_name   str   human-readable class name
        health_index float mean NDVI within the polygon (−1 … 1, clamped)
        area_sqm     float polygon area in source CRS metres²
        year         int   fixed to YEAR constant
        tile_id      str   source filename stem
    """
    features = []
    tile_id  = tif_path.stem

    with rasterio.open(tif_path) as src:
        # -- Read required bands (1-indexed) --------------------------------
        labels = src.read(LABEL_BAND)                                # uint8 / int
        red    = src.read(RED_BAND).astype(np.float32)               # reflectance ×10000
        nir    = src.read(NIR_BAND).astype(np.float32)

        # -- Compute per-pixel NDVI ----------------------------------------
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)

        src_transform = src.transform
        src_crs       = src.crs
        src_height    = src.height
        src_width     = src.width

        # -- Build mask of all valid (in-taxonomy) label pixels ------------
        valid_mask = np.zeros(labels.shape, dtype=np.uint8)
        for esa_val in ESA_TO_CLASS:
            valid_mask[labels == esa_val] = 1

        # -- Vectorize all connected regions in one pass -------------------
        # shapes() yields (geom_dict, pixel_value) for each connected component
        raw_shapes = list(
            rasterio.features.shapes(
                labels.astype(np.int32),
                mask=valid_mask,
                transform=src_transform,
            )
        )

        for geom_dict, esa_val_f in raw_shapes:
            esa_val = int(esa_val_f)
            if esa_val not in ESA_TO_CLASS:
                continue

            class_idx  = ESA_TO_CLASS[esa_val]
            class_name = CLASS_NAMES[class_idx]

            # Area in source CRS (UTM metres²)
            geom_utm  = shape(geom_dict)
            area_sqm  = float(geom_utm.area)

            # NDVI: clip to polygon bounding box for efficiency
            minx, miny, maxx, maxy = geom_utm.bounds
            # Convert spatial bounds → pixel row/col window
            row_off, col_off = src.index(minx, maxy)   # top-left corner
            row_end, col_end = src.index(maxx, miny)   # bottom-right corner

            row_off = max(0, int(row_off))
            col_off = max(0, int(col_off))
            row_end = min(src_height, int(row_end) + 1)
            col_end = min(src_width,  int(col_end) + 1)

            ndvi_clip    = ndvi[row_off:row_end, col_off:col_end]
            health_index = _mean_ndvi_in_window(
                ndvi_clip, geom_dict, src_transform, row_off, col_off
            )

            # Reproject geometry to WGS84 for GeoJSON output
            geom_wgs84 = transform_geom(src_crs, WGS84, geom_dict)

            features.append({
                "type": "Feature",
                "geometry": geom_wgs84,
                "properties": {
                    "class_idx":    class_idx,
                    "class_name":   class_name,
                    "health_index": round(health_index, 4),
                    "area_sqm":     round(area_sqm, 2),
                    "year":         YEAR,
                    "tile_id":      tile_id,
                },
            })

    return features


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tile_paths = sorted(TILES_DIR.glob("*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No .tif files found in {TILES_DIR}")

    print(f"Found {len(tile_paths)} tiles in {TILES_DIR}")
    print(f"Output → {OUTPUT_GEOJSON}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    for tif_path in tqdm(tile_paths, desc="Vectorizing tiles", unit="tile"):
        tile_features = vectorize_tile(tif_path)
        all_features.extend(tile_features)
        tqdm.write(f"  {tif_path.name}: {len(tile_features)} polygons")

    feature_collection = {
        "type":     "FeatureCollection",
        "features": all_features,
    }

    with open(OUTPUT_GEOJSON, "w") as f:
        json.dump(feature_collection, f, separators=(",", ":"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\nDone. Total polygons: {len(all_features)}")

    from collections import Counter
    counts = Counter(f["properties"]["class_name"] for f in all_features)
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<20}: {cnt:>6} polygons")

    print(f"\nOutput written to: {OUTPUT_GEOJSON}")


if __name__ == "__main__":
    main()
