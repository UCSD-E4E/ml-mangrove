"""
Unified GEE tile export script — reads per-region config from config/regions.yaml.

Modes:
  --dry-run          Print candidate tile stats only, no download or export.
  --local            Download tiles to local disk (for verifying 1-2 tiles before GCS run).
  (default)          Submit GEE batch export tasks directly to GCS. Fire-and-forget.

Examples:
  # Check what tiles would be selected (no download)
  python export_to_gcs.py --region brazil --dry-run

  # Download 2 tiles locally to verify correctness
  python export_to_gcs.py --region brazil --local --limit 2 --out-dir /tmp/brazil_test

  # Export all 100 tiles to GCS
  python export_to_gcs.py --region brazil --bucket e4e-mangrove-tiles
"""

import argparse
import os
import random
import sys

import ee
import geemap
import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'regions.yaml')
MAX_CANDIDATES = 1000  # toList cap — raise if a region has an unusually dense mangrove coast


def load_region_config(region_name: str) -> dict:
    with open(CONFIG_PATH) as f:
        configs = yaml.safe_load(f)
    if region_name not in configs:
        raise ValueError(
            f"Unknown region '{region_name}'. Available: {list(configs.keys())}"
        )
    return configs[region_name]


def build_geometry(cfg: dict) -> tuple[ee.Geometry, ee.Geometry]:
    """Returns (region_geometry, bounding_box)."""
    if cfg.get('bbox'):
        lon_min, lat_min, lon_max, lat_max = cfg['bbox']
        geom = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        return geom, geom

    cf = cfg['country_filter']
    countries = ee.FeatureCollection(cf['dataset'])

    if cfg.get('multi_country'):
        merged = None
        for name in cfg['multi_country']:
            fc = countries.filter(ee.Filter.eq(cf['field'], name))
            merged = fc if merged is None else merged.merge(fc)
        region = merged.union(maxError=1000).geometry()
    else:
        region = countries.filter(ee.Filter.eq(cf['field'], cf['value'])).geometry()

    return region, region.bounds()


def build_training_stack(bounds: ee.Geometry, cfg: dict) -> tuple[ee.Image, ee.Image]:
    """Returns (training_stack_69band, esa_all_classes)."""
    date_start, date_end = cfg['date_range']

    esa = ee.ImageCollection('ESA/WorldCover/v200').first()
    esa_all_classes = esa.select('Map').byte().rename('ESA_MultiClass_Label')

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(bounds)
          .filterDate(date_start, date_end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cfg['cloud_pct_max']))
          .median())
    s2_optical = s2.select(['B4', 'B3', 'B2', 'B8']).rename(['Red', 'Green', 'Blue', 'NIR'])

    embeddings = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                  .filterBounds(bounds)
                  .filterDate(date_start, date_end)
                  .mosaic())

    # Band layout: 1-4 optical, 5-68 embeddings, 69 ESA label
    # Cast entire stack to Float32 — S2 is Float64, embeddings are Float32.
    # esa_all_classes stays as Byte for the tile-filtering histogram step.
    training_stack = (s2_optical.addBands(embeddings)
                                .addBands(esa_all_classes.toFloat())
                                .toFloat())
    return training_stack, esa_all_classes


def get_candidate_tiles(
    region_geom: ee.Geometry,
    bounds: ee.Geometry,
    esa_all_classes: ee.Image,
    cfg: dict,
) -> list[dict]:
    """
    Two-pass candidate scan:
      Pass 1 (server-side): max-reducer at 1000m — presence check only, keeps tiles with any mangrove.
      Pass 2 (server-side + Python): frequencyHistogram at 500m — quality filter via cfg thresholds.

    Scale choices avoid sync timeout on large bounding boxes (e.g. Brazil ~48k tiles):
      1000m = 100x fewer pixels than 100m for Pass 1 (just need presence, not fractions).
      500m  = 16x fewer pixels than 100m for Pass 2 (fractions accurate enough for filtering).
    """
    esa_mangrove = ee.ImageCollection('ESA/WorldCover/v200').first().eq(95)

    # Cover the bounding box — no complex polygon intersection here.
    # Pass 1 naturally excludes non-mangrove tiles via the max reducer.
    base_grid = bounds.coveringGrid('EPSG:3857', 20480)

    # Pass 1: coarse presence check — scale=1000 is 100x faster than 100m and
    # sufficient to detect any mangrove pixels in a 20km tile.
    # Do NOT call .size().getInfo() here; it forces full evaluation and times out
    # on large bounding boxes. Let the result stay lazy until Pass 2 triggers it.
    tiles_with_mangrove = esa_mangrove.reduceRegions(
        collection=base_grid,
        reducer=ee.Reducer.max(),
        scale=1000,
        tileScale=4,
    ).filter(ee.Filter.gt('max', 0))

    # Pass 2: class histogram for quality filtering — scale=500 gives accurate
    # enough class fractions for threshold checks (e.g. ">70% water?").
    class_stats = esa_all_classes.reduceRegions(
        collection=tiles_with_mangrove,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=500,
        tileScale=4,
    ).map(lambda f: f.setGeometry(f.geometry().transform('EPSG:4326', 1)))

    raw = class_stats.toList(MAX_CANDIDATES).getInfo()
    if not raw:
        return []
    print(f"  Tiles with mangrove (pre-filter): {len(raw)}"
          + (" [WARNING: hit cap, some tiles excluded]" if len(raw) == MAX_CANDIDATES else ""))

    # Python-side filter — easy for team members to tune via regions.yaml
    kept = []
    for feat in raw:
        raw_hist = feat['properties'].get('histogram', {})
        hist = {int(float(k)): int(float(v)) for k, v in raw_hist.items()}
        total = sum(hist.values())
        if total == 0:
            continue

        mangrove_pct   = hist.get(95, 0) / total * 100
        water_pct      = hist.get(80, 0) / total * 100
        tree_pct       = hist.get(10, 0) / total * 100
        bare_built_pct = (hist.get(60, 0) + hist.get(50, 0)) / total * 100
        num_classes    = len(hist)

        if mangrove_pct   < cfg['min_mangrove_pct']:    continue
        if water_pct      > cfg['max_water_pct']:       continue
        if tree_pct       > cfg['max_tree_pct']:        continue
        if bare_built_pct > cfg['max_bare_built_pct']:  continue
        if num_classes    < cfg['min_classes']:         continue

        kept.append({
            'geometry':     feat['geometry'],
            'mangrove_pct': round(mangrove_pct, 1),
            'water_pct':    round(water_pct, 1),
            'tree_pct':     round(tree_pct, 1),
            'bare_built_pct': round(bare_built_pct, 1),
            'num_classes':  num_classes,
            'hist':         hist,
        })

    return kept


def print_tile_stats(tiles: list[dict], label: str = "") -> None:
    if not tiles:
        print(f"  {label}: 0 tiles")
        return
    avg = lambda key: sum(t[key] for t in tiles) / len(tiles)
    print(f"  {label}: {len(tiles)} tiles | "
          f"mangrove={avg('mangrove_pct'):.1f}% "
          f"water={avg('water_pct'):.1f}% "
          f"tree={avg('tree_pct'):.1f}% "
          f"classes={avg('num_classes'):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export GEE training tiles to GCS or locally.'
    )
    parser.add_argument('--region',  required=True,
                        help='Region key in regions.yaml (e.g. brazil)')
    parser.add_argument('--bucket',  default=None,
                        help='GCS bucket name (required for GCS export)')
    parser.add_argument('--limit',   type=int, default=None,
                        help='Override sample_size — useful for local testing (e.g. --limit 2)')
    parser.add_argument('--local',   action='store_true',
                        help='Download tiles locally instead of exporting to GCS')
    parser.add_argument('--out-dir', default=None,
                        help='Local output directory (defaults to <region>_training_dataset/)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print tile stats only — no download or export')
    args = parser.parse_args()

    if not args.dry_run and not args.local and not args.bucket:
        parser.error('--bucket is required for GCS export (or pass --local / --dry-run)')

    cfg = load_region_config(args.region)
    print(f"\n=== {cfg['display_name']} ===")

    ee.Authenticate()
    ee.Initialize(project='e4e-mangrove')

    print("\nBuilding geometry and training stack...")
    region_geom, bounds = build_geometry(cfg)
    training_stack, esa_all_classes = build_training_stack(bounds, cfg)

    print("\nScanning candidate tiles...")
    candidates = get_candidate_tiles(region_geom, bounds, esa_all_classes, cfg)
    print_tile_stats(candidates, "After quality filters")

    if not candidates:
        print("No tiles passed filters. Relax thresholds in config/regions.yaml.")
        sys.exit(1)

    random.seed(cfg['seed'])
    random.shuffle(candidates)
    n = args.limit if args.limit is not None else cfg['sample_size']
    selected = candidates[:n]
    print_tile_stats(selected, f"Selected for download (n={n})")

    if args.dry_run:
        print("\nDry run complete. No files written.")
        return

    if args.local:
        _download_local(selected, training_stack, cfg, args)
    else:
        _export_to_gcs(selected, training_stack, cfg, args)


def _download_local(
    tiles: list[dict],
    training_stack: ee.Image,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    out_dir = args.out_dir or os.path.join(
        os.getcwd(), f"{args.region}_training_dataset"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nDownloading {len(tiles)} tiles locally → {out_dir}")

    for i, tile in enumerate(tiles):
        filename = os.path.join(out_dir, f"{cfg['output_prefix']}_{i+1:03d}.tif")
        if os.path.exists(filename):
            print(f"  [{i+1}/{len(tiles)}] Exists, skipping.")
            continue
        print(f"  [{i+1}/{len(tiles)}] "
              f"mangrove={tile['mangrove_pct']}% "
              f"water={tile['water_pct']}% "
              f"tree={tile['tree_pct']}% "
              f"classes={tile['num_classes']}")
        try:
            geemap.download_ee_image(
                image=training_stack,
                filename=filename,
                region=ee.Geometry(tile['geometry']),
                crs='EPSG:3857',
                scale=10,
            )
            print(f"    Saved: {filename}")
        except Exception as e:
            print(f"    Failed: {e}")


def _export_to_gcs(
    tiles: list[dict],
    training_stack: ee.Image,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    bucket = args.bucket
    gcs_prefix = cfg['gcs_prefix']
    print(f"\nSubmitting {len(tiles)} export tasks → gs://{bucket}/{gcs_prefix}/")

    for i, tile in enumerate(tiles):
        task_name = f"{cfg['output_prefix']}_{i+1:03d}"
        geom = ee.Geometry(tile['geometry'])
        task = ee.batch.Export.image.toCloudStorage(
            image=training_stack.clip(geom),
            description=task_name,
            bucket=bucket,
            fileNamePrefix=f"{gcs_prefix}/{task_name}",
            region=geom,
            crs='EPSG:3857',
            scale=10,
            maxPixels=1e10,
            fileFormat='GeoTIFF',
        )
        task.start()
        print(f"  [{i+1}/{len(tiles)}] Submitted: {task_name}")

    print(f"\nAll tasks submitted. Monitor at: https://code.earthengine.google.com/tasks")


if __name__ == '__main__':
    main()
