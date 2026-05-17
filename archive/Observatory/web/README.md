# Global Mangrove Observatory

An interactive 3D web platform for exploring satellite-derived mangrove land-cover maps of Florida at 10-metre resolution. Built by the [Engineers for Exploration](https://e4e.ucsd.edu/) lab at UC San Diego.

![Platform Screenshot](public/image.png)

---

## Overview

The Observatory renders machine-learning classification results from a SegFormer model trained on Sentinel-2 satellite imagery directly on an interactive globe. Every polygon on the map represents a segment of land classified into one of eight categories: Mangrove, Tree Cover, Wetland, Built-up, Grassland, Cropland, Bare or Sparse Vegetation, and Water.

At high zoom levels the masks extrude into three dimensions. Mangrove polygons rise to heights that reflect their health index, so the spatial distribution of ecosystem stress is immediately visible. All other classes extrude to fixed representative heights for clear visual differentiation. The analytics panel on the right tracks area, health, and class composition for whatever is currently visible in the viewport.

---

## Features

- GPU-accelerated polygon rendering via Deck.gl — handles millions of vertices at 60 fps
- PMTiles-based delivery with no tile server required at runtime
- Adaptive tile zoom offset so masks appear at all viewport zoom levels
- Seamless zoom transitions with best-available tile refinement
- Health-driven 3D extrusion for mangrove polygons at zoom 11 and above
- Live viewport analytics: total area, mangrove area, mean health index, class breakdown
- Hover tooltip with per-polygon class, health, area, and tile metadata
- Auto-rotating landing globe with animated entry sequence
- Click-to-fly: clicking any polygon flies the camera to that location

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router) + React 19 |
| Language | TypeScript (strict) |
| Map engine | MapLibre GL 5 |
| Vector rendering | Deck.gl 9 |
| Tile format | PMTiles + Mapbox Vector Tiles (MVT) |
| Tile parsing | @loaders.gl/mvt (web worker) |
| Base imagery | MapTiler Satellite |
| UI animation | Framer Motion 12 + GSAP 3 |
| Styling | Tailwind CSS 4 |

---

## Getting Started

### Prerequisites

You will need Node.js 20 or later and npm. A MapTiler API key is required for the satellite base map — a free account at [maptiler.com](https://www.maptiler.com/) is sufficient for development.

### Installation

Clone the repository and navigate to the web directory.

```bash
git clone https://github.com/UCSD-E4E/ml-mangrove.git
cd ml-mangrove/Observatory/web
npm install
```

### Environment Setup

Create a `.env.local` file in the `Observatory/web` directory with the following contents. Never commit this file.

```
NEXT_PUBLIC_MAPTILER_KEY=your_maptiler_key_here
NEXT_PUBLIC_PMTILES_URL=/florida_mangroves.pmtiles
```

The second variable is optional if you are using the PMTiles file from the `public` directory. Set it to a CDN URL in production deployments.

### Running Locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser. The first load may take a moment while the PMTiles header is fetched; subsequent loads are fast because the header is cached.

### Production Build

```bash
npm run build
npm run start
```

---

## Data

The land-cover data is stored in `public/florida_mangroves.pmtiles` (358 MB). This archive contains Mapbox Vector Tile pyramids from zoom level 0 to 14. Each tile contains a `landcover` layer with polygon features carrying the following properties.

| Property | Type | Description |
|---|---|---|
| `class_name` | string | Land-cover class (e.g. `"Mangrove"`) |
| `class_idx` | number | Numeric class index 0–7 |
| `health_index` | number | Mangrove health score 0.0–1.0 |
| `area_sqm` | number | Polygon area in square metres |
| `year` | number | Data year (currently 2023) |
| `tile_id` | string | Source tile identifier |

---

## Project Structure

```
Observatory/web/
├── app/
│   ├── layout.tsx          Root HTML layout, metadata, global styles
│   └── page.tsx            App orchestrator: stage machine, global state
├── components/
│   ├── MangroveGlobe.tsx   MapLibre + Deck.gl map, hover, feature picking
│   ├── AnalyticsPanel.tsx  Viewport statistics sidebar
│   ├── HoverTooltip.tsx    Per-polygon hover tooltip
│   ├── LandingOverlay.tsx  Welcome screen with GSAP entry animations
│   ├── LoadingScreen.tsx   Animated progress overlay during initialisation
│   └── TimeSlider.tsx      Year navigation control
├── lib/
│   ├── classConfig.ts      Class taxonomy, colours, elevations, icons
│   └── layerBuilder.ts     Deck.gl TileLayer construction, PMTiles integration
└── public/
    └── florida_mangroves.pmtiles   Land-cover tile archive (358 MB)
```

For a detailed explanation of how each part works, see [doc.md](../doc.md).

---

## Configuration Reference

### Class Elevation Heights

Extrusion heights are defined in `lib/classConfig.ts` and are in Deck.gl map units (visually exaggerated for globe scale). Mangrove height is dynamic and is not read from this table.

| Class | Height |
|---|---|
| Built-up | 150 |
| Tree Cover | 120 |
| Wetland | 60 |
| Grassland | 25 |
| Cropland | 15 |
| Bare / Sparse Vegetation | 8 |
| Water | 0 |
| Mangrove | `health_index × 500` |

### Tile Layer Defaults

These values are set in `lib/layerBuilder.ts` and can be tuned for different hardware targets.

| Parameter | Value | Effect |
|---|---|---|
| `maxCacheSize` | 512 | Tiles kept in VRAM; increase for GPUs with more memory |
| `maxZoom` | 14 | Finest tile zoom in the archive |
| `zoomOffset` | dynamic | Computed from archive `minZoom` to ensure coverage at all viewport zooms |
| `extent` | Florida bbox | Prevents out-of-area tile fetches |
| `refinementStrategy` | `best-available` | Shows nearest cached tile during transitions |

---

## Deployment

The application can be deployed to any platform that supports Next.js. For production:

1. Host the `florida_mangroves.pmtiles` file on a CDN that supports HTTP Range requests (Cloudflare R2, AWS S3, or similar). Most CDNs support this by default.
2. Set `NEXT_PUBLIC_PMTILES_URL` to the CDN URL of the file.
3. Set `NEXT_PUBLIC_MAPTILER_KEY` to a production MapTiler key with appropriate domain restrictions.
4. Run `npm run build` and deploy the `.next` output to Vercel, a Docker container, or your platform of choice.

Serving the PMTiles file from a CDN rather than the Next.js server is strongly recommended for production traffic, as repeated 358 MB file serving would strain any application server.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for anything beyond a small bug fix, so the approach can be discussed first.

1. Fork the repository and create a branch from `master`.
2. Make your changes inside `Observatory/web`.
3. Ensure `npm run build` succeeds without TypeScript errors.
4. Open a pull request with a clear description of what changed and why.

---

## Related

The Observatory is one component of the larger ml-mangrove project. See the [root README](../../README.md) for documentation on the SegFormer training pipeline, the GEE satellite data pipeline, the drone classification models, and the ArcGIS Pro toolbox.

---

## Acknowledgements

Developed by the Engineers for Exploration (E4E) lab at the University of California San Diego. Land-cover data derived from ESA WorldCover using Sentinel-2 imagery processed through a SegFormer segmentation model trained by the E4E Mangrove Monitoring team. Base satellite imagery provided by MapTiler.
