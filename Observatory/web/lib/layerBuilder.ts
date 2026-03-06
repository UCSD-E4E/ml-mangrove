import { TileLayer } from '@deck.gl/geo-layers'
import { GeoJsonLayer } from '@deck.gl/layers'
import { MVTLoader } from '@loaders.gl/mvt'
import { parse } from '@loaders.gl/core'
import { PMTiles } from 'pmtiles'
import { getColor, getBrightColor } from './classConfig'

export interface LayerOptions {
  selectedYear: number
  onHover: (info: HoverInfo | null) => void
  onClick: (info: HoverInfo) => void
}

export interface HoverInfo {
  x: number
  y: number
  object: {
    properties: {
      class_name: string
      class_idx: number
      health_index: number
      area_sqm: number
      year: number
      tile_id: string
    }
    geometry: {
      type: string
      coordinates: number[][][]
    }
  }
}

const PMTILES_URL = process.env.NEXT_PUBLIC_PMTILES_URL ?? '/florida_mangroves.pmtiles'

// Singleton — one PMTiles handle per URL (avoids re-opening the archive on every layer rebuild)
let _pmtiles: PMTiles | null = null
function getPMTiles(): PMTiles {
  if (!_pmtiles) _pmtiles = new PMTiles(PMTILES_URL)
  return _pmtiles
}

export function buildMangroveLayer({ selectedYear, onHover, onClick }: LayerOptions) {
  const pmtiles = getPMTiles()

  return new TileLayer({
    id: 'mangrove-landcover',
    minZoom: 0,
    maxZoom: 14,
    pickable: true,

    // Fetch each tile directly from the PMTiles archive via byte-range requests
    getTileData: async ({ index }: any) => {
      const { x, y, z } = index
      const tile = await pmtiles.getZxy(z, x, y)
      if (!tile?.data) {
        console.log(`[Layer] tile ${z}/${x}/${y} → no data`)
        return null
      }
      const features = await parse(tile.data as ArrayBuffer, MVTLoader, {
        mvt: {
          coordinates: 'wgs84',
          tileIndex: { x, y, z },
          layers: ['landcover'],
        },
      })
      const arr = features as any[]
      if (arr?.length > 0) {
        const sample = arr[0]
        const coord = sample?.geometry?.coordinates?.[0]?.[0]
        console.log(`[Layer] tile ${z}/${x}/${y} → ${arr.length} features, sample coord:`, coord)
      } else {
        console.log(`[Layer] tile ${z}/${x}/${y} → 0 features (parsed empty)`)
      }
      return features
    },

    onHover: (info: any) => {
      onHover(info?.object ? (info as HoverInfo) : null)
    },

    onClick: (info: any) => {
      if (info?.object) onClick(info as HoverInfo)
    },

    // Render each tile's GeoJSON features as an extruded polygon layer
    renderSubLayers: (props: any) => {
      return new GeoJsonLayer({
        ...props,
        id: `${props.id}-geojson`,

        // Fill color by class — Water hidden
        getFillColor: (f: any) =>
          f.properties?.class_name === 'Water' ? [0, 0, 0, 0] : getColor(f.properties?.class_name),

        // Edge glow — Water hidden
        getLineColor: (f: any) =>
          f.properties?.class_name === 'Water' ? [0, 0, 0, 0] : getBrightColor(f.properties?.class_name, 180),
        lineWidthMinPixels: 0.5,

        // 3D extrusion — Mangrove only
        extruded: true,
        getElevation: (f: any) =>
          f.properties?.class_name === 'Mangrove'
            ? (f.properties?.health_index ?? 0) * 500
            : 0,

        material: {
          ambient: 0.35,
          diffuse: 0.6,
          shininess: 32,
          specularColor: [60, 64, 70] as [number, number, number],
        },

        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 40],

        updateTriggers: {
          getFillColor: [selectedYear],
          getElevation: [selectedYear],
        },
      })
    },
  })
}
