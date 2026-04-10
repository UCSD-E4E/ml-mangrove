import { TileLayer } from '@deck.gl/geo-layers'
import { GeoJsonLayer } from '@deck.gl/layers'
import { MVTLoader } from '@loaders.gl/mvt'
import { parse } from '@loaders.gl/core'
import { PMTiles } from 'pmtiles'
import { getColor, CLASS_ELEVATION } from './classConfig'

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

let _pmtiles: PMTiles | null = null
// Minimum zoom level where this PMTiles archive actually has tile data.
// Set by warmPMTiles(); used to compute zoomOffset so masks appear at all viewport zooms.
let _pmtilesMinZoom = 0

function getPMTiles(): PMTiles {
  if (!_pmtiles) _pmtiles = new PMTiles(PMTILES_URL)
  return _pmtiles
}

/**
 * Pre-fetches the PMTiles archive header so the tile index is cached before
 * the first viewport tile request. Call this during the loading screen so
 * tile fetches are instant when the user enters the map.
 * Also records the archive's minZoom so buildMangroveLayer can request the
 * right tile zoom even when the viewport is zoomed out.
 */
export async function warmPMTiles(): Promise<void> {
  const header = await getPMTiles().getHeader()
  _pmtilesMinZoom = (header as any).minZoom ?? 0
}

export function buildMangroveLayer({ selectedYear, onHover, onClick }: LayerOptions) {
  const pmtiles = getPMTiles()

  // At viewport zoom MIN_VIEWPORT_ZOOM (Florida overview) we still want tiles from the
  // archive's minimum data zoom, so features appear even when fully zoomed out.
  // zoomOffset = how many extra zoom levels to add to the viewport zoom when fetching tiles.
  const MIN_VIEWPORT_ZOOM = 4
  const zoomOffset = Math.max(2, _pmtilesMinZoom - MIN_VIEWPORT_ZOOM)

  return new TileLayer({
    id: 'mangrove-landcover',
    minZoom: 0,
    maxZoom: 14,
    zoomOffset,
    maxCacheSize: 512,
    extent: [-88, 23, -79, 32],
    refinementStrategy: 'best-available',
    pickable: true,

    getTileData: async ({ index }: any) => {
      const { x, y, z } = index
      const tile = await pmtiles.getZxy(z, x, y)
      if (!tile?.data) return null
      return parse(tile.data as ArrayBuffer, MVTLoader, {
        worker: true,
        mvt: {
          coordinates: 'wgs84',
          tileIndex: { x, y, z },
          layers: ['landcover'],
        },
      })
    },

    onHover: (info: any) => {
      onHover(info?.object ? (info as HoverInfo) : null)
    },

    onClick: (info: any) => {
      if (info?.object) onClick(info as HoverInfo)
    },

    renderSubLayers: (props: any) => {
      const zoom: number = props.tile?.index?.z ?? 0
      const shouldExtrude = zoom >= 13

      return new GeoJsonLayer({
        ...props,
        id: `${props.id}-geojson`,

        getFillColor: (f: any) =>
          f.properties?.class_name === 'Water' ? [0, 0, 0, 0] : getColor(f.properties?.class_name),

        // No strokes — they trace tile-boundary clipping seams and cost an extra draw pass
        stroked: false,

        extruded: shouldExtrude,
        getElevation: (f: any) => {
          if (!shouldExtrude) return 0
          const name: string = f.properties?.class_name ?? ''
          if (name === 'Mangrove') return (f.properties?.health_index ?? 0) * 500
          return CLASS_ELEVATION[name as keyof typeof CLASS_ELEVATION] ?? 0
        },

        material: {
          ambient: 0.35,
          diffuse: 0.6,
          shininess: 32,
          specularColor: [60, 64, 70] as [number, number, number],
        },

        pickable: true,

        updateTriggers: {
          getFillColor: [selectedYear],
          getElevation: [selectedYear, shouldExtrude],
        },
      })
    },
  })
}
