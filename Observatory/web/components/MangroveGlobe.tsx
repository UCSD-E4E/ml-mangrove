'use client'

import { useEffect, useRef, useImperativeHandle } from 'react'
import maplibregl from 'maplibre-gl'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { buildMangroveLayer, HoverInfo } from '@/lib/layerBuilder'

const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_KEY ?? ''

export interface GlobeHandle {
  flyTo: (lng: number, lat: number, zoom: number) => void
}

interface MangroveGlobeProps {
  // React 19: ref is passed as a regular prop
  ref?: React.Ref<GlobeHandle>
  rotating: boolean
  selectedYear: number
  onHover: (info: HoverInfo | null) => void
  onFeaturesChange: (features: any[]) => void
}

function getCentroid(geometry: { type: string; coordinates: any }): [number, number] | null {
  let ring: number[][] | undefined
  if (geometry.type === 'MultiPolygon') {
    ring = geometry.coordinates?.[0]?.[0]
  } else {
    ring = geometry.coordinates?.[0]
  }
  if (!ring?.length) return null
  const lng = ring.reduce((s: number, c: number[]) => s + c[0], 0) / ring.length
  const lat = ring.reduce((s: number, c: number[]) => s + c[1], 0) / ring.length
  if (!isFinite(lng) || !isFinite(lat)) return null
  return [lng, lat]
}

export default function MangroveGlobe({
  ref,
  rotating,
  selectedYear,
  onHover,
  onFeaturesChange,
}: MangroveGlobeProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const overlayRef = useRef<MapboxOverlay | null>(null)
  const rafRef = useRef<number>(0)
  const rotatingRef = useRef(rotating)

  // Keep rotatingRef in sync without re-running the map effect
  useEffect(() => {
    rotatingRef.current = rotating
  }, [rotating])

  // Expose flyTo via ref (React 19 style)
  useImperativeHandle(ref, () => ({
    flyTo: (lng, lat, zoom) => {
      mapRef.current?.flyTo({
        center: [lng, lat],
        zoom,
        pitch: 60,
        duration: 2500,
        essential: true,
      })
    },
  }))

  // Rebuild Deck.gl layer when year or hover callbacks change
  useEffect(() => {
    if (!overlayRef.current) return
    overlayRef.current.setProps({
      layers: [
        buildMangroveLayer({
          selectedYear,
          onHover,
          onClick: (info) => {
            if (!mapRef.current || !info.object) return
            const center = getCentroid(info.object.geometry)
            if (!center) return
            mapRef.current.flyTo({ center, zoom: 13, pitch: 60, duration: 1800, essential: true })
          },
        }),
      ],
    })
  }, [selectedYear, onHover])

  // Initialise map once on mount
  useEffect(() => {
    if (!containerRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: `https://api.maptiler.com/maps/satellite/style.json?key=${MAPTILER_KEY}`,
      center: [-82, 27],
      zoom: 3,
      pitch: 30,
      bearing: 0,
    })
    mapRef.current = map

    map.on('error', (e) => console.error('[Globe] map error:', e))

    map.on('load', () => {
      // Deck.gl overlay
      const overlay = new MapboxOverlay({
        interleaved: false,
        layers: [
          buildMangroveLayer({
            selectedYear,
            onHover,
            onClick: (info) => {
              if (!info.object) return
              const center = getCentroid(info.object.geometry)
              if (!center) return
              map.flyTo({ center, zoom: 13, pitch: 60, duration: 1800, essential: true })
            },
          }),
        ],
      })
      map.addControl(overlay as any)
      overlayRef.current = overlay
    })

    // Auto-rotation RAF loop
    const rotate = () => {
      if (rotatingRef.current && mapRef.current) {
        mapRef.current.setBearing((mapRef.current.getBearing() + 0.025) % 360)
      }
      rafRef.current = requestAnimationFrame(rotate)
    }
    rafRef.current = requestAnimationFrame(rotate)

    map.on('movestart', () => { rotatingRef.current = false })
    map.on('moveend', () => {
      if (!overlayRef.current) return
      const canvas = map.getCanvas()
      const picked = overlayRef.current.pickObjects({
        x: 0,
        y: 0,
        width: canvas.clientWidth,
        height: canvas.clientHeight,
      })
      onFeaturesChange(picked.map((p: any) => p.object).filter(Boolean))
    })

    return () => {
      cancelAnimationFrame(rafRef.current)
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )
}
