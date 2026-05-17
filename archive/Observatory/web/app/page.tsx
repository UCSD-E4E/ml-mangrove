'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import dynamic from 'next/dynamic'
import type { GlobeHandle } from '@/components/MangroveGlobe'
import LandingOverlay from '@/components/LandingOverlay'
import AnalyticsPanel from '@/components/AnalyticsPanel'
import TimeSlider from '@/components/TimeSlider'
import LoadingScreen from '@/components/LoadingScreen'
import { warmPMTiles } from '@/lib/layerBuilder'
import { AnimatePresence } from 'framer-motion'

// MapLibre + Deck.gl must be client-only (they need window/WebGL)
const MangroveGlobe = dynamic(() => import('@/components/MangroveGlobe'), { ssr: false })

const YEAR_MIN = 2023
const YEAR_MAX = 2023

type Stage = 'loading' | 'landing' | 'exploring'

export default function Page() {
  const [stage, setStage] = useState<Stage>('loading')
  const [visibleFeatures, setVisibleFeatures] = useState<any[]>([])
  const [selectedYear, setSelectedYear] = useState(2023)
  const [mapReady, setMapReady] = useState(false)
  const globeRef = useRef<GlobeHandle>(null)

  // Kick off PMTiles header pre-fetch as early as possible
  useEffect(() => {
    warmPMTiles().catch(() => {/* non-fatal if it fails */})
  }, [])

  // Loading screen is done when map has fired its load event
  const handleMapReady = useCallback(() => {
    setMapReady(true)
  }, [])

  const handleLoadingComplete = useCallback(() => {
    setStage('landing')
  }, [])

  const handleExplore = useCallback(() => {
    setStage('exploring')
    globeRef.current?.flyTo(-80.5, 25.5, 8)
  }, [])

  const handleFeaturesChange = useCallback((features: any[]) => {
    setVisibleFeatures(features)
  }, [])

  return (
    <div className="relative w-full bg-[#0d1117]" style={{ height: '100dvh' }}>
      {/* Map is always mounted so it initialises behind the loading screen */}
      <MangroveGlobe
        ref={globeRef}
        rotating={stage === 'landing'}
        selectedYear={selectedYear}
        onFeaturesChange={handleFeaturesChange}
        onReady={handleMapReady}
      />

      {/* Loading screen — sits on top until map + PMTiles are warm */}
      {stage === 'loading' && (
        <LoadingScreen ready={mapReady} onComplete={handleLoadingComplete} />
      )}

      {/* Landing overlay */}
      <AnimatePresence>
        {stage === 'landing' && <LandingOverlay onExplore={handleExplore} />}
      </AnimatePresence>

      {/* Post-landing UI */}
      <AnimatePresence>
        {stage === 'exploring' && (
          <>
            <AnalyticsPanel features={visibleFeatures} />
            <TimeSlider
              yearMin={YEAR_MIN}
              yearMax={YEAR_MAX}
              selectedYear={selectedYear}
              onYearChange={setSelectedYear}
            />
          </>
        )}
      </AnimatePresence>

    </div>
  )
}
