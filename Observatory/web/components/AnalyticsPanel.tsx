'use client'

import { motion, useSpring, useTransform } from 'framer-motion'
import { useMemo, useEffect } from 'react'
import { CLASS_NAMES, CLASS_HEX, CLASS_ICONS } from '@/lib/classConfig'

interface Feature {
  properties: {
    class_name: string
    health_index: number
    area_sqm: number
  }
}

interface AnalyticsPanelProps {
  features: Feature[]
}

function AnimatedNumber({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const spring = useSpring(value, { stiffness: 80, damping: 20 })
  const display = useTransform(spring, (v) => v.toFixed(decimals))

  useEffect(() => {
    spring.set(value)
  }, [spring, value])

  return <motion.span>{display}</motion.span>
}

export default function AnalyticsPanel({ features }: AnalyticsPanelProps) {
  const stats = useMemo(() => {
    if (!features.length) return null

    const totalArea = features.reduce((s, f) => s + f.properties.area_sqm, 0)
    const mangroveArea = features
      .filter((f) => f.properties.class_name === 'Mangrove')
      .reduce((s, f) => s + f.properties.area_sqm, 0)
    const meanHealth =
      features.reduce((s, f) => s + f.properties.health_index, 0) / features.length

    // Per-class counts
    const classCounts: Record<string, number> = {}
    for (const f of features) {
      classCounts[f.properties.class_name] = (classCounts[f.properties.class_name] ?? 0) + 1
    }

    // Only classes with hits, sorted by count desc
    const presentClasses = CLASS_NAMES.filter((n) => classCounts[n] > 0).sort(
      (a, b) => (classCounts[b] ?? 0) - (classCounts[a] ?? 0)
    )

    return { totalArea, mangroveArea, meanHealth, classCounts, presentClasses, total: features.length }
  }, [features])

  return (
    <motion.aside
      initial={{ x: 320, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.1 }}
      className="fixed right-4 top-4 bottom-4 w-72 z-10 flex flex-col gap-4
                 bg-[#0a0f1a]/90 border border-white/10 rounded-2xl p-5 overflow-y-auto"
    >
      {/* Header */}
      <div>
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <span>🌴</span>
          <span>Live Analytics</span>
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">Current viewport · Florida</p>
      </div>

      <div className="w-full h-px bg-white/10" />

      {stats ? (
        <>
          {/* Key metrics grid */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="Total Area">
              <AnimatedNumber value={stats.totalArea / 10000} decimals={0} />
              <span className="text-slate-400 text-xs ml-1">ha</span>
            </MetricCard>

            <MetricCard label="Mangrove Area" accent="#1abc9c">
              <AnimatedNumber value={stats.mangroveArea / 10000} decimals={0} />
              <span className="text-slate-400 text-xs ml-1">ha</span>
            </MetricCard>

            <MetricCard label="Mean Health">
              <div className="flex items-center gap-2 w-full">
                <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    animate={{ width: `${stats.meanHealth * 100}%` }}
                    style={{ backgroundColor: stats.meanHealth > 0.6 ? '#22c55e' : stats.meanHealth > 0.3 ? '#eab308' : '#ef4444' }}
                  />
                </div>
                <span className="text-white font-mono text-xs tabular-nums">
                  <AnimatedNumber value={stats.meanHealth} decimals={2} />
                </span>
              </div>
            </MetricCard>

            <MetricCard label="Polygons">
              <AnimatedNumber value={stats.total} decimals={0} />
            </MetricCard>
          </div>

          <div className="w-full h-px bg-white/10" />

          {/* Class breakdown */}
          <div>
            <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Class breakdown</p>

            {/* Stacked bar */}
            <div className="flex h-2 rounded-full overflow-hidden mb-3">
              {stats.presentClasses.map((name) => (
                <div
                  key={name}
                  title={name}
                  style={{
                    width: `${((stats.classCounts[name] ?? 0) / stats.total) * 100}%`,
                    backgroundColor: CLASS_HEX[name as keyof typeof CLASS_HEX],
                  }}
                />
              ))}
            </div>

            {/* Legend */}
            <div className="flex flex-col gap-1.5">
              {stats.presentClasses.map((name) => {
                const count = stats.classCounts[name] ?? 0
                const pct = ((count / stats.total) * 100).toFixed(1)
                return (
                  <div key={name} className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <div
                        className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                        style={{ backgroundColor: CLASS_HEX[name as keyof typeof CLASS_HEX] }}
                      />
                      <span className="text-xs text-slate-300">
                        {CLASS_ICONS[name as keyof typeof CLASS_ICONS]} {name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">{pct}%</span>
                      <span className="text-xs text-slate-400 font-mono tabular-nums w-12 text-right">
                        {count.toLocaleString()}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-xs text-slate-600 text-center">
            Pan the map to load polygons
          </p>
        </div>
      )}

      {/* Footer */}
      <div className="mt-auto pt-2 border-t border-white/5">
        <p className="text-xs text-slate-600 text-center">
          Powered by E4E Lab · UCSD
        </p>
      </div>
    </motion.aside>
  )
}

function MetricCard({
  label,
  accent,
  children,
}: {
  label: string
  accent?: string
  children: React.ReactNode
}) {
  return (
    <div className="bg-white/5 rounded-xl p-3 flex flex-col gap-1">
      <span className="text-xs text-slate-500">{label}</span>
      <div
        className="text-lg font-bold text-white flex items-baseline flex-wrap"
        style={accent ? { color: accent } : undefined}
      >
        {children}
      </div>
    </div>
  )
}
