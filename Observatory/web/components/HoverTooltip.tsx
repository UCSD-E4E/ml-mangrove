'use client'

import { AnimatePresence, motion } from 'framer-motion'
import { CLASS_ICONS, CLASS_HEX, healthColor } from '@/lib/classConfig'
import type { HoverInfo } from '@/lib/layerBuilder'

interface HoverTooltipProps {
  info: HoverInfo | null
}

export default function HoverTooltip({ info }: HoverTooltipProps) {
  const props = info?.object?.properties

  return (
    <AnimatePresence>
      {info && props && (
        <motion.div
          key="tooltip"
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.94 }}
          transition={{ duration: 0.08 }}
          className="pointer-events-none fixed z-50"
          style={{ left: info.x + 14, top: info.y + 14 }}
        >
          <div className="bg-black/75 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl min-w-[200px]">
            {/* Class header */}
            <div
              className="flex items-center gap-2 text-base font-semibold mb-2"
              style={{ color: CLASS_HEX[props.class_name as keyof typeof CLASS_HEX] ?? '#fff' }}
            >
              <span>{CLASS_ICONS[props.class_name as keyof typeof CLASS_ICONS] ?? '📍'}</span>
              <span>{props.class_name}</span>
            </div>

            <div className="w-full h-px bg-white/10 mb-2" />

            {/* Health index */}
            <div className="flex items-center justify-between gap-3 mb-1.5">
              <span className="text-xs text-slate-400 w-14">Health</span>
              <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.max(0, Math.min(100, props.health_index * 100))}%`,
                    backgroundColor: healthColor(props.health_index),
                  }}
                />
              </div>
              <span className="text-xs font-mono text-white w-8 text-right">
                {props.health_index.toFixed(2)}
              </span>
            </div>

            {/* Area */}
            <div className="flex justify-between text-xs mt-1">
              <span className="text-slate-400">Area</span>
              <span className="text-white font-mono">
                {props.area_sqm >= 10000
                  ? `${(props.area_sqm / 10000).toFixed(1)} ha`
                  : `${Math.round(props.area_sqm).toLocaleString()} m²`}
              </span>
            </div>

            {/* Year */}
            <div className="flex justify-between text-xs mt-1">
              <span className="text-slate-400">Year</span>
              <span className="text-white font-mono">{props.year}</span>
            </div>

            {/* Tile */}
            <div className="flex justify-between text-xs mt-1">
              <span className="text-slate-400">Tile</span>
              <span className="text-slate-300 font-mono truncate max-w-[120px]">{props.tile_id}</span>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
