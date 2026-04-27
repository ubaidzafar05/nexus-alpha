import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { RailNav } from './RailNav'
import { TopBar } from './TopBar'
import { LiveStrip } from './LiveStrip'
import { NoiseLayer } from '@/components/shared/NoiseLayer'

export function AppShell() {
  const [paletteOpen, setPaletteOpen] = useState(false)

  return (
    <div className="relative flex min-h-screen w-full overflow-hidden bg-ink-950 text-pearl">
      <NoiseLayer />
      <RailNav />
      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <TopBar onOpenPalette={() => setPaletteOpen(true)} />
        <LiveStrip />
        <main className="relative flex-1 overflow-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={typeof window !== 'undefined' ? window.location.pathname : 'route'}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
              className="h-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      {/* Palette mount point — wired in Phase 9 */}
      {paletteOpen && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-ink-950/60 backdrop-blur-sm"
          onClick={() => setPaletteOpen(false)}
        >
          <div
            className="mt-24 w-[640px] rounded-xl border border-ink-700 bg-ink-900 p-4 text-sm text-mercury shadow-panel"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="eyebrow mb-2">Command palette</div>
            <div>Coming online in phase 9 — ⌘K</div>
          </div>
        </div>
      )}
    </div>
  )
}
