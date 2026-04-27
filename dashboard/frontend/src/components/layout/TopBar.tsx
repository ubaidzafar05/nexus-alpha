import { useEffect, useState } from 'react'
import { Command, Search } from 'lucide-react'
import { cn } from '@/lib/cn'
import { LiveDot } from '@/components/shared/LiveDot'
import { useHeartbeat } from '@/hooks/useHeartbeat'
import { fmtAge, fmtClock } from '@/lib/fmt'
import { useLive } from '@/lib/ws'

interface Props {
  onOpenPalette: () => void
}

export function TopBar({ onOpenPalette }: Props) {
  const { freshness, label, color, ageSeconds, paused, blindHalt, cycle } = useHeartbeat()
  const wsConnected = useLive((s) => s.connected)
  const [clock, setClock] = useState(fmtClock())
  useEffect(() => {
    const id = window.setInterval(() => setClock(fmtClock()), 1000)
    return () => window.clearInterval(id)
  }, [])

  const modeLabel = blindHalt ? 'BLIND HALT' : paused ? 'PAUSED' : 'AUTONOMOUS'
  const modeColor: 'cinnabar' | 'amberish' | 'jade' = blindHalt
    ? 'cinnabar'
    : paused
      ? 'amberish'
      : 'jade'

  return (
    <header className="relative flex h-14 items-center justify-between border-b border-ink-700 bg-ink-950/80 px-5 backdrop-blur">
      {/* Heartbeat accent line */}
      {freshness === 'live' && (
        <div className="heartbeat-bar animate-pulse-bar absolute left-0 right-0 top-0 h-[1px] origin-left" />
      )}

      <div className="flex items-baseline gap-3">
        <span className="font-serif text-xl tracking-tight text-pearl">
          Nexus<span className="text-mercury"> · Terminal</span>
        </span>
        <span className="hidden items-center gap-1.5 text-[11px] font-mono text-mercury md:inline-flex">
          <LiveDot color={modeColor} pulse={!paused && !blindHalt} />
          <span className={cn('uppercase tracking-[0.18em]')}>{modeLabel}</span>
        </span>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onOpenPalette}
          className="group flex items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-2.5 py-1.5 text-xs text-mercury transition-colors hover:border-ink-600 hover:text-pearl"
        >
          <Search className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Search · Commands</span>
          <kbd className="ml-1 hidden items-center gap-1 rounded border border-ink-700 bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] tracking-widest text-mercury md:inline-flex">
            <Command className="h-2.5 w-2.5" /> K
          </kbd>
        </button>

        <div className="flex items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-2.5 py-1 text-[11px] font-mono uppercase tracking-[0.18em]">
          <LiveDot color={color} pulse={freshness === 'live'} />
          <span className={cn(color === 'jade' && 'text-jade', color === 'amberish' && 'text-amberish', color === 'cinnabar' && 'text-cinnabar', color === 'mercury' && 'text-mercury')}>
            {label}
          </span>
          <span className="text-mercury">·</span>
          <span className="text-mercury">{fmtAge(ageSeconds)}</span>
          {cycle != null && (
            <>
              <span className="text-mercury">·</span>
              <span className="text-mercury">cy {cycle.toLocaleString()}</span>
            </>
          )}
          <span className="text-mercury">·</span>
          <span className={cn(wsConnected ? 'text-jade' : 'text-mercury')}>ws</span>
        </div>

        <div className="hidden font-mono text-[11px] tracking-[0.12em] text-mercury lg:block">{clock}</div>
      </div>
    </header>
  )
}
