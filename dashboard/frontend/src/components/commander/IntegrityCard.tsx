import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'
import { InlineMeter } from '@/components/shared/InlineMeter'
import { useHeartbeat } from '@/hooks/useHeartbeat'
import { useBotStatus } from '@/lib/queries'
import { fmtAge } from '@/lib/fmt'
import { cn } from '@/lib/cn'

function Row({
  label,
  value,
  meter,
  meterColor = 'jade',
}: {
  label: string
  value: React.ReactNode
  meter?: number
  meterColor?: 'jade' | 'azure' | 'amberish' | 'cinnabar' | 'violetish'
}) {
  return (
    <div className="flex flex-col gap-1.5 py-2.5 first:pt-0 last:pb-0">
      <div className="flex items-center justify-between">
        <span className="eyebrow text-mercury">{label}</span>
        <span className="font-mono text-xs tabular-nums text-pearl">{value}</span>
      </div>
      {meter !== undefined && <InlineMeter value={meter} color={meterColor} />}
    </div>
  )
}

export function IntegrityCard() {
  const { freshness, ageSeconds, label, color } = useHeartbeat()
  const { data: bs } = useBotStatus()
  const cb = bs?.cb_level ?? 0

  const freshnessScore =
    freshness === 'live' ? 100 : freshness === 'stale' ? 55 : freshness === 'offline' ? 10 : 0
  const cbScore = cb === 0 ? 100 : cb === 1 ? 55 : 10

  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>Integrity</PanelTitle>
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mercury">
          NEXUS-ULTRA · Ω
        </span>
      </PanelHeader>
      <PanelBody className="divide-y divide-ink-700">
        <Row
          label="Heartbeat"
          value={<span className={cn(color === 'jade' && 'text-jade', color === 'amberish' && 'text-amberish', color === 'cinnabar' && 'text-cinnabar')}>{label} · {fmtAge(ageSeconds)}</span>}
          meter={freshnessScore}
          meterColor={color === 'jade' ? 'jade' : color === 'amberish' ? 'amberish' : 'cinnabar'}
        />
        <Row
          label="Circuit breaker"
          value={`L${cb}`}
          meter={cbScore}
          meterColor={cb === 0 ? 'jade' : cb === 1 ? 'amberish' : 'cinnabar'}
        />
        <Row
          label="Cycle"
          value={(bs?.cycle_counter ?? 0).toLocaleString()}
        />
        <Row
          label="Cycle interval"
          value={bs?.cycle_interval_s != null ? `${bs.cycle_interval_s.toFixed(1)}s` : '—'}
        />
      </PanelBody>
    </Panel>
  )
}
