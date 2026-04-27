import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'
import { InlineMeter } from '@/components/shared/InlineMeter'
import { usePerformance, usePortfolio, useAnalytics } from '@/lib/queries'
import { useLive } from '@/lib/ws'
import { fmtPct } from '@/lib/fmt'

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

export function EfficiencyCard() {
  const { data: perf } = usePerformance()
  const { data: a } = useAnalytics()
  const { data: pfQ } = usePortfolio()
  const livePf = useLive((s) => s.portfolio)
  const pf = livePf ?? pfQ

  const totalPct = perf?.total_pnl_pct ?? 0
  const wins = perf?.wins ?? 0
  const losses = perf?.losses ?? 0
  const be = perf?.break_even ?? 0
  const totalTrades = wins + losses + be
  const hitRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 0
  const avgTrade = a?.avg_trade_pnl ?? 0
  const leverage = pf?.leverage ?? 0

  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>Efficiency</PanelTitle>
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mercury">
          NEXUS-ULTRA · Σ
        </span>
      </PanelHeader>
      <PanelBody className="divide-y divide-ink-700">
        <Row
          label="Total PnL"
          value={<span className={totalPct >= 0 ? 'text-jade' : 'text-cinnabar'}>{fmtPct(totalPct, 2)}</span>}
          meter={Math.min(100, Math.abs(totalPct) * 2)}
          meterColor={totalPct >= 0 ? 'jade' : 'cinnabar'}
        />
        <Row
          label="Hit rate"
          value={`${hitRate.toFixed(0)}% · ${wins}w`}
          meter={hitRate}
          meterColor={hitRate >= 55 ? 'jade' : hitRate >= 45 ? 'azure' : 'cinnabar'}
        />
        <Row
          label="Avg trade"
          value={<span className={avgTrade >= 0 ? 'text-jade' : 'text-cinnabar'}>{fmtPct(avgTrade, 2)}</span>}
        />
        <Row
          label="Leverage"
          value={`${leverage.toFixed(2)}×`}
          meter={Math.min(100, leverage * 25)}
          meterColor={leverage > 3 ? 'amberish' : 'azure'}
        />
      </PanelBody>
    </Panel>
  )
}
