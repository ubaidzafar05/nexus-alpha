import { Panel } from '@/components/shared/Panel'
import { InlineMeter } from '@/components/shared/InlineMeter'
import { useAnalytics, useAnalytics24h, usePerformance } from '@/lib/queries'
import { fmtPct } from '@/lib/fmt'
import { cn } from '@/lib/cn'

type MeterColor = 'jade' | 'azure' | 'violetish' | 'amberish' | 'cinnabar'

function Kpi({
  label,
  value,
  hint,
  meter,
  meterColor = 'jade',
  valueClass,
}: {
  label: string
  value: string
  hint?: string
  meter?: number
  meterColor?: MeterColor
  valueClass?: string
}) {
  return (
    <div className="flex min-w-0 flex-col justify-between gap-2 border-r border-ink-700 px-5 py-4 last:border-r-0">
      <div className="eyebrow text-mercury">{label}</div>
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={cn(
            'font-serif text-[28px] leading-none tracking-tight text-pearl tabular-nums',
            valueClass,
          )}
          data-numeric
        >
          {value}
        </span>
        {hint && <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mercury">{hint}</span>}
      </div>
      {meter !== undefined && <InlineMeter value={meter} color={meterColor} />}
    </div>
  )
}

export function KpiGrid() {
  const { data: a } = useAnalytics()
  const { data: a24 } = useAnalytics24h()
  const { data: perf } = usePerformance()

  const winRate = (a?.win_rate ?? 0) * 100
  const pf = a?.profit_factor ?? 0
  const mdd = (a?.max_drawdown ?? 0) * 100
  const trades24 = a24?.trades_24h ?? 0
  const totalTrades = a?.total_trades ?? perf?.total_trades ?? 0

  const mddColor: MeterColor = mdd <= -5 ? 'cinnabar' : mdd <= -2 ? 'amberish' : 'jade'
  const pfColor: MeterColor = pf >= 1.5 ? 'jade' : pf >= 1 ? 'azure' : 'cinnabar'
  const winColor: MeterColor = winRate >= 55 ? 'jade' : winRate >= 45 ? 'azure' : 'cinnabar'
  const mddMeter = Math.min(100, Math.abs(mdd) * 5)
  const pfMeter = Math.min(100, pf * 40)

  return (
    <Panel>
      <div className="grid grid-cols-2 md:grid-cols-4">
        <Kpi
          label="Win rate"
          value={`${winRate.toFixed(0)}%`}
          hint={`${totalTrades} tr`}
          meter={winRate}
          meterColor={winColor}
        />
        <Kpi
          label="Profit factor"
          value={pf.toFixed(2)}
          meter={pfMeter}
          meterColor={pfColor}
        />
        <Kpi
          label="Max drawdown"
          value={fmtPct(mdd, 1)}
          meter={mddMeter}
          meterColor={mddColor}
          valueClass={mdd <= -5 ? 'text-cinnabar' : undefined}
        />
        <Kpi
          label="Trades · 24h"
          value={trades24.toString()}
          hint={`${a24?.wins_24h ?? 0}w · ${a24?.losses_24h ?? 0}l`}
          meter={Math.min(100, trades24 * 2)}
          meterColor="violetish"
        />
      </div>
    </Panel>
  )
}
