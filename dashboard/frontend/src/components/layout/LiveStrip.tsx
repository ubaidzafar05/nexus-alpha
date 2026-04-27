import { useLive } from '@/lib/ws'
import { usePortfolio, useRegime, useAnalytics24h, useBotStatus } from '@/lib/queries'
import { fmtMoney, fmtPct } from '@/lib/fmt'
import { NumberMorph } from '@/components/shared/NumberMorph'
import { cn } from '@/lib/cn'

function Item({
  eyebrow,
  children,
  className,
}: {
  eyebrow: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex min-w-0 flex-col gap-1 border-r border-ink-700 px-5 py-3 last:border-r-0', className)}>
      <span className="eyebrow">{eyebrow}</span>
      <div className="truncate text-sm font-medium text-pearl">{children}</div>
    </div>
  )
}

export function LiveStrip() {
  const livePf = useLive((s) => s.portfolio)
  const liveRegime = useLive((s) => s.regime)
  const liveCycle = useLive((s) => s.cycle)
  const { data: pfQ } = usePortfolio()
  const { data: regimeQ } = useRegime()
  const { data: a24 } = useAnalytics24h()
  const { data: bs } = useBotStatus()

  const pf = livePf ?? pfQ
  const regime = liveRegime ?? regimeQ
  const cycle = liveCycle ?? bs?.cycle_counter ?? null

  const nav = pf?.nav ?? 0
  const realized = pf?.realized_pnl ?? 0
  const positions = pf?.position_count ?? 0
  const pnl24 = a24?.pnl_24h_pct ?? 0
  const trades24 = a24?.trades_24h ?? 0
  const regimeName = (regime?.name ?? 'unknown').toUpperCase()
  const cp = (regime?.changepoint_probability ?? 0) * 100
  const pnlColor = realized >= 0 ? 'text-jade' : 'text-cinnabar'
  const pnl24Color = pnl24 >= 0 ? 'text-jade' : 'text-cinnabar'

  return (
    <div className="grid grid-cols-2 items-stretch border-b border-ink-700 bg-ink-900/60 backdrop-blur sm:grid-cols-3 md:grid-cols-6">
      <Item eyebrow="Portfolio NAV">
        <NumberMorph value={nav} format={(v) => fmtMoney(v, { compact: true })} className="text-pearl" />
      </Item>
      <Item eyebrow="Realized PnL">
        <span className={pnlColor}>
          <NumberMorph value={realized} format={(v) => fmtMoney(v, { compact: true })} />
        </span>
      </Item>
      <Item eyebrow="24h PnL">
        <span className={pnl24Color}>
          <NumberMorph value={pnl24} format={(v) => fmtPct(v, 2)} />
          <span className="ml-2 text-mercury">· {trades24} tr</span>
        </span>
      </Item>
      <Item eyebrow="Positions">
        <span className="text-pearl">{positions}</span>
      </Item>
      <Item eyebrow="Regime" className="md:col-span-1">
        <span className="text-pearl">{regimeName}</span>
        <span className="ml-2 text-mercury">cp {cp.toFixed(0)}%</span>
      </Item>
      <Item eyebrow="Cycle">
        <span className="text-pearl">{cycle?.toLocaleString() ?? '—'}</span>
      </Item>
    </div>
  )
}
