import { useMemo } from 'react'
import { ArrowDownRight, ArrowUpRight } from 'lucide-react'
import { Panel } from '@/components/shared/Panel'
import { Sparkline } from '@/components/shared/Sparkline'
import { InlineMeter } from '@/components/shared/InlineMeter'
import { usePortfolio, useEquityCurve, useAnalytics24h } from '@/lib/queries'
import { useLive } from '@/lib/ws'
import { fmtMoney, fmtPct } from '@/lib/fmt'
import { cn } from '@/lib/cn'

export function HeroCard() {
  const livePf = useLive((s) => s.portfolio)
  const { data: pfQ } = usePortfolio()
  const { data: equity } = useEquityCurve(240)
  const { data: a24 } = useAnalytics24h()

  const pf = livePf ?? pfQ
  const nav = pf?.nav ?? 0
  const cash = pf?.cash ?? 0
  const realized = pf?.realized_pnl ?? 0
  const leverage = pf?.leverage ?? 0
  const pnl24 = a24?.pnl_24h_pct ?? 0

  const curve = equity?.equity_curve ?? []
  const first = curve.length > 0 ? curve[0] : 0
  const last = curve.length > 0 ? curve[curve.length - 1] : 0
  const drift = first > 0 ? ((last - first) / first) * 100 : 0

  const navText = useMemo(() => fmtMoney(nav, { compact: false }), [nav])
  const up = pnl24 >= 0

  return (
    <Panel className="overflow-hidden">
      <div className="flex items-start justify-between gap-8 px-6 pt-6">
        <div className="min-w-0">
          <div className="eyebrow mb-2 text-mercury">Portfolio · NAV</div>
          <div className="flex items-baseline gap-3">
            <span
              className="font-serif text-[58px] leading-[0.95] tracking-tight text-pearl tabular-nums"
              data-numeric
            >
              {navText}
            </span>
            <span
              className={cn(
                'flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-xs tabular-nums',
                up
                  ? 'border-jade/30 bg-jade/10 text-jade'
                  : 'border-cinnabar/30 bg-cinnabar/10 text-cinnabar',
              )}
              title="24h PnL"
            >
              {up ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
              {fmtPct(pnl24, 2)}
            </span>
          </div>
          <div className="mt-1 flex items-center gap-3 text-[11px] font-mono uppercase tracking-[0.18em] text-mercury">
            <span>cash {fmtMoney(cash, { compact: true })}</span>
            <span className="text-ink-600">·</span>
            <span>realized {fmtMoney(realized, { compact: true })}</span>
            <span className="text-ink-600">·</span>
            <span>lev {leverage.toFixed(2)}x</span>
          </div>
        </div>
        <div className="shrink-0">
          <Sparkline
            data={curve}
            width={220}
            height={56}
            stroke={drift >= 0 ? '#4ADE80' : '#F87171'}
            fill={drift >= 0 ? 'rgba(74, 222, 128, 0.14)' : 'rgba(248, 113, 113, 0.14)'}
          />
          <div className="mt-1 text-right font-mono text-[10px] uppercase tracking-[0.2em] text-mercury">
            {curve.length > 0 ? `${curve.length} pts · drift ${fmtPct(drift, 2)}` : 'no curve'}
          </div>
        </div>
      </div>

      <div className="mt-5 border-t border-ink-700 bg-ink-950/40 px-6 py-4">
        <div className="flex items-center justify-between">
          <span className="eyebrow text-mercury">Cash / NAV</span>
          <span className="font-mono text-xs tabular-nums text-mercury">
            {nav > 0 ? ((cash / nav) * 100).toFixed(0) : '0'}%
          </span>
        </div>
        <InlineMeter
          value={nav > 0 ? (cash / nav) * 100 : 0}
          max={100}
          color="azure"
          className="mt-2"
        />
      </div>
    </Panel>
  )
}
