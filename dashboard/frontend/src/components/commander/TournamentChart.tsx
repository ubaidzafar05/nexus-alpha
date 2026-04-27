import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'
import { Sparkline } from '@/components/shared/Sparkline'
import { useRegistry } from '@/lib/queries'
import { fmtPct } from '@/lib/fmt'

export function TournamentChart() {
  const { data: reg } = useRegistry()
  const champId = reg?.champion?.id ?? '—'
  const curve = reg?.champion?.metrics?.equity_curve ?? []
  const first = curve.length > 0 ? curve[0] : 0
  const last = curve.length > 0 ? curve[curve.length - 1] : 0
  const drift = first > 0 ? ((last - first) / first) * 100 : 0
  const candidateN = reg?.candidates?.length ?? 0
  const pastN = reg?.past_champions?.length ?? 0

  return (
    <Panel className="flex flex-col">
      <PanelHeader>
        <PanelTitle>Tournament</PanelTitle>
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mercury">
          {candidateN} candidates · {pastN} past
        </span>
      </PanelHeader>
      <PanelBody className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <div>
            <div className="eyebrow mb-1 text-mercury">Champion</div>
            <div className="truncate font-mono text-sm text-pearl">{champId}</div>
          </div>
          <div className="text-right">
            <div className="eyebrow mb-1 text-mercury">Drift</div>
            <div
              className={
                'font-serif text-[22px] leading-none tracking-tight tabular-nums ' +
                (drift >= 0 ? 'text-jade' : 'text-cinnabar')
              }
              data-numeric
            >
              {fmtPct(drift, 1)}
            </div>
          </div>
        </div>
        <Sparkline
          data={curve}
          width={320}
          height={54}
          stroke={drift >= 0 ? '#4ADE80' : '#F87171'}
          fill={drift >= 0 ? 'rgba(74, 222, 128, 0.14)' : 'rgba(248, 113, 113, 0.14)'}
          className="w-full"
        />
      </PanelBody>
    </Panel>
  )
}
