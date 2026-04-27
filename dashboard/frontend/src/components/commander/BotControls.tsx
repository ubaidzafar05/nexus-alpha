import { useState } from 'react'
import { Pause, Play, LogOut, AlertTriangle } from 'lucide-react'
import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'
import { LiveDot } from '@/components/shared/LiveDot'
import { useBotStatus, useBotControls } from '@/lib/queries'
import { useLive } from '@/lib/ws'
import { cn } from '@/lib/cn'

function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel,
  tone = 'cinnabar',
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  message: string
  confirmLabel: string
  tone?: 'cinnabar' | 'amberish'
  onConfirm: () => void
  onCancel: () => void
}) {
  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/70 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="w-[440px] rounded-xl border border-ink-700 bg-ink-900 shadow-panel"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-ink-700 px-5 py-3">
          <AlertTriangle className={cn('h-4 w-4', tone === 'cinnabar' ? 'text-cinnabar' : 'text-amberish')} />
          <span className="eyebrow text-pearl">{title}</span>
        </div>
        <div className="px-5 py-4 text-sm text-mercury">{message}</div>
        <div className="flex items-center justify-end gap-2 border-t border-ink-700 px-5 py-3">
          <button
            onClick={onCancel}
            className="rounded-md border border-ink-700 bg-ink-900 px-3 py-1.5 text-xs text-mercury hover:border-ink-600 hover:text-pearl"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={cn(
              'rounded-md px-3 py-1.5 text-xs font-medium',
              tone === 'cinnabar'
                ? 'bg-cinnabar/20 text-cinnabar hover:bg-cinnabar/30'
                : 'bg-amberish/20 text-amberish hover:bg-amberish/30',
            )}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export function BotControls() {
  const { data: bs } = useBotStatus()
  const livePaused = useLive((s) => s.paused)
  const liveBlindHalt = useLive((s) => s.blindHalt)
  const { pause, resume, marketExit } = useBotControls()

  const paused = livePaused ?? bs?.paused ?? false
  const blindHalt = liveBlindHalt ?? bs?.blind_halt ?? false
  const exitPending = bs?.market_exit_pending ?? false
  const cbLevel = bs?.cb_level ?? 0

  const [confirmPause, setConfirmPause] = useState(false)
  const [confirmExit, setConfirmExit] = useState(false)

  const state = blindHalt ? 'blind-halt' : paused ? 'paused' : 'autonomous'
  const stateLabel = blindHalt ? 'BLIND HALT' : paused ? 'PAUSED' : 'AUTONOMOUS'
  const stateColor: 'jade' | 'amberish' | 'cinnabar' = blindHalt ? 'cinnabar' : paused ? 'amberish' : 'jade'

  return (
    <>
      <Panel className="flex flex-col">
        <PanelHeader>
          <PanelTitle>Controls</PanelTitle>
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em]">
            <LiveDot color={stateColor} pulse={state === 'autonomous'} />
            <span
              className={cn(
                stateColor === 'jade' && 'text-jade',
                stateColor === 'amberish' && 'text-amberish',
                stateColor === 'cinnabar' && 'text-cinnabar',
              )}
            >
              {stateLabel}
            </span>
          </div>
        </PanelHeader>
        <PanelBody className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => (paused ? resume.mutate() : setConfirmPause(true))}
              disabled={pause.isPending || resume.isPending || blindHalt}
              className={cn(
                'flex items-center justify-center gap-2 rounded-md border px-3 py-2.5 text-xs font-medium transition-colors',
                paused
                  ? 'border-jade/30 bg-jade/10 text-jade hover:bg-jade/20'
                  : 'border-amberish/30 bg-amberish/10 text-amberish hover:bg-amberish/20',
                'disabled:cursor-not-allowed disabled:opacity-50',
              )}
            >
              {paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button
              onClick={() => setConfirmExit(true)}
              disabled={marketExit.isPending || exitPending}
              className={cn(
                'flex items-center justify-center gap-2 rounded-md border border-cinnabar/30 bg-cinnabar/10 px-3 py-2.5 text-xs font-medium text-cinnabar transition-colors hover:bg-cinnabar/20',
                'disabled:cursor-not-allowed disabled:opacity-50',
              )}
            >
              <LogOut className="h-3.5 w-3.5" />
              {exitPending ? 'Exit pending' : 'Market exit'}
            </button>
          </div>

          <div className="flex flex-col gap-1 rounded-md border border-ink-700 bg-ink-950/40 px-3 py-2.5">
            <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.18em]">
              <span className="text-mercury">Circuit breaker</span>
              <span
                className={cn(
                  'tabular-nums',
                  cbLevel === 0 && 'text-jade',
                  cbLevel === 1 && 'text-amberish',
                  cbLevel >= 2 && 'text-cinnabar',
                )}
              >
                L{cbLevel}
              </span>
            </div>
            <div className="text-[11px] text-mercury">
              {cbLevel === 0 && 'normal operation'}
              {cbLevel === 1 && 'drawdown guard engaged'}
              {cbLevel >= 2 && 'forced flatten'}
            </div>
          </div>
        </PanelBody>
      </Panel>

      <ConfirmDialog
        open={confirmPause}
        title="Pause trading loop"
        message="The bot will stop opening new positions. Existing positions remain open."
        confirmLabel="Pause loop"
        tone="amberish"
        onCancel={() => setConfirmPause(false)}
        onConfirm={() => {
          setConfirmPause(false)
          pause.mutate()
        }}
      />
      <ConfirmDialog
        open={confirmExit}
        title="Flatten all positions"
        message="This will close every open position at market and halt the loop. Not reversible once executed."
        confirmLabel="Flatten now"
        tone="cinnabar"
        onCancel={() => setConfirmExit(false)}
        onConfirm={() => {
          setConfirmExit(false)
          marketExit.mutate()
        }}
      />
    </>
  )
}
