import { create } from 'zustand'
import type { QueryClient } from '@tanstack/react-query'
import type { Microstructure, Portfolio, Regime } from './api'

interface LiveSlice {
  connected: boolean
  heartbeat: string | null
  cycle: number | null
  paused: boolean | null
  blindHalt: boolean | null
  cbLevel: number | null
  portfolio: Portfolio | null
  regime: Regime | null
  microstructure: Microstructure | null
  lastMessageAt: number | null
}

interface LiveActions {
  setConnected: (v: boolean) => void
  applyHeartbeat: (d: Partial<LiveSlice> & { portfolio?: Portfolio; regime?: Regime; microstructure?: Microstructure }) => void
  reset: () => void
}

const INITIAL: LiveSlice = {
  connected: false,
  heartbeat: null,
  cycle: null,
  paused: null,
  blindHalt: null,
  cbLevel: null,
  portfolio: null,
  regime: null,
  microstructure: null,
  lastMessageAt: null,
}

export const useLive = create<LiveSlice & LiveActions>((set) => ({
  ...INITIAL,
  setConnected: (v) => set({ connected: v }),
  applyHeartbeat: (d) =>
    set((s) => ({
      heartbeat: d.heartbeat ?? s.heartbeat,
      cycle: d.cycle ?? s.cycle,
      paused: d.paused ?? s.paused,
      blindHalt: d.blindHalt ?? s.blindHalt,
      cbLevel: d.cbLevel ?? s.cbLevel,
      portfolio: d.portfolio ?? s.portfolio,
      regime: d.regime ?? s.regime,
      microstructure: d.microstructure ?? s.microstructure,
      lastMessageAt: Date.now(),
    })),
  reset: () => set(INITIAL),
}))

interface WsMessage {
  type: 'HEARTBEAT' | 'NEW_TRADE' | 'REGISTRY_UPDATE'
  data?: Record<string, unknown>
  timestamp?: string
}

export function connectWebSocket(qc: QueryClient): () => void {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${window.location.host}/ws/live`
  let socket: WebSocket | null = null
  let retryTimer: number | null = null
  let stopped = false

  const open = () => {
    if (stopped) return
    socket = new WebSocket(url)

    socket.onopen = () => useLive.getState().setConnected(true)
    socket.onclose = () => {
      useLive.getState().setConnected(false)
      if (!stopped) retryTimer = window.setTimeout(open, 2500)
    }
    socket.onerror = () => socket?.close()

    socket.onmessage = (evt) => {
      let msg: WsMessage
      try {
        msg = JSON.parse(evt.data) as WsMessage
      } catch {
        return
      }
      if (msg.type === 'HEARTBEAT') {
        const d = (msg.data ?? {}) as Record<string, unknown>
        useLive.getState().applyHeartbeat({
          heartbeat: (d.heartbeat as string | null) ?? null,
          cycle: (d.cycle_counter as number | null) ?? null,
          paused: (d.paused as boolean | null) ?? null,
          blindHalt: (d.blind_halt as boolean | null) ?? null,
          cbLevel: (d.cb_level as number | null) ?? null,
          portfolio: (d.portfolio as Portfolio | null) ?? undefined,
          regime: (d.regime as Regime | null) ?? undefined,
          microstructure: (d.microstructure as Microstructure | null) ?? undefined,
        })
        // Keep slow-refresh queries in sync with the heartbeat tick
        qc.invalidateQueries({ queryKey: ['botStatus'] })
      }
      if (msg.type === 'NEW_TRADE') {
        qc.invalidateQueries({ queryKey: ['trades'] })
        qc.invalidateQueries({ queryKey: ['performance'] })
        qc.invalidateQueries({ queryKey: ['analytics'] })
        qc.invalidateQueries({ queryKey: ['analytics24h'] })
        qc.invalidateQueries({ queryKey: ['equityCurve'] })
      }
      if (msg.type === 'REGISTRY_UPDATE') {
        qc.invalidateQueries({ queryKey: ['registry'] })
      }
    }
  }

  open()

  return () => {
    stopped = true
    if (retryTimer) window.clearTimeout(retryTimer)
    socket?.close()
  }
}
