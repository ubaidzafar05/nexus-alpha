import { useEffect, useState } from 'react'
import { useLive } from '@/lib/ws'
import { useBotStatus } from '@/lib/queries'

export type Freshness = 'live' | 'stale' | 'offline' | 'unknown'

export interface HeartbeatView {
  ageSeconds: number | null
  freshness: Freshness
  label: string
  color: 'jade' | 'amberish' | 'cinnabar' | 'mercury'
  cycle: number | null
  paused: boolean
  blindHalt: boolean
  cbLevel: number
}

export function useHeartbeat(): HeartbeatView {
  const { data } = useBotStatus()
  const live = useLive()
  const [tick, setTick] = useState(0)

  // 1s ticker so the age display counts up between updates
  useEffect(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 1000)
    return () => window.clearInterval(id)
  }, [])

  const heartbeat = live.heartbeat ?? data?.heartbeat ?? null
  const cycle = live.cycle ?? data?.cycle_counter ?? null
  const paused = (live.paused ?? data?.paused ?? false) as boolean
  const blindHalt = (live.blindHalt ?? data?.blind_halt ?? false) as boolean
  const cbLevel = (live.cbLevel ?? data?.cb_level ?? 0) as number

  let ageSeconds: number | null = null
  if (heartbeat) {
    const ts = new Date(heartbeat.endsWith('Z') ? heartbeat : heartbeat + 'Z').getTime()
    if (Number.isFinite(ts)) ageSeconds = Math.max(0, (Date.now() - ts) / 1000)
  } else if (data?.heartbeat_age_s != null) {
    ageSeconds = data.heartbeat_age_s + tick // approximate interpolation
  }

  let freshness: Freshness = 'unknown'
  let color: HeartbeatView['color'] = 'mercury'
  let label = 'NO SIGNAL'

  if (ageSeconds == null) {
    freshness = 'unknown'
  } else if (ageSeconds < 30) {
    freshness = 'live'
    color = 'jade'
    label = 'LIVE'
  } else if (ageSeconds < 180) {
    freshness = 'stale'
    color = 'amberish'
    label = 'STALE'
  } else {
    freshness = 'offline'
    color = 'cinnabar'
    label = 'OFFLINE'
  }

  return { ageSeconds, freshness, label, color, cycle, paused, blindHalt, cbLevel }
}
