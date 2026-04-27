export const API_BASE = '/api'

export interface BotStatus {
  heartbeat: string | null
  heartbeat_age_s: number | null
  fresh: boolean
  cycle_counter: number
  cycle_interval_s: number | null
  paused: boolean
  blind_halt: boolean
  cb_level: number
  market_exit_pending: boolean
}

export interface Position {
  symbol: string
  qty?: number
  side?: string
  avg_entry_price?: number
  unrealized_pnl?: number
  [key: string]: unknown
}

export interface Portfolio {
  nav: number
  cash: number
  realized_pnl: number
  position_count: number
  leverage: number
  positions: Position[]
}

export interface Regime {
  name: string
  changepoint_probability: number
}

export interface Signal {
  timestamp?: string
  symbol: string
  direction: string
  confidence: number
  regime?: string
  [key: string]: unknown
}

export interface Microstructure {
  vpin_max: number
  ofi_max: number
}

export interface Analytics {
  win_rate: number
  profit_factor: number
  max_drawdown: number
  total_trades: number
  avg_trade_pnl?: number
}

export interface Analytics24h {
  trades_24h: number
  pnl_24h_pct: number
  wins_24h: number
  losses_24h: number
}

export interface Performance {
  wins: number
  losses: number
  break_even: number
  total_pnl_pct: number
  total_trades?: number
}

export interface Trade {
  trade_id: string
  timestamp: string
  symbol: string
  side: string
  entry_price: number
  exit_price: number | null
  realized_pnl_pct: number
  status: string
}

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Registry {
  champion: { id: string; metrics?: { equity_curve?: number[] } } | null
  candidates: unknown[]
  past_champions: unknown[]
}

export interface EquityCurve {
  equity_curve: number[]
  source: string
  n?: number
}

export interface TelemetryRow {
  metric_name: string
  metric_value: number
  timestamp: string
}

export interface Agent {
  agent_id: string
  ancestor_id: string
  lineage_depth: number
  is_hedge: boolean
  capital_weight: number
  cluster_id: string | number
  metrics: { sharpe?: number; pnl_pct?: number }
}

export interface Genealogy {
  swarm: Agent[]
  error?: string
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return (await r.json()) as T
}

async function post<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { method: 'POST' })
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return (await r.json()) as T
}

export const api = {
  botStatus: () => get<BotStatus>('/bot/status'),
  portfolio: () => get<Portfolio>('/portfolio'),
  regime: () => get<Regime>('/regime'),
  signals: (limit = 30) => get<Signal[]>(`/signals/recent?limit=${limit}`),
  microstructure: () => get<Microstructure>('/microstructure'),
  analytics: () => get<Analytics>('/analytics'),
  analytics24h: () => get<Analytics24h>('/analytics/24h'),
  performance: () => get<Performance>('/internal/performance'),
  trades: (symbol?: string, limit = 100) =>
    get<Trade[]>(symbol ? `/trades?symbol=${encodeURIComponent(symbol)}&limit=${limit}` : `/trades?limit=${limit}`),
  candles: (symbol: string, timeframe = '1h') =>
    get<Candle[] | { error: string }>(`/candles/${encodeURIComponent(symbol.replace('/', '_'))}?timeframe=${timeframe}`),
  registry: () => get<Registry>('/registry'),
  equityCurve: (limit = 500) => get<EquityCurve>(`/equity-curve?limit=${limit}`),
  telemetry: () => get<TelemetryRow[]>('/v4/telemetry'),
  genealogy: () => get<Genealogy>('/swarm/genealogy'),
  controlStatus: () => get<{ paused: boolean; market_exit_pending: boolean }>('/control/status'),
  pause: () => post<unknown>('/control/pause'),
  resume: () => post<unknown>('/control/resume'),
  marketExit: () => post<unknown>('/control/market-exit'),
}
