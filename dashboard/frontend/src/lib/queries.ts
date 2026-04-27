import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './api'

const FAST = 3_000
const MEDIUM = 5_000
const SLOW = 15_000

export const useBotStatus = () =>
  useQuery({ queryKey: ['botStatus'], queryFn: api.botStatus, refetchInterval: FAST })

export const usePortfolio = () =>
  useQuery({ queryKey: ['portfolio'], queryFn: api.portfolio, refetchInterval: FAST })

export const useRegime = () =>
  useQuery({ queryKey: ['regime'], queryFn: api.regime, refetchInterval: FAST })

export const useSignals = (limit = 30) =>
  useQuery({ queryKey: ['signals', limit], queryFn: () => api.signals(limit), refetchInterval: FAST })

export const useMicrostructure = () =>
  useQuery({ queryKey: ['microstructure'], queryFn: api.microstructure, refetchInterval: MEDIUM })

export const useAnalytics = () =>
  useQuery({ queryKey: ['analytics'], queryFn: api.analytics, refetchInterval: MEDIUM })

export const useAnalytics24h = () =>
  useQuery({ queryKey: ['analytics24h'], queryFn: api.analytics24h, refetchInterval: MEDIUM })

export const usePerformance = () =>
  useQuery({ queryKey: ['performance'], queryFn: api.performance, refetchInterval: MEDIUM })

export const useTrades = (symbol?: string, limit = 100) =>
  useQuery({
    queryKey: ['trades', symbol ?? null, limit],
    queryFn: () => api.trades(symbol, limit),
    refetchInterval: MEDIUM,
  })

export const useCandles = (symbol: string, timeframe = '1h') =>
  useQuery({
    queryKey: ['candles', symbol, timeframe],
    queryFn: () => api.candles(symbol, timeframe),
    staleTime: 60_000,
    refetchInterval: 60_000,
  })

export const useRegistry = () =>
  useQuery({ queryKey: ['registry'], queryFn: api.registry, refetchInterval: SLOW })

export const useEquityCurve = (limit = 500) =>
  useQuery({ queryKey: ['equityCurve', limit], queryFn: () => api.equityCurve(limit), refetchInterval: SLOW })

export const useTelemetry = () =>
  useQuery({ queryKey: ['telemetry'], queryFn: api.telemetry, refetchInterval: MEDIUM })

export const useGenealogy = () =>
  useQuery({ queryKey: ['genealogy'], queryFn: api.genealogy, refetchInterval: SLOW })

export function useBotControls() {
  const qc = useQueryClient()
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['botStatus'] })
    qc.invalidateQueries({ queryKey: ['portfolio'] })
  }
  return {
    pause: useMutation({ mutationFn: api.pause, onSuccess: invalidate }),
    resume: useMutation({ mutationFn: api.resume, onSuccess: invalidate }),
    marketExit: useMutation({ mutationFn: api.marketExit, onSuccess: invalidate }),
  }
}
