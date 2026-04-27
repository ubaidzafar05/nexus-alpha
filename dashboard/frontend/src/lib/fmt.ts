export function fmtMoney(v: number | null | undefined, opts: { compact?: boolean } = {}): string {
  const n = Number(v ?? 0)
  const sign = n < 0 ? '-' : ''
  const abs = Math.abs(n)
  if (opts.compact) {
    if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`
    if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`
    if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(2)}K`
  }
  return `${sign}$${abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

export function fmtPct(v: number | null | undefined, decimals = 2): string {
  const n = Number(v ?? 0)
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(decimals)}%`
}

export function fmtInt(v: number | null | undefined): string {
  const n = Math.round(Number(v ?? 0))
  return n.toLocaleString('en-US')
}

export function fmtAge(seconds: number | null | undefined): string {
  const s = Number(seconds ?? 0)
  if (!Number.isFinite(s) || s < 0) return '--'
  if (s < 1) return 'now'
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  return `${Math.round(s / 3600)}h`
}

export function fmtTime(ts: string | number | Date | undefined): string {
  if (!ts) return '--'
  const d = ts instanceof Date ? ts : new Date(ts)
  if (Number.isNaN(d.getTime())) return '--'
  return d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
}

export function fmtClock(d: Date = new Date()): string {
  return d.toISOString().replace('T', ' ').slice(0, 19)
}
