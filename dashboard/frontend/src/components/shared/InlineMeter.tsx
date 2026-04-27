import { cn } from '@/lib/cn'

interface Props {
  value: number
  max?: number
  className?: string
  color?: 'jade' | 'azure' | 'violetish' | 'amberish' | 'cinnabar'
}

export function InlineMeter({ value, max = 100, className, color = 'jade' }: Props) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  const fill =
    color === 'jade'
      ? 'bg-jade'
      : color === 'azure'
        ? 'bg-azure'
        : color === 'violetish'
          ? 'bg-violetish'
          : color === 'amberish'
            ? 'bg-amberish'
            : 'bg-cinnabar'
  return (
    <div className={cn('h-[2px] w-full overflow-hidden rounded-full bg-ink-700', className)}>
      <div
        className={cn('h-full transition-[width] duration-500 ease-out', fill)}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
