import { cn } from '@/lib/cn'

interface Props {
  color?: 'jade' | 'amberish' | 'cinnabar' | 'mercury'
  pulse?: boolean
  className?: string
}

export function LiveDot({ color = 'jade', pulse = true, className }: Props) {
  const bg =
    color === 'jade'
      ? 'bg-jade'
      : color === 'amberish'
        ? 'bg-amberish'
        : color === 'cinnabar'
          ? 'bg-cinnabar'
          : 'bg-mercury'
  return (
    <span className={cn('relative inline-flex h-1.5 w-1.5', className)}>
      {pulse && (
        <span className={cn('absolute inset-0 animate-ping rounded-full opacity-60', bg)} />
      )}
      <span className={cn('relative inline-flex h-1.5 w-1.5 rounded-full', bg)} />
    </span>
  )
}
