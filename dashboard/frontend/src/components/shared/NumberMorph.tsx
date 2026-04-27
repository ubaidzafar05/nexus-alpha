import { useEffect, useRef } from 'react'
import { animate, useMotionValue, useMotionValueEvent } from 'framer-motion'
import { cn } from '@/lib/cn'

interface Props {
  value: number
  format?: (n: number) => string
  className?: string
  duration?: number
}

export function NumberMorph({ value, format, className, duration = 0.6 }: Props) {
  const ref = useRef<HTMLSpanElement>(null)
  const mv = useMotionValue(value)

  useEffect(() => {
    const controls = animate(mv, value, { duration, ease: [0.16, 1, 0.3, 1] })
    return controls.stop
  }, [value, mv, duration])

  useMotionValueEvent(mv, 'change', (v) => {
    if (ref.current) ref.current.textContent = format ? format(v) : v.toFixed(2)
  })

  return (
    <span
      ref={ref}
      data-numeric
      className={cn('tabular-nums', className)}
    >
      {format ? format(value) : value.toFixed(2)}
    </span>
  )
}
