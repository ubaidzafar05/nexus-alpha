import { useMemo } from 'react'
import { cn } from '@/lib/cn'

interface Props {
  data: number[]
  width?: number
  height?: number
  stroke?: string
  fill?: string
  className?: string
}

export function Sparkline({
  data,
  width = 120,
  height = 32,
  stroke = '#4ADE80',
  fill = 'rgba(74, 222, 128, 0.14)',
  className,
}: Props) {
  const { d, area } = useMemo(() => {
    if (!data || data.length < 2) return { d: '', area: '' }
    const min = Math.min(...data)
    const max = Math.max(...data)
    const span = max - min || 1
    const step = width / (data.length - 1)
    const points = data.map((v, i) => {
      const x = i * step
      const y = height - ((v - min) / span) * height
      return [x, y] as const
    })
    const line = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`).join(' ')
    const areaPath = `${line} L${width} ${height} L0 ${height} Z`
    return { d: line, area: areaPath }
  }, [data, width, height])

  if (!d) return <div className={cn('h-8', className)} />

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={cn('overflow-visible', className)}
    >
      <path d={area} fill={fill} />
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.25} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
