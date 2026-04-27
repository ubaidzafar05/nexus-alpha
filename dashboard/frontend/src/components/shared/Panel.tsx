import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface Props extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  as?: 'div' | 'section'
}

export function Panel({ children, className, ...rest }: Props) {
  return (
    <div
      className={cn(
        'relative rounded-lg border border-ink-700 bg-ink-900/80 shadow-panel backdrop-blur-[1px]',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}

export function PanelHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-center justify-between border-b border-ink-700 px-5 py-3', className)}>
      {children}
    </div>
  )
}

export function PanelTitle({ children, className }: { children: ReactNode; className?: string }) {
  return <h3 className={cn('eyebrow text-pearl/90', className)}>{children}</h3>
}

export function PanelBody({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('p-5', className)}>{children}</div>
}
