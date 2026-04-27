import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  LineChart,
  Notebook,
  Network,
  Sigma,
  Crosshair,
} from 'lucide-react'
import { cn } from '@/lib/cn'

const ITEMS = [
  { to: '/', label: 'Commander', icon: LayoutDashboard, end: true },
  { to: '/terminal', label: 'Terminal', icon: LineChart },
  { to: '/journal', label: 'Journal', icon: Notebook },
  { to: '/genealogy', label: 'Genealogy', icon: Network },
  { to: '/analytics', label: 'Analytics', icon: Sigma },
]

export function RailNav() {
  return (
    <nav className="flex h-full w-16 shrink-0 flex-col items-center border-r border-ink-700 bg-ink-950/60 backdrop-blur">
      <div className="flex h-14 w-full items-center justify-center border-b border-ink-700">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-jade/15 text-jade">
          <Crosshair className="h-4 w-4" strokeWidth={2.2} />
        </div>
      </div>
      <ul className="mt-3 flex flex-col gap-1">
        {ITEMS.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              title={label}
              className={({ isActive }) =>
                cn(
                  'group relative flex h-11 w-11 items-center justify-center rounded-md text-mercury transition-colors',
                  'hover:bg-ink-800 hover:text-pearl',
                  isActive && 'bg-ink-800 text-pearl',
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span className="absolute left-0 top-1/2 h-6 w-[2px] -translate-y-1/2 rounded-r-full bg-jade" />
                  )}
                  <Icon className="h-[18px] w-[18px]" strokeWidth={1.6} />
                </>
              )}
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="mt-auto mb-3 px-1 text-center text-[9px] font-mono uppercase tracking-[0.2em] text-ink-600">
        vα
      </div>
    </nav>
  )
}
