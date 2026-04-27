import { useEffect } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import Commander from '@/pages/Commander'
import Terminal from '@/pages/Terminal'
import Journal from '@/pages/Journal'
import Genealogy from '@/pages/Genealogy'
import Analytics from '@/pages/Analytics'
import { connectWebSocket } from '@/lib/ws'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2_000,
      gcTime: 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function WsLifecycle() {
  useEffect(() => {
    const disconnect = connectWebSocket(queryClient)
    return () => disconnect()
  }, [])
  return null
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WsLifecycle />
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<Commander />} />
            <Route path="/terminal" element={<Terminal />} />
            <Route path="/journal" element={<Journal />} />
            <Route path="/genealogy" element={<Genealogy />} />
            <Route path="/analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
