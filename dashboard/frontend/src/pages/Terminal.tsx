import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'

export default function Terminal() {
  return (
    <div className="grid h-full grid-cols-12 gap-4 p-5">
      <Panel className="col-span-12 lg:col-span-8">
        <PanelHeader>
          <PanelTitle>Chart</PanelTitle>
          <span className="eyebrow text-mercury">BTC · ETH · SOL</span>
        </PanelHeader>
        <PanelBody>
          <div className="h-[480px] text-mercury">
            TradingView lightweight chart mounts in phase 7.
          </div>
        </PanelBody>
      </Panel>
      <Panel className="col-span-12 lg:col-span-4">
        <PanelHeader>
          <PanelTitle>Signal Feed</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">Virtualized stream of recent signals.</div>
        </PanelBody>
      </Panel>
    </div>
  )
}
