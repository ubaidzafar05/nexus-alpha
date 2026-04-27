import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'

export default function Analytics() {
  return (
    <div className="grid grid-cols-12 gap-4 p-5">
      <Panel className="col-span-12 lg:col-span-8">
        <PanelHeader>
          <PanelTitle>Equity Curve</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="h-[320px] text-mercury">Recharts area chart in phase 8.</div>
        </PanelBody>
      </Panel>
      <Panel className="col-span-12 lg:col-span-4">
        <PanelHeader>
          <PanelTitle>Win / Loss</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="h-[320px] text-mercury">Donut split.</div>
        </PanelBody>
      </Panel>
    </div>
  )
}
