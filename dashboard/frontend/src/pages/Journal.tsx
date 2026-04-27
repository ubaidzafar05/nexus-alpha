import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'

export default function Journal() {
  return (
    <div className="p-5">
      <Panel>
        <PanelHeader>
          <PanelTitle>Trade Journal</PanelTitle>
          <span className="eyebrow text-mercury">virtualized · sortable · filterable</span>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">TanStack Table lands in phase 8.</div>
        </PanelBody>
      </Panel>
    </div>
  )
}
