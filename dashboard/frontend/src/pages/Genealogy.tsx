import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'

export default function Genealogy() {
  return (
    <div className="p-5">
      <Panel>
        <PanelHeader>
          <PanelTitle>Swarm Genealogy</PanelTitle>
          <span className="eyebrow text-mercury">phase 8</span>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">Agent cards + lineage edges drawn via SVG.</div>
        </PanelBody>
      </Panel>
    </div>
  )
}
