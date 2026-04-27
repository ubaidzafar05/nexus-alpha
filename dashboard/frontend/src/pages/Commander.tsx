import { Panel, PanelBody, PanelHeader, PanelTitle } from '@/components/shared/Panel'

export default function Commander() {
  return (
    <div className="grid auto-rows-min grid-cols-12 gap-4 p-5">
      <Panel className="col-span-12 lg:col-span-8">
        <PanelHeader>
          <PanelTitle>Hero · Portfolio</PanelTitle>
          <span className="eyebrow text-mercury">phase 6</span>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">
            Serif NAV display, 24h sparkline, equity meter — lands next phase.
          </div>
        </PanelBody>
      </Panel>
      <Panel className="col-span-12 lg:col-span-4">
        <PanelHeader>
          <PanelTitle>Controls</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">Pause · Resume · Market exit — next phase.</div>
        </PanelBody>
      </Panel>
      <Panel className="col-span-12 md:col-span-6 lg:col-span-4">
        <PanelHeader>
          <PanelTitle>Win % · PF · MDD</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">KPI grid with inline meters.</div>
        </PanelBody>
      </Panel>
      <Panel className="col-span-12 md:col-span-6 lg:col-span-4">
        <PanelHeader>
          <PanelTitle>Tournament</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">Champion vs contenders chart.</div>
        </PanelBody>
      </Panel>
      <Panel className="col-span-12 md:col-span-6 lg:col-span-4">
        <PanelHeader>
          <PanelTitle>Microstructure</PanelTitle>
        </PanelHeader>
        <PanelBody>
          <div className="text-mercury">Spread · Depth · OFI · Toxicity.</div>
        </PanelBody>
      </Panel>
    </div>
  )
}
