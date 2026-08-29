import type { ReactNode } from "react";

interface ChildrenProps {
  children: ReactNode;
}

export function AppShell({ children }: ChildrenProps) {
  return <div className="app">{children}</div>;
}

export function ProjectRail({ open, children }: ChildrenProps & { open: boolean }) {
  return (
    <aside
      id="project-library"
      className={`library ${open ? "is-open" : ""}`}
      aria-label="Project library"
    >
      {children}
    </aside>
  );
}

export function PlanInspector({ open, children }: ChildrenProps & { open: boolean }) {
  return (
    <section
      id="composition-plan"
      className={`pane plan ${open ? "is-open" : ""}`}
      aria-label="Composition plan"
    >
      {children}
    </section>
  );
}

export function StudioStage({ children }: ChildrenProps) {
  return (
    <section className="pane waveform" aria-label="Waveform studio">
      {children}
    </section>
  );
}

export function StudioPlayer({ children }: ChildrenProps) {
  return <div className="stage-transport">{children}</div>;
}

export function OperationDock({ children }: ChildrenProps) {
  return <div className="operation-dock">{children}</div>;
}

export function TakesRail({ open, children }: ChildrenProps & { open: boolean }) {
  return (
    <aside
      id="studio-inspector"
      className={`studio-inspector ${open ? "is-open" : ""}`}
      aria-label="Project takes and style packs"
    >
      {children}
    </aside>
  );
}

export function StylePackPanel({ active, children }: ChildrenProps & { active: boolean }) {
  return (
    <section className="style-packs" hidden={!active}>
      {children}
    </section>
  );
}
