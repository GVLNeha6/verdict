import { Check, Circle, Search, Scale, Sparkles } from 'lucide-react';

const stages = [
  { label: 'Verifying claim', icon: Check },
  { label: 'Retrieving evidence', icon: Search },
  { label: 'Running AI debate', icon: Scale },
  { label: 'Generating verdict', icon: Sparkles },
];

interface LoadingStateProps {
  claim: string;
}

export function LoadingState({ claim }: LoadingStateProps) {
  return (
    <section className="loading-state reveal" aria-live="polite" aria-busy="true" data-testid="status-loading">
      <div className="loading-heading">
        <span className="eyebrow">Research in progress</span>
        <h2>Taking a careful look.</h2>
        <p>VERDICT is preparing a final report. The steps below describe the review process conceptually; the backend returns one completed result.</p>
      </div>
      <div className="loading-claim"><span>Reviewing</span><strong>{claim}</strong></div>
      <ol className="stage-list">
        {stages.map(({ label, icon: Icon }, index) => (
          <li className="stage-item" key={label} data-testid={`stage-${index}`}>
            <span className={`stage-icon stage-icon-${index}`}><Icon size={16} aria-hidden="true" /></span>
            <span>{label}</span>
            <span className="stage-line" aria-hidden="true" />
            <span className="stage-state">conceptual</span>
          </li>
        ))}
      </ol>
      <div className="skeleton-block" aria-hidden="true"><Circle className="skeleton-orb" size={18} /><span /></div>
    </section>
  );
}