import { ChevronDown, MessagesSquare } from 'lucide-react';
import type { Debate } from '@/services/api';

interface DebateDetailsProps {
  debate?: Debate;
}

export function DebateDetails({ debate }: DebateDetailsProps) {
  if (!debate) return null;
  return (
    <details className="debate-details" data-testid="details-debate">
      <summary data-testid="button-toggle-debate"><span className="debate-summary-icon"><MessagesSquare size={16} /></span><span><b>AI debate</b><small>{debate.rounds || 'Multiple'} rounds · two reviewing agents</small></span><ChevronDown className="chevron" size={18} aria-hidden="true" /></summary>
      <div className="debate-content">
        <div className="agent-row"><span className="agent-number">A1</span><span>{debate.agents[0] ?? 'Evidence analyst'}</span><span className="agent-number">A2</span><span>{debate.agents[1] ?? 'Skeptical reviewer'}</span></div>
        <div className="reasoning-grid">
          <div><span className="micro-label">Round {debate.final_reasoning.round || debate.rounds || 1} · Agent one</span><p>{debate.final_reasoning.agent_1}</p></div>
          <div><span className="micro-label">Round {debate.final_reasoning.round || debate.rounds || 1} · Agent two</span><p>{debate.final_reasoning.agent_2}</p></div>
        </div>
      </div>
    </details>
  );
}