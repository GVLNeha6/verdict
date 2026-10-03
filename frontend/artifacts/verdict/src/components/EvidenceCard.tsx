import { ExternalLink, FileText } from 'lucide-react';
import type { EvidenceItem } from '@/services/api';

interface EvidenceCardProps {
  evidence: EvidenceItem;
  index: number;
}

export function EvidenceCard({ evidence, index }: EvidenceCardProps) {
  const relevance = typeof evidence.score === 'number' ? Math.round(Math.max(0, Math.min(1, evidence.score)) * 100) : null;
  return (
    <article className="evidence-card" data-testid={`card-evidence-${index}`}>
      <div className="evidence-index">{String(index + 1).padStart(2, '0')}</div>
      <div className="evidence-body">
        <div className="evidence-title-row">
          <FileText size={15} aria-hidden="true" />
          {evidence.title && <h3>{evidence.title}</h3>}
          {evidence.sentence_id && <span className="sentence-id">{evidence.sentence_id}</span>}
        </div>
        {evidence.text && <p>{evidence.text}</p>}
        <div className="evidence-meta">
          {relevance !== null && <span><b>{relevance}%</b> retrieval relevance</span>}
          {evidence.source && <span>{evidence.source}</span>}
          {evidence.url && <a href={evidence.url} target="_blank" rel="noreferrer" data-testid={`link-evidence-${index}`}>Open source <ExternalLink size={12} /></a>}
        </div>
      </div>
    </article>
  );
}