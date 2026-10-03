import { ArchiveX } from 'lucide-react';
import type { EvidenceItem } from '@/services/api';
import { EvidenceCard } from './EvidenceCard';

interface EvidenceListProps {
  evidence: EvidenceItem[];
}

export function EvidenceList({ evidence }: EvidenceListProps) {
  return (
    <section className="evidence-section" data-testid="section-evidence">
      <div className="section-heading">
        <div><span className="eyebrow">02 / Evidence</span><h2>What we found</h2></div>
        <span className="count-badge" data-testid="text-evidence-count">{evidence.length} {evidence.length === 1 ? 'item' : 'items'}</span>
      </div>
      {evidence.length > 0 ? (
        <div className="evidence-list">{evidence.map((item, index) => <EvidenceCard key={`${item.sentence_id ?? 'evidence'}-${index}`} evidence={item} index={index} />)}</div>
      ) : (
        <div className="empty-evidence" data-testid="empty-evidence"><ArchiveX size={23} /><strong>No evidence was returned</strong><p>The verifier did not receive retrievable passages for this claim. Treat the result cautiously.</p></div>
      )}
    </section>
  );
}