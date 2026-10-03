import { CheckCircle2, HelpCircle, XCircle } from 'lucide-react';
import type { Verdict } from '@/services/api';

interface VerdictCardProps {
  confidence: number;
  verdict: Verdict;
}

const verdictCopy: Record<Verdict, { label: string; description: string; icon: typeof CheckCircle2 }> = {
  SUPPORTED: { label: 'Supported', description: 'The evidence supports this claim.', icon: CheckCircle2 },
  REFUTED: { label: 'Refuted', description: 'The evidence contradicts this claim.', icon: XCircle },
  UNCERTAIN: { label: 'Uncertain', description: 'The evidence is not decisive enough.', icon: HelpCircle },
};

export function VerdictCard({ confidence, verdict }: VerdictCardProps) {
  const copy = verdictCopy[verdict];
  const Icon = copy.icon;
  const percentage = Math.round(confidence * 100);
  return (
    <section className={`verdict-card verdict-${verdict.toLowerCase()}`} data-testid="card-verdict">
      <div className="verdict-label"><Icon size={24} strokeWidth={1.8} aria-hidden="true" /><span>Verdict</span></div>
      <div className="verdict-word" data-testid="status-verdict">{copy.label}</div>
      <p>{copy.description}</p>
      <div className="confidence-row">
        <div>
          <span className="micro-label">Confidence</span>
          <strong data-testid="text-confidence">{percentage}%</strong>
        </div>
        <div className="confidence-meter" role="meter" aria-label="Verdict confidence" aria-valuenow={percentage} aria-valuemin={0} aria-valuemax={100}>
          <span style={{ transform: `scaleX(${confidence})` }} />
        </div>
      </div>
    </section>
  );
}