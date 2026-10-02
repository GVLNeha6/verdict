import { RotateCcw } from 'lucide-react';
import type { VerificationResult as VerificationResultData } from '@/services/api';
import { DebateDetails } from './DebateDetails';
import { EvidenceList } from './EvidenceList';
import { VerdictCard } from './VerdictCard';

interface VerificationResultProps {
  result: VerificationResultData;
  onReset: () => void;
}

export function VerificationResult({ result, onReset }: VerificationResultProps) {
  return (
    <div className="result-wrap reveal" data-testid="verification-result">
      <div className="result-topline"><h1 className="eyebrow">01 / Assessment</h1><button className="reset-button" onClick={onReset} data-testid="button-verify-another"><RotateCcw size={14} /> Verify another claim</button></div>
      <section className="claim-review" data-testid="section-submitted-claim"><span className="micro-label">Submitted claim</span><blockquote>{result.claim}</blockquote></section>
      <VerdictCard verdict={result.verdict} confidence={result.confidence} />
      <section className="explanation-section" data-testid="section-explanation"><span className="eyebrow">Why</span><h2>The short answer</h2><p>{result.explanation}</p></section>
      <EvidenceList evidence={result.evidence} />
      <DebateDetails debate={result.debate} />
    </div>
  );
}