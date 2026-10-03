import { AlertCircle, ShieldCheck } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import { ClaimInput } from '@/components/ClaimInput';
import { Header } from '@/components/Header';
import { LoadingState } from '@/components/LoadingState';
import { VerificationResult } from '@/components/VerificationResult';
import { isMockMode, verifyClaim, type VerificationResult as VerificationResultData } from '@/services/api';

type AppState = 'idle' | 'loading' | 'success' | 'error';

function Home() {
  const [state, setState] = useState<AppState>('idle');
  const [result, setResult] = useState<VerificationResultData | null>(null);
  const [lastClaim, setLastClaim] = useState('');
  const [error, setError] = useState('');
  const [inputKey, setInputKey] = useState(0);
  const errorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (state === 'error') errorRef.current?.focus();
  }, [state]);

  async function handleSubmit(claim: string) {
    if (state === 'loading') return;
    if (!claim.trim()) {
      setError('Enter a factual claim before submitting. A specific sentence works best.');
      setState('error');
      window.setTimeout(() => document.getElementById('claim')?.focus(), 0);
      return;
    }
    setLastClaim(claim);
    setError('');
    setState('loading');
    try {
      const verified = await verifyClaim(claim);
      setResult(verified);
      setState('success');
      window.setTimeout(() => document.querySelector('[data-testid="verification-result"]')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'We could not complete the review. Please try again.');
      setState('error');
    }
  }

  function reset() {
    setState('idle');
    setResult(null);
    setError('');
    setLastClaim('');
    setInputKey((key) => key + 1);
    window.setTimeout(() => document.getElementById('claim')?.focus(), 0);
  }

  return (
    <div className="verdict-page">
      <Header isMockMode={isMockMode} />
      <main className="app-main">
        {state !== 'success' && (
          <section className="desk-intro reveal">
            <div className="intro-rule"><span>RESEARCH DESK / 001</span><span className="intro-rule-line" /></div>
            <h1>Make the claim.<br /><em>We’ll check it.</em></h1>
            <p className="intro-copy">A measured answer to a factual question. VERDICT retrieves relevant evidence, compares opposing reasoning, and shows you what supports the result.</p>
            <ClaimInput key={inputKey} disabled={state === 'loading'} onSubmit={handleSubmit} />
            {error && (
              <div ref={errorRef} className="error-message" role="alert" tabIndex={-1} data-testid="status-error">
                <AlertCircle size={18} aria-hidden="true" />
                <div><strong>Review could not be completed</strong><p>{error}</p></div>
                {lastClaim && state === 'error' && <button type="button" onClick={() => handleSubmit(lastClaim)} data-testid="button-retry">Retry</button>}
              </div>
            )}
            <div className="desk-principle"><ShieldCheck size={16} /><span>No account. No conversational padding. Just the claim and the evidence.</span></div>
          </section>
        )}
        {state === 'loading' && <LoadingState claim={lastClaim} />}
        {state === 'success' && result && <VerificationResult result={result} onReset={reset} />}
        {state === 'idle' && <div className="idle-footer"><span>Built for questions where “probably” is not enough.</span><span className="footer-mark">v.01 / mock research adapter</span></div>}
      </main>
    </div>
  );
}

function Router() {
  return <Switch><Route path="/" component={Home} /><Route component={Home} /></Switch>;
}

function App() {
  return <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router /></WouterRouter>;
}

export default App;