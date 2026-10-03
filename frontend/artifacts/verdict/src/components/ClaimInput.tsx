import { ArrowUpRight, CornerDownLeft } from 'lucide-react';
import { type FormEvent, type KeyboardEvent, useState } from 'react';

interface ClaimInputProps {
  disabled?: boolean;
  initialValue?: string;
  onSubmit: (claim: string) => void;
}

export function ClaimInput({ disabled = false, initialValue = '', onSubmit }: ClaimInputProps) {
  const [claim, setClaim] = useState(initialValue);
  const trimmed = claim.trim();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!disabled) onSubmit(trimmed);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      if (!disabled) onSubmit(trimmed);
    }
  }

  return (
    <form className="claim-form" onSubmit={handleSubmit} data-testid="form-claim">
      <label className="eyebrow" htmlFor="claim">Your claim</label>
      <div className="input-frame">
        <textarea
          id="claim"
          data-testid="input-claim"
          value={claim}
          onChange={(event) => setClaim(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. Regular exercise improves memory in older adults."
          rows={4}
          maxLength={1200}
          disabled={disabled}
          aria-describedby="claim-hint"
        />
        <div className="input-footer">
          <span id="claim-hint" className="input-hint">
            State one factual claim. VERDICT checks the wording as written.
          </span>
          <span className="character-count">{claim.length}/1200</span>
        </div>
      </div>
      <button className="submit-button" type="submit" disabled={disabled} data-testid="button-submit-claim">
        {disabled ? 'Reviewing claim' : 'Verify claim'}
        {!disabled && <ArrowUpRight size={17} aria-hidden="true" />}
      </button>
      {!disabled && <div className="keyboard-hint"><CornerDownLeft size={13} /> Ctrl + Enter to submit</div>}
    </form>
  );
}