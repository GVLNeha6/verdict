import { ClipboardCheck } from 'lucide-react';

interface HeaderProps {
  isMockMode: boolean;
}

export function Header({ isMockMode }: HeaderProps) {
  return (
    <header className="site-header" data-testid="header-verdict">
      <div className="header-inner">
        <a className="brand-mark" href="/" data-testid="link-home">
          <span className="brand-icon" aria-hidden="true"><ClipboardCheck size={17} strokeWidth={2.4} /></span>
          <span>VERDICT</span>
        </a>
        <div className="header-note" data-testid="mode-indicator">
          <span className="status-dot" aria-hidden="true" />
          <span>{isMockMode ? 'Mock data' : 'Live verification'}</span>
        </div>
      </div>
    </header>
  );
}