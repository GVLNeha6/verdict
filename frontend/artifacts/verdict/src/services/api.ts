import { mockResult } from '@/data/mockResult';

export type Verdict = 'SUPPORTED' | 'REFUTED' | 'UNCERTAIN';

export interface EvidenceItem {
  title?: string;
  sentence_id?: string;
  text: string;
  score?: number;
  source?: string;
  url?: string;
}

export interface Debate {
  agents: string[];
  rounds: number;
  final_reasoning: {
    round: number;
    agent_1: string;
    agent_2: string;
  };
}

export interface VerificationResult {
  claim: string;
  verdict: Verdict;
  confidence: number;
  explanation: string;
  evidence: EvidenceItem[];
  debate: Debate;
}

export type VerificationErrorCode =
  | 'bad-request'
  | 'unprocessable'
  | 'server'
  | 'network'
  | 'timeout'
  | 'invalid-response'
  | 'unknown';

export class VerificationApiError extends Error {
  constructor(
    public readonly code: VerificationErrorCode,
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = 'VerificationApiError';
  }
}

export const isMockMode = import.meta.env.VITE_VERDICT_MOCK === 'true';

const INVALID_RESPONSE_MESSAGE =
  'The verification service returned an unexpected result. Please try again.';

type RecordValue = Record<string, unknown>;

function isRecord(value: unknown): value is RecordValue {
  return typeof value === 'object' && value !== null;
}

function invalidResponse(): never {
  throw new VerificationApiError('invalid-response', INVALID_RESPONSE_MESSAGE);
}

function requiredString(value: unknown): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    return invalidResponse();
  }
  return value;
}

function requiredFiniteNumber(value: unknown, min?: number, max?: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return invalidResponse();
  }
  if (min !== undefined && value < min) {
    return invalidResponse();
  }
  if (max !== undefined && value > max) {
    return invalidResponse();
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'string') return invalidResponse();
  return value;
}

function safeUrl(value: unknown): string | undefined {
  const url = optionalString(value);
  if (url === undefined) return undefined;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return invalidResponse();
    }
  } catch {
    return invalidResponse();
  }
  return url;
}

function normalizeVerdict(value: unknown): Verdict {
  if (value === 'SUPPORTS') return 'SUPPORTED';
  if (value === 'REFUTES') return 'REFUTED';
  if (value === 'NOT ENOUGH INFO') return 'UNCERTAIN';
  return invalidResponse();
}

function normalizeEvidence(value: unknown): EvidenceItem[] {
  if (!Array.isArray(value)) return invalidResponse();

  return value.map((item) => {
    if (!isRecord(item)) return invalidResponse();

    const sentenceId = item.sentence_id;
    if (
      sentenceId !== undefined &&
      (typeof sentenceId !== 'string' ||
        sentenceId.trim().length === 0)
    ) {
      if (
        typeof sentenceId !== 'number' ||
        !Number.isFinite(sentenceId)
      ) {
        return invalidResponse();
      }
    }

    const score =
      item.score === undefined
        ? undefined
        : requiredFiniteNumber(item.score);

    return {
      title: optionalString(item.title),
      sentence_id:
        sentenceId === undefined ? undefined : String(sentenceId),
      text: requiredString(item.text),
      score,
      source: optionalString(item.source),
      url: safeUrl(item.url),
    };
  });
}

function normalizeDebate(value: unknown): Debate {
  if (!isRecord(value)) return invalidResponse();

  const agentCount = requiredFiniteNumber(value.agents, 1);
  if (!Number.isInteger(agentCount)) return invalidResponse();

  const rounds = requiredFiniteNumber(value.rounds, 1);
  if (!Number.isInteger(rounds)) return invalidResponse();

  if (!isRecord(value.final_reasoning)) return invalidResponse();
  const reasoning = value.final_reasoning;
  const round = requiredFiniteNumber(reasoning.round, 1);
  if (!Number.isInteger(round)) return invalidResponse();

  const agentNames = Array.from({ length: agentCount }, (_, index) =>
    index === 0
      ? 'Evidence analyst'
      : index === 1
        ? 'Skeptical reviewer'
        : `Reviewer ${index + 1}`,
  );

  return {
    agents: agentNames,
    rounds,
    final_reasoning: {
      round,
      agent_1: requiredString(reasoning.agent_1),
      agent_2: requiredString(reasoning.agent_2),
    },
  };
}

function normalizeResult(raw: unknown): VerificationResult {
  if (!isRecord(raw)) return invalidResponse();

  return {
    claim: requiredString(raw.claim),
    verdict: normalizeVerdict(raw.verdict),
    confidence: requiredFiniteNumber(raw.confidence, 0, 1),
    explanation: requiredString(raw.explanation),
    evidence: normalizeEvidence(raw.evidence),
    debate: normalizeDebate(raw.debate),
  };
}

function mockVerification(claim: string): Promise<VerificationResult> {
  return new Promise((resolve) => {
    window.setTimeout(() => {
      const lower = claim.toLowerCase();
      const result = structuredClone(mockResult);
      result.claim = claim;
      if (/\bnever\b|\bimpossible\b|\bflat earth\b|\bcauses\b/.test(lower)) {
        result.verdict = 'REFUTED';
        result.confidence = 0.79;
        result.explanation =
          'The available evidence contradicts the claim as written. Some adjacent ideas may be true, but they do not support this absolute or causal wording.';
      } else if (/\bmaybe\b|\baliens\b|\balways\b|\bguarantee\b/.test(lower)) {
        result.verdict = 'UNCERTAIN';
        result.confidence = 0.48;
        result.explanation =
          'The claim cannot be settled confidently from the evidence available. Its wording is too broad, or the underlying question remains actively debated.';
      }
      resolve(result);
    }, 1450);
  });
}

function errorForStatus(status: number): VerificationApiError {
  if (status === 400) {
    return new VerificationApiError(
      'bad-request',
      'Please enter a valid claim and try again.',
      status,
    );
  }
  if (status === 422) {
    return new VerificationApiError(
      'unprocessable',
      'The verification request could not be processed.',
      status,
    );
  }
  if (status >= 500) {
    return new VerificationApiError(
      'server',
      'The verification service is temporarily unavailable. Please try again.',
      status,
    );
  }
  return new VerificationApiError(
    'unknown',
    'The verification service could not complete the request. Please try again.',
    status,
  );
}

export async function verifyClaim(claim: string): Promise<VerificationResult> {
  if (isMockMode) return mockVerification(claim);

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 120000);

  try {
    const response = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ claim }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw errorForStatus(response.status);
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      throw new VerificationApiError(
        'invalid-response',
        INVALID_RESPONSE_MESSAGE,
      );
    }
    return normalizeResult(payload);
  } catch (error) {
    if (error instanceof VerificationApiError) throw error;
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new VerificationApiError(
        'timeout',
        'The request took too long. Check your connection and try again.',
      );
    }
    if (error instanceof TypeError) {
      throw new VerificationApiError(
        'network',
        'We could not reach the verification service. Check your connection and try again.',
      );
    }
    throw new VerificationApiError(
      'unknown',
      'Something went wrong while verifying this claim. Please try again.',
    );
  } finally {
    window.clearTimeout(timeout);
  }
}