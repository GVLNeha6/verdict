import type { VerificationResult } from '@/services/api';

export const mockResult: VerificationResult = {
  claim: '',
  verdict: 'SUPPORTED',
  confidence: 0.86,
  explanation:
    'The claim is consistent with the strongest available evidence. The sources converge on the underlying mechanism, though the wording is broader than the research itself.',
  evidence: [
    {
      title: 'A review of the underlying research',
      sentence_id: 's-01',
      text: 'The study found a repeatable relationship under the conditions described in the claim.',
      score: 0.91,
    },
    {
      title: 'Independent replication and context',
      sentence_id: 's-02',
      text: 'A later analysis reached a similar conclusion while noting that results vary by population and measurement.',
      score: 0.82,
    },
    {
      title: 'Limits on how far the finding travels',
      sentence_id: 's-03',
      text: 'The evidence supports the central direction of the claim, but does not establish that it holds in every setting.',
      score: 0.74,
    },
  ],
  debate: {
    agents: ['Evidence Analyst', 'Skeptical Reviewer'],
    rounds: 2,
    final_reasoning: {
      round: 2,
      agent_1: 'The evidence supports the core statement with appropriate caveats.',
      agent_2: 'No decisive contradiction was found, but the scope should not be overstated.',
    },
  },
};