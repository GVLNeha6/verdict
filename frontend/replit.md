# VERDICT

Evidence-based AI claim verification: submit a factual claim, review a normalized verdict, inspect supporting evidence, and optionally open the secondary AI debate.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/verdict/src/App.tsx` — single-page verification flow and state machine
- `artifacts/verdict/src/components/` — claim input, loading, report, evidence, and debate UI
- `artifacts/verdict/src/services/api.ts` — mock adapter, `/api/verify` seam, and backend normalization
- `artifacts/verdict/src/data/mockResult.ts` — realistic complete mock response for frontend-only operation
- `artifacts/verdict/src/index.css` — VERDICT visual system and responsive styles
- `artifacts/api-server/` — shared API server; unchanged by the frontend build
- `lib/api-spec/openapi.yaml` — shared API contract source of truth

## Architecture decisions

- The frontend defaults to a local mock adapter and switches to the production endpoint only when `VITE_VERDICT_MOCK=false`.
- Backend-specific verdict and debate shapes are normalized once in the API service so React components use frontend-owned data types.
- Evidence relevance is kept separate from verdict confidence and optional source metadata is never fabricated.
- The report stays on one route with local IDLE, LOADING, SUCCESS, and ERROR state instead of adding unnecessary navigation.

## Product

VERDICT helps users check factual claims through a calm, research-oriented flow: it presents a verdict first, then confidence, explanation, evidence, and a collapsed AI debate for deeper inspection.

## User preferences

The frontend brief prioritizes product-level polish, restrained motion, accessibility, responsive behavior, and no unrelated marketing or account features.

## Gotchas

- The Vite build expects `PORT` and `BASE_PATH`; managed workflows provide them automatically.
- The current backend contract provides evidence without guaranteed `source` or `url`; the UI must keep those fields optional.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
