# Care-GraphRAG

Clinical question answering over NICE Clinical Knowledge Summaries, served as a
**Cloudflare Worker** on `https://api.graphrag.care`.

> Rewritten 2026-07-26. The previous version of this file described an AWS
> Lambda + SST + MongoDB Atlas system. **None of that exists any more** — the
> Atlas cluster was deleted, the SST stacks were torn down, and the service was
> rebuilt on Cloudflare in July 2026. The Python code under `src/`, `functions/`
> and `sst.config.ts` is retained for reference only and is **not deployed**.

## What actually runs

`worker/` — a Hono Worker with D1 for the corpus and entity graph.

| Route | Auth | Notes |
|---|---|---|
| `POST /query` | `x-api-key` | Retrieval + answer synthesis |
| `GET /health` | — | Reports bindings, chunk and embedding counts |
| `GET|POST /admin/ingest` | `x-api-key` | Re-scrapes NICE CKS into D1 (see below) |

Consumed by care.engineering through its Worker's `/graphrag/*` proxy, so the
API key stays server-side. The SPA never calls this service directly.

### Retrieval

In-memory cosine similarity over 512-dimension `text-embedding-3-small` vectors,
with entity-graph expansion, then `gpt-4o-mini` synthesises an answer citing NICE
sections. The corpus is small (~107 chunks), so it is loaded into the isolate and
cached, keyed on `meta.corpus_version` — bump that and isolates reload.

The response deliberately keeps the legacy shape the client already read
(`query_id`, `answer`, `sources[]`, `confidence`, `response_time`, `search_type`).
Add fields; never remove or rename one.

**When retrieval clears nothing, the model is not called.** If no chunk scores
above the cosine floor (0.2), `/query` short-circuits to a fixed answer saying
the corpus does not cover the question — `sources: []`, `retrieval_strength: 0`,
`search_type: "none"`, still HTTP 200. Handing `gpt-4o-mini` an empty excerpt
block and trusting the system prompt to refuse is not a guarantee: an
ungrounded completion is an invitation to invent guidance that the client
renders identically to a retrieved, cited answer. Do not "simplify" this back
into the normal path.

**`confidence` is not confidence.** It is the cosine similarity of the single
best-matching chunk — a property of retrieval, saying nothing about whether the
answer is right. An asthma question still scores ~0.4 against hypertension
prose. `confidence` and `confidence_score` are frozen for backward
compatibility with care.engineering; `retrieval_strength` carries the same
value under the honest name and is what new consumers should read. Never label
any of the three as answer confidence in a UI.

`scripts/eval/` holds a 30-question honesty eval (in-scope, out-of-scope traps,
adversarial prompts) that produces a grading sheet for a clinician to mark by
hand. See its README — nothing in it grades automatically, on purpose.

### Storage (D1 `care-graphrag`)

- `chunks` — text plus a base64 `Float32Array` embedding
- `entities`, `links` — the graph; chunks sharing entities pull each other in
- `meta` — `corpus_version`, used to invalidate the isolate cache

## Commands

```bash
cd worker
npm run deploy          # wrangler deploy
npm run db:migrate      # apply schema.sql to remote D1
npm run ingest:enrich   # embed chunks + build the entity graph
npx wrangler tail       # live logs
```

## Re-ingesting the corpus

Two steps, and the first one has a catch.

**1. Scrape (must be triggered from the UK).** NICE geo-blocks CKS to UK IP
addresses. A Worker's outbound fetch egresses from the Cloudflare colo serving
the request, so triggering this from outside the UK fails with 403s on every
page. Open in a UK browser:

```
https://api.graphrag.care/admin/ingest?key=<GRAPHRAG_API_KEY>
```

It replaces the corpus and reports how many pages and chunks it stored. The
response includes the serving colo, which is the quickest way to confirm you
were routed through the UK.

**2. Enrich (runs anywhere).**

```bash
cd worker && OPENAI_API_KEY=… CLOUDFLARE_API_TOKEN=… npm run ingest:enrich
```

Embeds every chunk and extracts entities via `gpt-4o-mini`, writing through
wrangler to remote D1, then bumps `corpus_version`.

**Scope:** currently only the **Hypertension** topic, a snapshot last revised
May 2025. Adding topics means extending `CONTENT_PATHS` in `worker/src/nice.js`
and re-running both steps. Worth a quarterly check for NICE updates.

## Config

Secrets (`wrangler secret put` from `worker/`):

- `OPENAI_API_KEY` — embeddings and answer synthesis
- `GRAPHRAG_API_KEY` — authenticates callers; care.engineering holds the same
  value as its own Worker secret

Vars in `worker/wrangler.toml`: `ALLOWED_ORIGINS`, `NICE_BASE_URL`.
Binding: `DB` → D1 `care-graphrag`.

## Conventions

- Worker code has no Node APIs. `worker/scripts/` is Node and may use them.
- D1 queries use `.prepare(...).bind(...)`. The one exception is
  `embed-and-graph.mjs`, which builds SQL text for the wrangler CLI and escapes
  manually; its inputs are operator-supplied.
- FTS/query tokens are quoted before reaching SQLite so clinical punctuation
  cannot be parsed as query syntax.
- API-key comparison is constant-time. There is no hardcoded fallback key — the
  previous deployment shipped `test-api-key-2024` as a default, which became the
  de-facto production credential.

## Known issues / next steps

- **`deployments/production-deployment-info.json` was removed** but remains in
  git history, containing a live-looking key from the old deployment. That key
  is dead, so this is hygiene: a `git filter-repo` purge across all branches is
  outstanding. Rewriting history does not remove blobs from GitHub's servers
  without contacting Support.
- **`staging-api.graphrag.care` pointed at the old AWS deployment**, which has
  been torn down, so that hostname is dead. Only `api.graphrag.care` is live.
- **No CI.** Unlike care.engineering, this repo has no build gate, preview or
  deploy workflow — deploys are `npm run deploy` by hand.
- **Legacy Python is still in the tree** (`src/`, `functions/`, `scripts/`,
  `sst.config.ts`, `layers/`). It documents the original graph-building approach
  and the NICE scraping logic, but nothing runs it. A cleanup PR removing it
  would make the repo considerably easier to read.
