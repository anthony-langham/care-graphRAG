import { Hono } from "hono";
import { cors } from "hono/cors";
import { extractSections, chunkSections, CONTENT_PATHS } from "./nice.js";

const app = new Hono();

// --- CORS: origin allowlist from env (comma-separated) -------------------
app.use("*", async (c, next) => {
  const allowed = (c.env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((o) => o.trim())
    .filter(Boolean);
  const handler = cors({
    origin: (origin) => (allowed.includes(origin) ? origin : null),
    allowHeaders: ["Content-Type", "x-api-key"],
    allowMethods: ["GET", "POST", "OPTIONS"],
    maxAge: 86400,
  });
  return handler(c, next);
});

// --- API-key auth --------------------------------------------------------

function timingSafeEqual(a, b) {
  const enc = new TextEncoder();
  const ab = enc.encode(a);
  const bb = enc.encode(b);
  if (ab.length !== bb.length) return false;
  let diff = 0;
  for (let i = 0; i < ab.length; i++) diff |= ab[i] ^ bb[i];
  return diff === 0;
}

function apiKeyRequired() {
  return async (c, next) => {
    const key = c.req.header("x-api-key") || "";
    const expected = c.env.GRAPHRAG_API_KEY || "";
    if (!expected || !timingSafeEqual(key, expected)) {
      return c.json({ error: "Invalid API key" }, 401);
    }
    await next();
  };
}

// --- Embeddings ----------------------------------------------------------

// Not exported: workerd rejects any named export from the entry module that
// is not a handler ("Incorrect type for map entry 'EMBEDDING_DIMS'"), which
// stops `wrangler dev` booting at all. Nothing imported these — the
// enrichment script keeps its own copies — so they are module-local.
// They must stay in step with scripts/embed-and-graph.mjs: a mismatch in
// dimensions silently produces meaningless cosine scores.
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIMS = 512;

async function embedQuery(text, env) {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      dimensions: EMBEDDING_DIMS,
      input: text,
    }),
  });
  if (!response.ok) {
    throw new Error(`Embedding request failed (${response.status})`);
  }
  const data = await response.json();
  return Float32Array.from(data.data[0].embedding);
}

function decodeEmbedding(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function cosine(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// --- Corpus cache (per isolate) ------------------------------------------
// The corpus is small (a few hundred chunks) and static between ingests;
// meta.corpus_version invalidates the cache after re-ingestion.

let corpusCache = null; // { version, chunks: [{id,url,page_title,section,text,vec}] }

async function loadCorpus(env) {
  const versionRow = await env.DB.prepare(
    "SELECT v FROM meta WHERE k = 'corpus_version'",
  ).first();
  const version = versionRow?.v || "0";
  if (corpusCache && corpusCache.version === version) return corpusCache;

  const { results } = await env.DB.prepare(
    "SELECT id, url, page_title, section, text, embedding FROM chunks WHERE embedding IS NOT NULL",
  ).all();
  corpusCache = {
    version,
    chunks: results.map((r) => ({
      id: r.id,
      url: r.url,
      page_title: r.page_title,
      section: r.section,
      text: r.text,
      vec: decodeEmbedding(r.embedding),
    })),
  };
  return corpusCache;
}

// --- Graph expansion -----------------------------------------------------
// Chunks sharing entities with the vector hits get pulled in (and vector
// hits that share entities get boosted) — graph-first, vector-fallback.

async function graphNeighbours(env, chunkIds) {
  if (chunkIds.length === 0) return { neighbours: new Map(), entities: new Map() };
  const placeholders = chunkIds.map(() => "?").join(",");
  const { results } = await env.DB.prepare(
    `SELECT l2.chunk_id AS neighbour_id, e.name, e.type, l1.chunk_id AS seed_id
     FROM links l1
     JOIN entities e ON e.id = l1.entity_id
     JOIN links l2 ON l2.entity_id = l1.entity_id
     WHERE l1.chunk_id IN (${placeholders})`,
  )
    .bind(...chunkIds)
    .all();

  const neighbours = new Map(); // neighbour chunk id -> shared-entity count
  const entities = new Map(); // chunk id -> [{name,type}]
  for (const row of results) {
    if (!chunkIds.includes(row.neighbour_id)) {
      neighbours.set(row.neighbour_id, (neighbours.get(row.neighbour_id) || 0) + 1);
    }
    const list = entities.get(row.seed_id) || [];
    if (!list.some((e) => e.name === row.name)) list.push({ name: row.name, type: row.type });
    entities.set(row.seed_id, list);
  }
  return { neighbours, entities };
}

// --- Retrieval honesty ---------------------------------------------------
// A chunk has to clear this cosine floor to be worth showing the model at
// all. Below it we treat the corpus as having nothing to say.
const RELEVANCE_FLOOR = 0.2;

// Returned verbatim when retrieval clears nothing. Naming the corpus's actual
// scope is the useful part: it tells the clinician where to ask instead,
// rather than leaving them to guess whether the system broke or the answer is
// genuinely absent.
const NO_RELEVANT_CONTEXT_ANSWER =
  "The NICE CKS Hypertension corpus does not contain guidance relevant to this question. " +
  "Try the corpus's own scope: diagnosis, investigation and management of hypertension in adults. " +
  "This tool supports, never replaces, professional clinical judgement.";

// --- Routes --------------------------------------------------------------

app.get("/health", async (c) => {
  let chunkCount = null;
  let embeddedCount = null;
  let dbOk = false;
  try {
    const row = await c.env.DB.prepare(
      "SELECT COUNT(*) AS total, SUM(embedding IS NOT NULL) AS embedded FROM chunks",
    ).first();
    chunkCount = row.total;
    embeddedCount = row.embedded || 0;
    dbOk = true;
  } catch {
    dbOk = false;
  }
  const ready = dbOk && embeddedCount > 0 && Boolean(c.env.OPENAI_API_KEY);
  return c.json({
    status: ready ? "ok" : "degraded",
    service: "care-graphrag",
    version: "2.0.0",
    platform: "cloudflare-workers",
    timestamp: new Date().toISOString(),
    checks: {
      database: dbOk,
      openai_key_configured: Boolean(c.env.OPENAI_API_KEY),
      api_key_configured: Boolean(c.env.GRAPHRAG_API_KEY),
      chunks: chunkCount,
      chunks_embedded: embeddedCount,
    },
  });
});

app.post("/query", apiKeyRequired(), async (c) => {
  const started = Date.now();
  const body = await c.req.json().catch(() => null);
  const question = body?.question?.trim();
  if (!question) {
    return c.json(
      { detail: [{ type: "missing", loc: ["body", "question"], msg: "Field required" }] },
      422,
    );
  }
  if (question.length > 1000) {
    return c.json({ error: "Question too long (max 1000 characters)" }, 400);
  }

  const corpus = await loadCorpus(c.env);
  if (corpus.chunks.length === 0) {
    return c.json({ error: "Corpus not ingested yet" }, 503);
  }

  // If we cannot embed the question we cannot retrieve, and without retrieval
  // there is nothing honest to say. Answer with the same machine-readable 502
  // the completion failure below uses, rather than letting the throw escape to
  // Hono's default handler — that returned a plain-text 500 body, which the
  // care.engineering client cannot parse for an error message and so surfaces
  // as an unexplained failure.
  let queryVec;
  try {
    queryVec = await embedQuery(question, c.env);
  } catch (err) {
    // The question itself is never logged: it can carry patient detail.
    console.error("Query embedding failed:", String(err).slice(0, 200));
    return c.json({ error: "Search failed" }, 502);
  }

  const scored = corpus.chunks
    .map((ch) => ({ ...ch, score: cosine(queryVec, ch.vec) }))
    .sort((a, b) => b.score - a.score);

  const TOP_K = 6;
  const seeds = scored.slice(0, TOP_K).filter((ch) => ch.score > RELEVANCE_FLOOR);

  // Retrieval found nothing above the floor. Answer here and do NOT call the
  // model.
  //
  // The previous behaviour was to carry on and hand gpt-4o-mini an EMPTY
  // excerpt block, leaving the whole refusal burden on the system prompt's
  // "if the excerpts do not contain the answer, say so plainly". That is a
  // request, not a guarantee: a completion with zero grounding context is an
  // invitation to answer from the model's own weights, and it will sometimes
  // accept — producing fluent, plausible, uncited clinical guidance that the
  // client renders identically to a real retrieval-backed answer. In a
  // clinical tool the failure has to be visible: no sources, zero retrieval
  // strength, search_type "none", and an answer that says outright the corpus
  // does not cover this. A wrong-but-plausible answer here is worse than no
  // answer, so we make it impossible rather than unlikely.
  //
  // Still HTTP 200 with the standard response shape: this is a legitimate,
  // fully-determined result ("not in scope"), not a server error, and the
  // client renders it through its normal answer path.
  if (seeds.length === 0) {
    return c.json({
      query_id: crypto.randomUUID(),
      answer: NO_RELEVANT_CONTEXT_ANSWER,
      sources: [],
      confidence: 0,
      confidence_score: 0,
      retrieval_strength: 0,
      response_time: (Date.now() - started) / 1000,
      search_type: "none",
    });
  }

  const { neighbours, entities } = await graphNeighbours(
    c.env,
    seeds.map((s) => s.id),
  );

  // Pull in up to 3 graph neighbours not already in the seed set.
  const byId = new Map(scored.map((ch) => [ch.id, ch]));
  const graphPicks = [...neighbours.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([id]) => byId.get(id))
    .filter(Boolean);

  const context = [...seeds, ...graphPicks];
  const searchType = graphPicks.length > 0 ? "hybrid" : "vector";

  const contextBlock = context
    .map(
      (ch, i) =>
        `[${i + 1}] (${ch.page_title}${ch.section ? " — " + ch.section : ""})\n${ch.text}`,
    )
    .join("\n\n");

  const completion = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${c.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.1,
      messages: [
        {
          role: "system",
          content:
            "You are a clinical information assistant answering questions strictly from the provided NICE Clinical Knowledge Summaries (CKS) Hypertension excerpts. " +
            "Answer concisely and clinically, citing excerpt numbers like [1]. " +
            "If the excerpts do not contain the answer, say so plainly — never invent guidance. " +
            "This supports, never replaces, professional clinical judgement.",
        },
        { role: "user", content: `Excerpts:\n\n${contextBlock}\n\nQuestion: ${question}` },
      ],
    }),
  });
  if (!completion.ok) {
    console.error("OpenAI completion failed:", completion.status);
    return c.json({ error: "Answer generation failed" }, 502);
  }
  const answer =
    (await completion.json()).choices?.[0]?.message?.content ||
    "Unable to generate response";

  // What this number is, and what it is not.
  //
  // It is the cosine similarity between the question's embedding and the
  // single best-matching corpus chunk — i.e. "how close is the nearest thing
  // we hold to what was asked". It is a property of RETRIEVAL only. It says
  // nothing about whether the generated answer is correct, complete, or even
  // supported by the chunk it scored.
  //
  // It is emphatically NOT answer confidence, and must never be labelled as
  // such downstream. The failure mode is concrete: ask an asthma question and
  // hypertension prose still scores ~0.4 on shared clinical vocabulary, which
  // a UI showing "40% confidence" turns into a claim about the answer's
  // reliability that nothing here supports.
  //
  // `confidence` and `confidence_score` are kept byte-for-byte as they were
  // for backward compatibility with the care.engineering client; the honest
  // name is `retrieval_strength`, emitted alongside with the same value, and
  // that is the field new consumers should read.
  const topScore = seeds[0]?.score ?? 0;
  const retrievalStrength = Math.max(0, Math.min(1, Number(topScore.toFixed(2))));
  const confidence = retrievalStrength;

  return c.json({
    query_id: crypto.randomUUID(),
    answer,
    sources: context.map((ch, i) => ({
      title: `NICE CKS — ${ch.page_title}${ch.section ? ": " + ch.section : ""}`,
      url: ch.url,
      section: ch.section,
      relevance_score: Number((ch.score ?? 0).toFixed(3)),
      content: ch.text,
      excerpt: ch.text.length > 200 ? ch.text.slice(0, 200) + "..." : ch.text,
      metadata: {
        entity_name: (entities.get(ch.id) || [])[0]?.name || "",
        entity_type: (entities.get(ch.id) || [])[0]?.type || "",
        retrieval_method: neighbours.has(ch.id) ? ["graph"] : ["vector"],
        index: i + 1,
      },
    })),
    confidence,
    confidence_score: confidence,
    retrieval_strength: retrievalStrength,
    response_time: (Date.now() - started) / 1000,
    search_type: searchType,
  });
});

// --- Ingestion -----------------------------------------------------------
// Fetches the NICE CKS hypertension pages server-side. NICE geo-blocks CKS
// to UK IPs, so this must be triggered from the UK: the Worker's outbound
// fetch egresses from the Cloudflare colo serving the request. GET so it
// can be tapped from a phone browser.

async function runIngest(c) {
  const base = c.env.NICE_BASE_URL || "https://cks.nice.org.uk";
  const pages = [];
  let chunkRows = [];

  for (const path of CONTENT_PATHS) {
    const url = `${base}${path}`;
    let status = 0;
    try {
      const res = await fetch(url, {
        headers: {
          "User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
          Accept: "text/html",
        },
      });
      status = res.status;
      if (res.ok) {
        const html = await res.text();
        const { title, sections } = extractSections(html);
        const chunks = chunkSections(sections);
        chunkRows.push(
          ...chunks.map((ch) => ({ url, page_title: title, ...ch })),
        );
        pages.push({ path, status, sections: sections.length, chunks: chunks.length });
      } else {
        pages.push({ path, status, error: "fetch failed" });
      }
    } catch (err) {
      pages.push({ path, status, error: String(err).slice(0, 120) });
    }
  }

  const okPages = pages.filter((p) => !p.error).length;
  if (okPages === 0 || chunkRows.length === 0) {
    return c.json(
      {
        ok: false,
        message:
          "No pages could be fetched — NICE CKS is UK-only; trigger this from a UK connection.",
        colo: c.req.raw.cf?.colo,
        pages,
      },
      502,
    );
  }

  // Replace the corpus atomically enough for our purposes: wipe, insert,
  // bump version. Embeddings are re-added by scripts/embed-and-graph.mjs.
  await c.env.DB.batch([
    c.env.DB.prepare("DELETE FROM links"),
    c.env.DB.prepare("DELETE FROM entities"),
    c.env.DB.prepare("DELETE FROM chunks"),
  ]);
  const insert = c.env.DB.prepare(
    "INSERT INTO chunks (url, page_title, section, text) VALUES (?, ?, ?, ?)",
  );
  // D1 batches are capped; insert in slices.
  for (let i = 0; i < chunkRows.length; i += 50) {
    await c.env.DB.batch(
      chunkRows
        .slice(i, i + 50)
        .map((r) => insert.bind(r.url, r.page_title, r.section, r.text)),
    );
  }
  await c.env.DB.prepare(
    "INSERT INTO meta (k, v) VALUES ('corpus_version', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
  )
    .bind(String(Date.now()))
    .run();

  return c.json({
    ok: true,
    colo: c.req.raw.cf?.colo,
    pages_fetched: okPages,
    pages_total: CONTENT_PATHS.length,
    chunks_stored: chunkRows.length,
    next_step: "Run `npm run ingest:enrich` in worker/ to embed chunks and build the entity graph.",
    pages,
  });
}

app.get("/admin/ingest", async (c) => {
  const key = c.req.query("key") || "";
  if (!c.env.GRAPHRAG_API_KEY || !timingSafeEqual(key, c.env.GRAPHRAG_API_KEY)) {
    return c.json({ error: "Invalid API key" }, 401);
  }
  return runIngest(c);
});

app.post("/admin/ingest", apiKeyRequired(), (c) => runIngest(c));

app.notFound((c) => c.json({ error: "Not found" }, 404));

export default app;
