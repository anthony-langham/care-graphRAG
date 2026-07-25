#!/usr/bin/env node
// Post-ingestion enrichment: embed every chunk (text-embedding-3-small,
// 512 dims) and build the chunk<->entity graph with gpt-4o-mini entity
// extraction. Reads/writes the remote D1 database via wrangler, so it can
// run from any machine with OPENAI_API_KEY and CLOUDFLARE_API_TOKEN set —
// no UK connection needed (only /admin/ingest is geo-sensitive).
//
// Usage (from worker/): node scripts/embed-and-graph.mjs

import { spawnSync } from "node:child_process";
import { writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const OPENAI_KEY = (process.env.OPENAI_API_KEY || "").trim();
if (!OPENAI_KEY) {
  console.error("OPENAI_API_KEY is required");
  process.exit(1);
}

const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIMS = 512;

function d1(sqlOrFlags) {
  const args = [
    "wrangler",
    "d1",
    "execute",
    "care-graphrag",
    "--remote",
    "--json",
    ...sqlOrFlags,
  ];
  const res = spawnSync("npx", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (res.status !== 0) {
    throw new Error(`wrangler d1 failed: ${res.stderr?.slice(0, 500)}`);
  }
  // wrangler prints progress lines before the JSON payload; skip to it.
  const start = res.stdout.indexOf("[\n") >= 0 ? res.stdout.indexOf("[\n") : res.stdout.indexOf("[");
  if (start < 0) throw new Error(`no JSON in wrangler output: ${res.stdout.slice(0, 300)}`);
  return JSON.parse(res.stdout.slice(start))[0].results;
}

function d1File(sql) {
  const path = join(tmpdir(), `d1-batch-${Date.now()}.sql`);
  writeFileSync(path, sql);
  try {
    return d1(["--file", path]);
  } finally {
    rmSync(path, { force: true });
  }
}

const esc = (s) => String(s).replaceAll("'", "''");

async function openai(path, body) {
  const res = await fetch(`https://api.openai.com/v1/${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`OpenAI ${path} failed (${res.status}): ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

function toBase64(f32) {
  return Buffer.from(new Uint8Array(f32.buffer)).toString("base64");
}

// --- 1. Embeddings -------------------------------------------------------

const pending = d1(["--command", "SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id"]);
console.log(`${pending.length} chunks to embed`);

for (let i = 0; i < pending.length; i += 64) {
  const batch = pending.slice(i, i + 64);
  const { data } = await openai("embeddings", {
    model: EMBEDDING_MODEL,
    dimensions: EMBEDDING_DIMS,
    input: batch.map((c) => c.text),
  });
  const updates = batch
    .map((c, j) => {
      const vec = Float32Array.from(data[j].embedding);
      return `UPDATE chunks SET embedding = '${toBase64(vec)}' WHERE id = ${c.id};`;
    })
    .join("\n");
  d1File(updates);
  console.log(`embedded ${Math.min(i + 64, pending.length)}/${pending.length}`);
}

// --- 2. Entity graph -----------------------------------------------------

const chunks = d1(["--command", "SELECT id, section, text FROM chunks ORDER BY id"]);
const linked = new Set(d1(["--command", "SELECT DISTINCT chunk_id FROM links"]).map((r) => r.chunk_id));
const todo = chunks.filter((c) => !linked.has(c.id));
console.log(`${todo.length} chunks need entity extraction`);

const EXTRACT_PROMPT =
  "Extract the clinically meaningful entities from this NICE CKS hypertension excerpt. " +
  "Return JSON: {\"entities\": [{\"name\": string, \"type\": string}]}. " +
  "Types: drug, drug_class, condition, symptom, investigation, threshold, population, procedure, lifestyle_factor. " +
  "Use canonical lowercase names (e.g. \"ace inhibitor\", \"stage 1 hypertension\", \"ambulatory blood pressure monitoring\"). " +
  "Max 12 entities; only entities central to the excerpt.";

for (let i = 0; i < todo.length; i += 8) {
  const batch = todo.slice(i, i + 8);
  const results = await Promise.all(
    batch.map((c) =>
      openai("chat/completions", {
        model: "gpt-4o-mini",
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: EXTRACT_PROMPT },
          { role: "user", content: `Section: ${c.section}\n\n${c.text.slice(0, 4000)}` },
        ],
      }).then(
        (r) => ({ id: c.id, entities: JSON.parse(r.choices[0].message.content).entities || [] }),
        (err) => ({ id: c.id, entities: [], error: String(err).slice(0, 120) }),
      ),
    ),
  );
  const sql = results
    .flatMap(({ id, entities }) =>
      entities
        .filter((e) => e?.name && typeof e.name === "string")
        .slice(0, 12)
        .flatMap((e) => {
          const name = esc(e.name.trim().toLowerCase());
          const type = esc((e.type || "").trim().toLowerCase());
          return [
            `INSERT INTO entities (name, type) VALUES ('${name}', '${type}') ON CONFLICT(name) DO NOTHING;`,
            `INSERT OR IGNORE INTO links (chunk_id, entity_id) SELECT ${id}, id FROM entities WHERE name = '${name}';`,
          ];
        }),
    )
    .join("\n");
  if (sql) d1File(sql);
  const failed = results.filter((r) => r.error);
  if (failed.length) console.warn("extraction failures:", failed.map((f) => f.id).join(","));
  console.log(`entities ${Math.min(i + 8, todo.length)}/${todo.length}`);
}

// Bump the corpus version so Worker isolates reload their cache.
d1(["--command", `INSERT INTO meta (k, v) VALUES ('corpus_version', '${Date.now()}') ON CONFLICT(k) DO UPDATE SET v = excluded.v`]);

const stats = d1([
  "--command",
  "SELECT (SELECT COUNT(*) FROM chunks) AS chunks, (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) AS embedded, (SELECT COUNT(*) FROM entities) AS entities, (SELECT COUNT(*) FROM links) AS links",
]);
console.log("done:", JSON.stringify(stats[0]));
