-- D1 schema for the care-graphrag Worker (replaces MongoDB Atlas ckshtn).
-- chunks.embedding is a base64-encoded Float32Array (512 dims,
-- text-embedding-3-small) written by scripts/embed-and-graph.mjs.

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL,
  page_title TEXT NOT NULL DEFAULT '',
  section TEXT NOT NULL DEFAULT '',
  text TEXT NOT NULL,
  embedding TEXT
);

CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  type TEXT NOT NULL DEFAULT ''
);

-- chunk <-> entity membership; graph expansion joins chunks that share entities.
CREATE TABLE IF NOT EXISTS links (
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  PRIMARY KEY (chunk_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_links_entity ON links (entity_id);

CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);
