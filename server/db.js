import Database from 'better-sqlite3'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url))

// Single file-based SQLite database. Set DB_PATH to override (e.g. for tests).
const dbPath = process.env.DB_PATH || join(__dirname, 'data.sqlite')

const db = new Database(dbPath)
db.pragma('journal_mode = WAL')
db.pragma('foreign_keys = ON')

db.exec(`
  CREATE TABLE IF NOT EXISTS datasets (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT,
    config            TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL DEFAULT 'pending',
    stage             TEXT NOT NULL DEFAULT 'idle',
    observation_count INTEGER DEFAULT 0,
    error_message     TEXT,
    created_at        TEXT DEFAULT (datetime('now')),
    updated_at        TEXT DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS observations (
    id            TEXT PRIMARY KEY,
    dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    inat_uuid     TEXT NOT NULL,
    taxon_name    TEXT,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    observed_on   TEXT,
    quality_grade TEXT,
    place_guess   TEXT,
    elevation     REAL,
    temp_avg      REAL,
    precip_total  REAL,
    wind_speed    REAL,
    prcp_d0       REAL,
    prcp_d1       REAL,
    prcp_d2       REAL,
    prcp_d3       REAL,
    prcp_d4       REAL,
    prcp_d5       REAL,
    prcp_d6       REAL,
    ndvi          REAL,
    soil_moisture REAL,
    land_cover    INTEGER,
    cluster       INTEGER,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE (inat_uuid, dataset_id)
  );

  CREATE INDEX IF NOT EXISTS observations_dataset_id_idx ON observations(dataset_id);
  CREATE INDEX IF NOT EXISTS observations_cluster_idx    ON observations(dataset_id, cluster);
`)

export default db
