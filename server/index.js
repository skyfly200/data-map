import express from 'express'
import cors from 'cors'
import { randomUUID } from 'crypto'
import db from './db.js'
import { runPipeline } from './pipeline.js'

const app = express()
app.use(cors())
app.use(express.json())

function serializeDataset(row) {
  return { ...row, config: JSON.parse(row.config) }
}

// ── Datasets ──────────────────────────────────────────────────────────────────

app.get('/api/datasets', (req, res) => {
  const rows = db.prepare('SELECT * FROM datasets ORDER BY created_at DESC').all()
  res.json(rows.map(serializeDataset))
})

app.get('/api/datasets/:id', (req, res) => {
  const row = db.prepare('SELECT * FROM datasets WHERE id = ?').get(req.params.id)
  if (!row) return res.status(404).json({ error: 'Not found' })
  res.json(serializeDataset(row))
})

app.post('/api/datasets', (req, res) => {
  const { name, description, config } = req.body
  if (!name) return res.status(400).json({ error: 'name is required' })

  const id = randomUUID()
  db.prepare(`
    INSERT INTO datasets (id, name, description, config, status, stage)
    VALUES (?, ?, ?, ?, 'pending', 'idle')
  `).run(id, name, description ?? null, JSON.stringify(config ?? {}))

  const row = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id)

  // Kick off the pipeline in the background — don't await.
  runPipeline(id)

  res.status(201).json(serializeDataset(row))
})

app.delete('/api/datasets/:id', (req, res) => {
  db.prepare('DELETE FROM datasets WHERE id = ?').run(req.params.id)
  res.status(204).end()
})

// ── Observations ────────────────────────────────────────────────────────────

app.get('/api/datasets/:id/observations', (req, res) => {
  const rows = db
    .prepare('SELECT * FROM observations WHERE dataset_id = ? ORDER BY observed_on DESC')
    .all(req.params.id)
  res.json(rows)
})

// ── Start ─────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3001
app.listen(PORT, () => console.log(`🍄 API listening on http://localhost:${PORT}`))
