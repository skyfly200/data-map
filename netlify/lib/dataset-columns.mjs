// What a dataset carries, read from the file rather than from any record of
// how it was made. This is the one answer that holds for every dataset: an
// uploaded file that was enriched elsewhere has no job, but it has columns.
//
// No imports, so the detail page can run it in the browser over the GeoJSON
// it already downloads; the stage catalogue comes from the server.

const SKIP_COLUMNS = new Set(['images', 'dna'])

/**
 * Every property in the features, how many carry it, and which stage (if any)
 * it comes from. Grouped by stage in catalogue order, observation fields last.
 */
export function columnSummary(features = [], stages = []) {
  const stageOf = new Map()
  for (const s of stages) for (const b of s.bands || []) stageOf.set(b, s.key)
  const counts = new Map()
  for (const f of features) {
    const props = f?.properties || {}
    for (const [k, v] of Object.entries(props)) {
      if (SKIP_COLUMNS.has(k)) continue
      if (!counts.has(k)) counts.set(k, 0)
      if (v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && Number.isNaN(v))) {
        counts.set(k, counts.get(k) + 1)
      }
    }
  }
  const total = features.length
  const groups = new Map(stages.map((s) => [s.key, { key: s.key, label: s.label, columns: [] }]))
  const other = { key: 'observation', label: 'Observation fields', columns: [] }
  for (const [name, present] of counts) {
    const col = { name, present, fraction: total ? present / total : 0 }
    const key = stageOf.get(name)
    ;(key ? groups.get(key) : other).columns.push(col)
  }
  // Bands in the catalogue's own order, so prcp_d0..d6 read in sequence.
  for (const s of stages) {
    const g = groups.get(s.key)
    g.columns.sort((a, b) => s.bands.indexOf(a.name) - s.bands.indexOf(b.name))
  }
  other.columns.sort((a, b) => a.name.localeCompare(b.name))
  const out = [...groups.values()].filter((g) => g.columns.length)
  if (other.columns.length) out.push(other)
  return { total, groups: out }
}
