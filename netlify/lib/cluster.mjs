// Lightweight k-means clustering in Node (no deps), for the on-demand
// fetch-species flow. Real environmental clustering is the Python pipeline's
// job (KMeans over enriched rasters); freshly-fetched species have no
// enrichment yet, so this clusters on whatever numeric dimensions are present
// (day-of-year, coordinates, and any enriched fields that happen to exist),
// giving meaningful spatial/temporal groups until the full pipeline re-runs.

const DIMS = ['day_of_year', 'elevation', 'tmax', 'tmin', 'lat', 'lng']

function num(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length }
function std(a, m) { return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) }
function dist2(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d } return s }

export function kmeans(vectors, k, iterations = 30) {
  const n = vectors.length
  if (!n) return []
  k = Math.max(1, Math.min(k, n))
  const dimN = vectors[0].length

  // Farthest-point seeding (deterministic k-means++ variant): start from the
  // first point, then repeatedly add the point farthest from the chosen set.
  // This spreads seeds across the data instead of clumping on nearby early
  // points, which naive index-stepping does for tightly grouped observations.
  const seeds = [0]
  while (seeds.length < k) {
    let far = 0, fd = -1
    for (let i = 0; i < n; i++) {
      let nearest = Infinity
      for (const s of seeds) { const d = dist2(vectors[i], vectors[s]); if (d < nearest) nearest = d }
      if (nearest > fd) { fd = nearest; far = i }
    }
    if (seeds.includes(far)) break
    seeds.push(far)
  }
  let centroids = seeds.map((i) => [...vectors[i]])
  k = centroids.length
  const assign = new Array(n).fill(0)

  for (let it = 0; it < iterations; it++) {
    let changed = false
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity
      for (let c = 0; c < k; c++) { const d = dist2(vectors[i], centroids[c]); if (d < bd) { bd = d; best = c } }
      if (assign[i] !== best) { assign[i] = best; changed = true }
    }
    const sums = Array.from({ length: k }, () => new Array(dimN).fill(0))
    const counts = new Array(k).fill(0)
    for (let i = 0; i < n; i++) { counts[assign[i]]++; const v = vectors[i]; for (let d = 0; d < dimN; d++) sums[assign[i]][d] += v[d] }
    for (let c = 0; c < k; c++) if (counts[c]) centroids[c] = sums[c].map((s) => s / counts[c])
    if (!changed) break
  }
  return assign
}

// Assign a `cluster` id to each GeoJSON feature (in place-safe: returns copies).
export function clusterFeatures(features, k = 4) {
  const feats = features || []
  if (feats.length < 2) return feats.map((f) => ({ ...f, properties: { ...f.properties, cluster: feats.length ? 0 : null } }))

  const raw = feats.map((f) => {
    const p = f.properties || {}
    const coords = f.geometry?.coordinates || [null, null]
    return {
      day_of_year: num(p.day_of_year), elevation: num(p.elevation),
      tmax: num(p.tmax), tmin: num(p.tmin), lat: num(coords[1]), lng: num(coords[0]),
    }
  })
  const cols = DIMS.filter((c) => raw.some((r) => r[c] !== null))
  if (!cols.length) return feats.map((f) => ({ ...f, properties: { ...f.properties, cluster: null } }))

  const stats = {}
  for (const c of cols) {
    const vals = raw.map((r) => r[c]).filter((v) => v !== null)
    const m = mean(vals)
    stats[c] = { m, sd: std(vals, m) || 1 }
  }
  const vectors = raw.map((r) => cols.map((c) => (r[c] === null ? 0 : (r[c] - stats[c].m) / stats[c].sd)))
  const assign = kmeans(vectors, Math.min(k, feats.length))
  return feats.map((f, i) => ({ ...f, properties: { ...f.properties, cluster: assign[i] } }))
}
