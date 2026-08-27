// Optional in-browser (realtime) clustering of the loaded observations with
// k-means. Toggle on, pick a mode — environmental features or geographic
// location — and an adjustable k. The result is a reactive Map (uuid → cluster
// index) that the map and charts colour by, computed live with no pipeline run.

import { useObservations } from '~/composables/useObservations'

// Environmental feature vector for "features" mode. Only the dimensions actually
// present in the data are used.
const FEATURE_KEYS = [
  'ndvi', 'soil_moisture', 'prcp_d0', 'solar_exposure', 'wind_exposure',
  'water_retention', 'slope', 'aspect', 'elevation', 'tmax', 'tmin',
]

function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function dist2(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d }
  return s
}

// Deterministic k-means (k-means++ seeding, fixed seed) → array of labels.
function kmeans(vectors, k, iters = 14) {
  const n = vectors.length
  if (n === 0) return []
  const dim = vectors[0].length
  if (n <= k) return vectors.map((_, i) => i)
  const rand = mulberry32(42)
  const centroids = [vectors[Math.floor(rand() * n)].slice()]
  while (centroids.length < k) {
    const d2 = vectors.map((v) => Math.min(...centroids.map((c) => dist2(v, c))))
    const sum = d2.reduce((s, x) => s + x, 0)
    let r = rand() * sum, idx = 0
    while (r > 0 && idx < n - 1) { r -= d2[idx]; idx++ }
    centroids.push(vectors[idx].slice())
  }
  const assign = new Array(n).fill(0)
  for (let it = 0; it < iters; it++) {
    let moved = false
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity
      for (let c = 0; c < k; c++) { const d = dist2(vectors[i], centroids[c]); if (d < bd) { bd = d; best = c } }
      if (assign[i] !== best) { assign[i] = best; moved = true }
    }
    const sums = Array.from({ length: k }, () => new Array(dim).fill(0))
    const counts = new Array(k).fill(0)
    for (let i = 0; i < n; i++) { counts[assign[i]]++; const v = vectors[i]; const s = sums[assign[i]]; for (let d = 0; d < dim; d++) s[d] += v[d] }
    for (let c = 0; c < k; c++) if (counts[c]) for (let d = 0; d < dim; d++) centroids[c][d] = sums[c][d] / counts[c]
    if (!moved && it > 0) break
  }
  return assign
}

export function useLiveClusters() {
  const { data } = useObservations()
  const enabled = useState('live-cluster-enabled', () => false)
  const k = useState('live-cluster-k', () => 5)
  const mode = useState('live-cluster-mode', () => 'features') // 'features' | 'geographic'

  // uuid → cluster index. Recomputes reactively when the toggle, k, mode, or
  // dataset changes. Reading `.value` inside a chart/map computed makes that
  // computed depend on the clustering, so colours refresh automatically.
  const assignments = computed(() => {
    const map = new Map()
    if (!enabled.value) return map
    const feats = data.value?.features || []
    if (!feats.length) return map

    const kv = Math.max(2, Math.min(12, Number(k.value) || 5))
    const uuids = []
    const raw = [] // per-row numeric vectors (may contain nulls to impute)
    let dims

    if (mode.value === 'geographic') {
      dims = ['lon', 'lat']
      for (const f of feats) {
        const co = f.geometry?.coordinates
        if (!co || f.properties?.uuid == null) continue
        uuids.push(f.properties.uuid)
        raw.push([co[0], co[1]])
      }
    } else {
      const present = FEATURE_KEYS.filter((key) => feats.some((f) => {
        const v = f.properties?.[key]; return v !== null && v !== undefined && v !== ''
      }))
      if (!present.length) return map
      dims = present
      for (const f of feats) {
        if (f.properties?.uuid == null) continue
        uuids.push(f.properties.uuid)
        raw.push(present.map((key) => {
          const v = f.properties[key]; return v === null || v === undefined || v === '' ? null : Number(v)
        }))
      }
    }
    if (raw.length < kv) return map

    // Impute missing dims with the column mean, then z-normalise each dim so no
    // single feature (or longitude vs latitude) dominates the distance.
    const D = dims.length
    const mean = new Array(D).fill(0), cnt = new Array(D).fill(0)
    for (const v of raw) for (let d = 0; d < D; d++) if (Number.isFinite(v[d])) { mean[d] += v[d]; cnt[d]++ }
    for (let d = 0; d < D; d++) mean[d] = cnt[d] ? mean[d] / cnt[d] : 0
    const sd = new Array(D).fill(0)
    for (const v of raw) for (let d = 0; d < D; d++) { const x = Number.isFinite(v[d]) ? v[d] : mean[d]; sd[d] += (x - mean[d]) ** 2 }
    for (let d = 0; d < D; d++) sd[d] = Math.sqrt(sd[d] / Math.max(1, raw.length)) || 1
    const vectors = raw.map((v) => v.map((x, d) => ((Number.isFinite(x) ? x : mean[d]) - mean[d]) / sd[d]))

    const labels = kmeans(vectors, kv)
    labels.forEach((lab, i) => map.set(uuids[i], lab))
    return map
  })

  // Whether a usable live clustering exists (drives whether "Live cluster" shows
  // as a colour/group option).
  const active = computed(() => enabled.value && assignments.value.size > 0)

  // Category value for a row/properties object: "K<n>" or null.
  function labelFor(props) {
    if (!enabled.value || !props) return null
    const c = assignments.value.get(props.uuid)
    return c === undefined ? null : `K${c}`
  }

  return { enabled, k, mode, assignments, active, labelFor }
}
