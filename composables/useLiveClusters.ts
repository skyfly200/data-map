// Optional in-browser (realtime) clustering of the loaded observations with
// k-means. Toggle on, pick what to cluster on — environmental features,
// geographic location, or both (weighted) — choose which features, and set k.
// The result is a reactive Map (uuid → cluster index) that the map and charts
// color by, computed live with no pipeline run.

import { useObservations } from '~/composables/useObservations'

interface ClusterFeature {
  key: string
  label: string
}

// Candidate environmental features, with labels for the picker.
export const CLUSTER_FEATURES: ClusterFeature[] = [
  { key: 'ndvi', label: 'NDVI' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'prcp_d0', label: 'Precipitation' },
  { key: 'solar_exposure', label: 'Solar exposure' },
  { key: 'wind_exposure', label: 'Wind exposure' },
  { key: 'water_retention', label: 'Wetness (TWI)' },
  { key: 'slope', label: 'Slope' },
  { key: 'aspect', label: 'Aspect' },
  { key: 'elevation', label: 'Elevation' },
  { key: 'tmax', label: 'High temp' },
  { key: 'tmin', label: 'Low temp' },
]
const FEATURE_KEYS = CLUSTER_FEATURES.map((f) => f.key)

function mulberry32(seed: number) {
  let a = seed >>> 0
  return function (): number {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function dist2(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d }
  return s
}

// Deterministic k-means (k-means++ seeding, fixed seed) → array of labels.
function kmeans(vectors: number[][], k: number, iters = 14): number[] {
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

const has = (v: any) => v !== null && v !== undefined && v !== ''

// z-normalise each column of `raw` (nulls imputed with the column mean).
function normalize(raw: (number | null)[][], D: number): number[][] {
  const mean = new Array(D).fill(0), cnt = new Array(D).fill(0)
  for (const v of raw) for (let d = 0; d < D; d++) if (Number.isFinite(v[d] as number)) { mean[d] += (v[d] as number); cnt[d]++ }
  for (let d = 0; d < D; d++) mean[d] = cnt[d] ? mean[d] / cnt[d] : 0
  const sd = new Array(D).fill(0)
  for (const v of raw) for (let d = 0; d < D; d++) { const x = Number.isFinite(v[d] as number) ? (v[d] as number) : mean[d]; sd[d] += (x - mean[d]) ** 2 }
  for (let d = 0; d < D; d++) sd[d] = Math.sqrt(sd[d] / Math.max(1, raw.length)) || 1
  return raw.map((v) => v.map((x, d) => ((Number.isFinite(x as number) ? (x as number) : mean[d]) - mean[d]) / sd[d]))
}

interface ClusterSize {
  label: string
  n: number
}

export function useLiveClusters() {
  const { data } = useObservations()
  const enabled = useState('live-cluster-enabled', () => false)
  const k = useState('live-cluster-k', () => 5)
  const mode = useState<'features' | 'geographic' | 'both'>('live-cluster-mode', () => 'features')
  // Selected feature keys (empty ⇒ use all present). geoWeight ∈ [0,1] balances
  // location vs features in 'both' mode.
  const features = useState<string[]>('live-cluster-features', () => [...FEATURE_KEYS])
  const geoWeight = useState('live-cluster-geoweight', () => 0.5)

  // Which candidate features actually have data in the current dataset.
  const presentFeatures = computed(() => {
    const feats = data.value?.features || []
    return CLUSTER_FEATURES.filter((f) => feats.some((ft) => has(ft.properties?.[f.key])))
  })

  const assignments = computed(() => {
    const map = new Map<string, number>()
    if (!enabled.value) return map
    const feats = data.value?.features || []
    if (!feats.length) return map
    const kv = Math.max(2, Math.min(16, Number(k.value) || 5))

    const usePresent = presentFeatures.value.map((f) => f.key)
    const chosen = (features.value || []).filter((key) => usePresent.includes(key))
    const featKeys = chosen.length ? chosen : usePresent
    const useFeatures = mode.value !== 'geographic' && featKeys.length > 0
    const useGeo = mode.value !== 'features'

    const uuids: string[] = [], featRaw: (number | null)[][] = [], geoRaw: number[][] = []
    for (const f of feats) {
      if (f.properties?.uuid == null) continue
      const co = f.geometry?.coordinates
      if (useGeo && (!co || co.length < 2)) continue
      uuids.push(f.properties.uuid as string)
      if (useFeatures) featRaw.push(featKeys.map((key) => { const v = f.properties[key]; return has(v) ? Number(v) : null }))
      if (useGeo && co) geoRaw.push([co[0] as number, co[1] as number])
    }
    if (uuids.length < kv) return map

    // Combine the (independently normalised) feature and geo blocks. In "both"
    // mode each block is scaled so its total influence tracks the weight, and
    // features are divided by sqrt(#features) so a long vector doesn't dominate.
    const w = Math.max(0, Math.min(1, Number(geoWeight.value)))
    const fz = useFeatures ? normalize(featRaw, featKeys.length) : null
    const gz = useGeo ? normalize(geoRaw, 2) : null
    const fScale = mode.value === 'both' ? (1 - w) / Math.sqrt(featKeys.length || 1) : 1
    const gScale = mode.value === 'both' ? w / Math.SQRT2 : 1

    const vectors = uuids.map((_, i) => {
      const v = []
      if (fz) for (const x of fz[i]) v.push(x * fScale)
      if (gz) for (const x of gz[i]) v.push(x * gScale)
      return v
    })

    const labels = kmeans(vectors, kv)
    labels.forEach((lab, i) => map.set(uuids[i], lab))
    return map
  })

  const active = computed(() => enabled.value && assignments.value.size > 0)

  // Cluster sizes, largest first: [{ label: 'K0', n }].
  const sizes = computed((): ClusterSize[] => {
    const counts = new Map<number, number>()
    for (const c of assignments.value.values()) counts.set(c, (counts.get(c) || 0) + 1)
    return [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([c, n]) => ({ label: `K${c}`, n }))
  })

  function labelFor(props: any) {
    if (!enabled.value || !props) return null
    const c = assignments.value.get(props.uuid)
    return c === undefined ? null : `K${c}`
  }

  return { enabled, k, mode, features, geoWeight, presentFeatures, assignments, active, sizes, labelFor }
}
