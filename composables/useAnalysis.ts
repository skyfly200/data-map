// Statistics behind the Analysis page.
//
// Everything here runs in the browser over the filtered observation set, so the
// same filters that drive the map and charts drive these too — an analysis of
// "Colorado, September, ≥25 records" is one filter change away.
//
// Each function is a single pass, or a pass plus a small pass over the summary,
// because the dataset is ~48k rows and a quadratic here is a frozen tab (which
// is exactly how the Charts page used to behave).
//
// A caveat worth carrying into every reading below: iNaturalist records are
// opportunistic observations, not surveys. These describe what people recorded,
// which correlates with — but is not — what grows where.

import { computed } from 'vue'
import { hasValue } from '~/composables/useObservations'
import {
  ANALYSIS_FIELDS, fieldValue, spearman, meanSd, median,
} from '~/composables/statistics'

export { ANALYSIS_FIELDS, fieldValue, spearman, meanSd, median }

export interface AnalysisField extends typeof ANALYSIS_FIELDS[number] {
  n: number
}

export interface CorrelationPair {
  a: string
  b: string
  rho: number
  n: number
}

export interface CorrelationResult {
  fields: AnalysisField[]
  matrix: (number | null)[][]
  pairs: CorrelationPair[]
}

export interface SpeciesProfile {
  species: string
  n: number
  z: (number | null)[]
  means: (number | null)[]
}

export interface SpeciesProfilesResult {
  fields: AnalysisField[]
  species: SpeciesProfile[]
}

export interface CoOccurrenceResult {
  a: string
  b: string
  cells: number
  lift: number
}

export interface YearSummary {
  year: number
  n: number
  medianDoy: number
  medianElevation: number
  species: number
}

export interface FieldCoverage {
  key: string
  label: string
  filled: number
  total: number
  pct: number
}

export interface YearCoverage {
  year: number
  n: number
  pct: Record<string, number>
}

export interface CoverageResult {
  total: number
  fields: FieldCoverage[]
  years: YearCoverage[]
}

export function useAnalysis() {
  const { rows } = useObservations()

  /** Fields that actually carry data here — the rest would be empty columns. */
  const presentFields = computed((): AnalysisField[] => {
    const counts = new Map<string, number>(ANALYSIS_FIELDS.map((f) => [f.key, 0]))
    for (const r of rows.value) {
      for (const f of ANALYSIS_FIELDS) {
        if (fieldValue(r, f.key) !== null) counts.set(f.key, counts.get(f.key) + 1)
      }
    }
    // A handful of values cannot support a correlation; 30 is a low bar that
    // still keeps a nearly-empty column out of the matrix.
    return ANALYSIS_FIELDS.filter((f) => (counts.get(f.key) || 0) >= 30)
      .map((f) => ({ ...f, n: counts.get(f.key) || 0 }))
  })

  /**
   * Spearman correlation between every pair of populated fields.
   *
   * Pairs are PAIRWISE-complete: each cell uses only the rows where both fields
   * are present. With coverage this uneven (soil moisture is on ~23% of rows),
   * dropping any row missing any field would throw away most of the data and
   * silently compute the matrix on an unrepresentative remainder.
   */
  const correlationMatrix = computed((): CorrelationResult => {
    const fields = presentFields.value
    if (fields.length < 2) return { fields: [], matrix: [], pairs: [] }

    const columns = fields.map((f) => rows.value.map((r) => fieldValue(r, f.key)))
    const matrix: (number | null)[][] = fields.map(() => new Array(fields.length).fill(null))
    const pairs: CorrelationPair[] = []

    for (let i = 0; i < fields.length; i++) {
      matrix[i][i] = 1
      for (let j = i + 1; j < fields.length; j++) {
        const xs: number[] = []
        const ys: number[] = []
        for (let k = 0; k < columns[i].length; k++) {
          const a = columns[i][k]
          const b = columns[j][k]
          if (a !== null && b !== null) { xs.push(a as number); ys.push(b as number) }
        }
        const rho = xs.length >= 30 ? spearman(xs, ys) : null
        matrix[i][j] = rho
        matrix[j][i] = rho
        if (rho !== null) {
          pairs.push({ a: fields[i].label, b: fields[j].label, rho, n: xs.length })
        }
      }
    }
    pairs.sort((p, q) => Math.abs(q.rho) - Math.abs(p.rho))
    return { fields, matrix, pairs }
  })

  /**
   * How each species differs from the dataset as a whole, per field, in
   * standard deviations.
   *
   * A z-score answers "is this species found higher/wetter/later than average",
   * which is the question a fingerprint is for. Raw means cannot be compared
   * across fields with different units.
   */
  function speciesProfiles(minObs = 20, limit = 25): SpeciesProfilesResult {
    const fields = presentFields.value
    if (!fields.length) return { fields, species: [] }

    // Baseline over everything, then per-species sums in the same pass.
    const totals = fields.map(() => ({ sum: 0, n: 0 }))
    const perSpecies = new Map<string, { n: number, f: { sum: number, n: number }[] }>()
    for (const r of rows.value) {
      const sp = r.species
      let acc = null
      if (hasValue(sp)) {
        const sName = String(sp)
        acc = perSpecies.get(sName)
        if (!acc) perSpecies.set(sName, acc = { n: 0, f: fields.map(() => ({ sum: 0, n: 0 })) })
        acc.n += 1
      }
      for (let i = 0; i < fields.length; i++) {
        const v = fieldValue(r, fields[i].key)
        if (v === null) continue
        totals[i].sum += v
        totals[i].n += 1
        if (acc) { acc.f[i].sum += v; acc.f[i].n += 1 }
      }
    }

    // Population SD needs a second pass; only for the fields being reported.
    const baseline = fields.map((f, i) => {
      const mean = totals[i].n ? totals[i].sum / totals[i].n : null
      if (mean === null) return { mean: null, sd: null }
      let ss = 0
      let n = 0
      for (const r of rows.value) {
        const v = fieldValue(r, f.key)
        if (v === null) continue
        ss += (v - mean) ** 2
        n += 1
      }
      return { mean, sd: n ? Math.sqrt(ss / n) : null }
    })

    const species = [...perSpecies.entries()]
      .filter(([, a]) => a.n >= minObs)
      .sort((a, b) => b[1].n - a[1].n)
      .slice(0, limit)
      .map(([name, a]) => ({
        species: name,
        n: a.n,
        z: fields.map((f, i) => {
          const { mean, sd } = baseline[i]
          if (!a.f[i].n || mean === null || !sd) return null
          return (a.f[i].sum / a.f[i].n - mean) / sd
        }),
        means: fields.map((f, i) => (a.f[i].n ? a.f[i].sum / a.f[i].n : null)),
      }))
    return { fields, species }
  }

  /**
   * Species recorded in the same place at the same time of year.
   *
   * Scored with LIFT — how much more often a pair co-occurs than their
   * individual frequencies would predict — rather than a raw count, which would
   * just rank the two commonest species top regardless of any relationship.
   * Lift > 1 means they turn up together more than chance.
   */
  function coOccurrence({ cell = 0.05, dayWindow = 30, minObs = 20, limit = 20 } = {}): CoOccurrenceResult[] {
    const cells = new Map<string, Set<string>>()
    const speciesCount = new Map<string, number>()

    for (const r of rows.value) {
      const { species, lat, lon } = r
      if (!hasValue(species) || !Number.isFinite(lat) || !Number.isFinite(lon)) continue
      const sName = String(species)
      speciesCount.set(sName, (speciesCount.get(sName) || 0) + 1)
      // Bucket by place AND season: two species in one valley six months apart
      // are not co-occurring in any useful sense.
      const doy = Number(r.day_of_year)
      const season = Number.isFinite(doy) ? Math.floor(doy / dayWindow) : 'x'
      const key = `${Math.floor(lat / cell)}:${Math.floor(lon / cell)}:${season}`
      let set = cells.get(key)
      if (!set) cells.set(key, set = new Set())
      set.add(sName)
    }

    const common = new Set([...speciesCount.entries()]
      .filter(([, n]) => n >= minObs).map(([s]) => s))
    if (common.size < 2) return []

    const together = new Map<string, number>()
    const present = new Map<string, number>()
    let cellTotal = 0
    for (const set of cells.values()) {
      const list = [...set].filter((s) => common.has(s))
      if (!list.length) continue
      cellTotal += 1
      for (const s of list) present.set(s, (present.get(s) || 0) + 1)
      list.sort()
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
          const k = `${list[i]} ${list[j]}`
          together.set(k, (together.get(k) || 0) + 1)
        }
      }
    }
    if (!cellTotal) return []

    const out: CoOccurrenceResult[] = []
    for (const [k, n] of together) {
      if (n < 3) continue          // one shared cell is a coincidence
      const [a, b] = k.split(' ')
      const pa = present.get(a) / cellTotal
      const pb = present.get(b) / cellTotal
      const lift = (n / cellTotal) / (pa * pb)
      out.push({ a, b, cells: n, lift })
    }
    return out.sort((x, y) => y.lift - x.lift).slice(0, limit)
  }

  /**
   * Season timing and elevation, year by year.
   *
   * Median rather than mean day-of-year: a few winter records would drag a mean
   * badly, and the median is what "the middle of the season" actually means.
   * `n` is reported alongside because a year with 30 records says much less
   * than one with 5,000, and the shape of the trend depends on which is which.
   */
  const byYear = computed((): YearSummary[] => {
    const years = new Map<number, { year: number, doy: number[], elevation: number[], species: Set<string> }>()
    for (const r of rows.value) {
      const y = Number(r.year)
      if (!Number.isFinite(y)) continue
      let acc = years.get(y)
      if (!acc) years.set(y, acc = { year: y, doy: [], elevation: [], species: new Set() })
      const doy = fieldValue(r, 'day_of_year')
      if (doy !== null) acc.doy.push(doy as number)
      const el = fieldValue(r, 'elevation')
      if (el !== null) acc.elevation.push(el as number)
      if (hasValue(r.species)) acc.species.add(String(r.species))
    }
    return [...years.values()]
      .map((a) => ({
        year: a.year,
        n: a.doy.length,
        medianDoy: median(a.doy),
        medianElevation: median(a.elevation),
        species: a.species.size,
      }))
      .filter((a) => a.n >= 10)
      .sort((a, b) => a.year - b.year)
  })

  /**
   * Field coverage, overall and by year.
   *
   * The gap this closes: a chart drawn from a 23%-covered column looks exactly
   * as confident as one drawn from a 100%-covered column. This says which is
   * which, so a reader knows how much weight a conclusion can carry.
   */
  const coverage = computed((): CoverageResult => {
    const total = rows.value.length
    const counts = new Map<string, number>(ANALYSIS_FIELDS.map((f) => [f.key, 0]))
    const byYearCounts = new Map<number, { year: number, n: number, f: Map<string, number> }>()

    for (const r of rows.value) {
      const y = Number(r.year)
      let yearAcc = null
      if (Number.isFinite(y)) {
        yearAcc = byYearCounts.get(y)
        if (!yearAcc) byYearCounts.set(y, yearAcc = { year: y, n: 0, f: new Map() })
        yearAcc.n += 1
      }
      for (const f of ANALYSIS_FIELDS) {
        if (fieldValue(r, f.key) === null) continue
        counts.set(f.key, (counts.get(f.key) || 0) + 1)
        if (yearAcc) yearAcc.f.set(f.key, (yearAcc.f.get(f.key) || 0) + 1)
      }
    }

    const fields: FieldCoverage[] = ANALYSIS_FIELDS.map((f) => ({
      ...f,
      filled: counts.get(f.key) || 0,
      total,
      pct: total ? (counts.get(f.key) || 0) / total : 0,
    })).sort((a, b) => b.pct - a.pct)

    const years: YearCoverage[] = [...byYearCounts.values()]
      .filter((a) => a.n >= 10)
      .sort((a, b) => a.year - b.year)
      .map((a) => ({
        year: a.year,
        n: a.n,
        pct: Object.fromEntries(ANALYSIS_FIELDS.map((f) => [f.key, a.n ? (a.f.get(f.key) || 0) / a.n : 0])),
      }))

    return { total, fields, years }
  })

  return {
    ANALYSIS_FIELDS, presentFields,
    correlationMatrix, speciesProfiles, coOccurrence, byYear, coverage,
    fieldValue, spearman, meanSd, median,
  }
}
