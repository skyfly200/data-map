// "What is fruiting around this time of year", shared by the dashboard's Now
// Fruiting card and the foray planner (WANT-17). Pure functions plus a thin
// composable; nothing here touches Nuxt state so tests import it directly.

import { computed, unref } from 'vue'
import type { Ref } from 'vue'

export const IN_SEASON_WINDOW = 30
export const IN_SEASON_TOP_N = 8

export interface InSeasonSpecies {
  name: string
  count: number
  dist: number
  iqr: number
  score: number
  peakLabel: string
  elevBand: { loM: number, hiM: number } | null
  hasModel: boolean
}

export function inSeasonDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0)
  return Math.floor((date.getTime() - start.getTime()) / 86400000)
}

/** Circular distance in days between two days of the year. */
export function inSeasonDist(a: number, b: number): number {
  const d = Math.abs(a - b)
  return d > 182 ? 365 - d : d
}

/** Day of year for one observation row: its date, else mid-month, else null. */
export function inSeasonObsDoy(r: any): number | null {
  if (r.date) {
    const d = new Date(r.date)
    if (!isNaN(d.getTime())) return inSeasonDayOfYear(d)
  }
  const month = r.month ?? null
  if (month) return inSeasonDayOfYear(new Date(2001, month - 1, 15))
  return null
}

export interface InSeasonOptions {
  /** Day of year to centre on. Default: today. */
  day?: number
  /** Half-width in days of the in-season window. */
  window?: number
  /** Max species returned; Infinity/0 for all. */
  topN?: number
  /** Species names that have a model. */
  modelSpecies?: Set<string>
}

/**
 * Species with finds within `window` days of `day`, ranked by how close their
 * median find is plus half their fruiting-window width (lower is better).
 */
export function inSeasonSpecies(rows: any[], opts: InSeasonOptions = {}): InSeasonSpecies[] {
  const day = opts.day ?? inSeasonDayOfYear(new Date())
  const window = opts.window ?? IN_SEASON_WINDOW
  const topN = opts.topN ?? IN_SEASON_TOP_N
  const modelSpecies = opts.modelSpecies ?? new Set<string>()

  const buckets = new Map<string, { count: number, doys: number[], elevs: number[] }>()
  for (const r of rows || []) {
    if (!r.species) continue
    const doy = inSeasonObsDoy(r)
    if (doy === null || inSeasonDist(doy, day) > window) continue
    if (!buckets.has(r.species)) buckets.set(r.species, { count: 0, doys: [], elevs: [] })
    const b = buckets.get(r.species)!
    b.count++
    b.doys.push(doy)
    const elev = Number(r.elevation)
    if (Number.isFinite(elev)) b.elevs.push(elev)
  }

  const out = [...buckets.entries()]
    .map(([name, { count, doys, elevs }]) => {
      const sorted = [...doys].sort((a, b) => a - b)
      const n = sorted.length
      const mid = Math.floor(n / 2)
      const medianDoy = n % 2 === 0
        ? Math.round((sorted[mid - 1] + sorted[mid]) / 2)
        : sorted[mid]
      const q1 = sorted[Math.floor(n * 0.25)]
      const q3 = sorted[Math.floor(n * 0.75)]
      const iqr = q3 - q1
      const dist = inSeasonDist(medianDoy, day)
      const score = dist + iqr * 0.5
      const peakLabel = dist === 0 ? 'today' : `±${dist}d`
      let elevBand = null
      if (elevs.length >= 3) {
        const es = [...elevs].sort((a, b) => a - b)
        elevBand = { loM: es[Math.floor(es.length * 0.25)], hiM: es[Math.floor(es.length * 0.75)] }
      }
      return { name, count, dist, iqr, score, peakLabel, elevBand, hasModel: modelSpecies.has(name) }
    })
    .sort((a, b) => a.score - b.score || b.count - a.count)
  return topN && Number.isFinite(topN) ? out.slice(0, topN) : out
}

/** Species names that have at least one model. */
export function modelSpeciesOf(models: any[]): Set<string> {
  const s = new Set<string>()
  for (const m of models || []) {
    if (m.species) s.add(m.species)
    if (m.target_species) s.add(m.target_species)
  }
  return s
}

/** Reactive wrapper. `rows` and `models` may be refs or plain arrays. */
export function useInSeason(
  rows: Ref<any[]> | any[],
  models: Ref<any[]> | any[],
  opts: { day?: Ref<number> | number, window?: number, topN?: number } = {},
) {
  const modelSpecies = computed(() => modelSpeciesOf(unref(models) || []))
  const species = computed(() => inSeasonSpecies(unref(rows) || [], {
    day: unref(opts.day), window: opts.window, topN: opts.topN, modelSpecies: modelSpecies.value,
  }))
  return { species, modelSpecies }
}
