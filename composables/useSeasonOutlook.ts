// The season outlook for a place (WANT-19): fetches the weather from the
// season-weather function and runs seasonForecast.ts over it for every species
// with enough past seasons nearby. Errors become a message, never a throw.

import { ref } from 'vue'
import {
  MIN_FINDS_PER_YEAR, MIN_YEARS, buildScenarios, correct, currentToDate, dayOfYear, describeTrigger, fitTrigger, forecastBias, forecastSeason,
  indicators, localSeasons,
} from './seasonForecast'
import type { Bias, DaySeries, SeasonOutlook, SeasonYear, Trigger } from './seasonForecast'

/** Most past years fetched; also the most analog years per scenario set. */
export const MAX_YEARS = 12
/** Most species forecast at once. */
export const MAX_SPECIES = 10
const CONCURRENCY = 4

export interface SpeciesOutlook {
  name: string
  outlook: SeasonOutlook
  trigger: Trigger
  basis: string
  years: number
  /** The past seasons the trigger was fitted on. */
  past: SeasonYear[]
}

export interface OutlookMeta {
  /** ISO time of the WeatherNext run used, or null for climatology only. */
  init: string | null
  bias: Bias | null
  note: string
  analogYears: number[]
}

const DAY_MS = 86400000
const doyOfIso = (iso: string) => dayOfYear(new Date(`${iso}T00:00:00Z`))

/**
 * The species worth an outlook: enough past seasons, not long over, most finds
 * first. Naming `species` asks for that one alone, whatever its dates.
 */
export function outlookCandidates(
  seasons: ReturnType<typeof localSeasons>,
  today: number,
  { minYears = MIN_YEARS, limit = MAX_SPECIES, species = '' }: { minYears?: number, limit?: number, species?: string } = {},
) {
  return [...seasons.entries()]
    .filter(([name]) => !species || name === species)
    .filter(([, s]) => s.past.length >= minYears)
    // Keep seasons whose usual end has not long passed: the outlook is for the rest of this year.
    .filter(([, s]) => {
      if (species) return true
      const ends = s.past.map((y) => y.end).sort((a, b) => a - b)
      return ends[ends.length - 1] >= today - 7
    })
    .map(([name, s]) => ({ name, ...s, finds: s.past.reduce((a, y) => a + y.n, 0) }))
    .sort((a, b) => b.finds - a.finds)
    .slice(0, limit)
}

/** Species with enough past seasons near a place for an outlook, by name. */
export function outlookSpecies(seasons: ReturnType<typeof localSeasons>, { minYears = MIN_YEARS } = {}): string[] {
  return [...seasons.entries()].filter(([, s]) => s.past.length >= minYears).map(([name]) => name).sort()
}

export function useSeasonOutlook() {
  const { accessToken } = useAuth()
  const loading = ref(false)
  const message = ref('')
  const outlooks = ref<SpeciesOutlook[]>([])
  const meta = ref<OutlookMeta | null>(null)
  const progress = ref('')

  async function call(query: string) {
    const token = await accessToken()
    let res: Response
    try {
      res = await fetch(`/.netlify/functions/season-weather?${query}`, {
        headers: token ? { authorization: `Bearer ${token}` } : {},
      })
    } catch {
      throw new Error('Could not reach the server.')
    }
    const data = await res.json().catch(() => null)
    if (res.status === 401) throw new Error('Sign in to see the season outlook.')
    if (!res.ok || !data?.ok) throw new Error(typeof data?.error === 'string' ? data.error : `That request failed (${res.status}).`)
    return data
  }

  async function run(rows: any[], { lat, lon, radiusKm = 50, species = '', distance = '' }: { lat: number, lon: number, radiusKm?: number, species?: string, distance?: string }) {
    loading.value = true
    message.value = ''
    outlooks.value = []
    meta.value = null
    try {
      const now = new Date()
      const year = now.getUTCFullYear()
      const today = dayOfYear(now)
      const seasons = localSeasons(rows, { lat, lon, radiusKm, year })
      const picked = outlookCandidates(seasons, today, { species })
      if (!picked.length) {
        const within = distance || `${Math.round(radiusKm)} km`
        if (species) {
          const n = seasons.get(species)?.past.length ?? 0
          message.value = `${species} has ${n} past season${n === 1 ? '' : 's'} within ${within} with ${MIN_FINDS_PER_YEAR}+ finds; an outlook needs ${MIN_YEARS}. Try a wider distance.`
        } else {
          message.value = `No species has ${MIN_YEARS}+ past seasons within ${within} with a season still to come.`
        }
        return
      }

      const wanted = [...new Set(picked.flatMap((s) => s.past.map((y) => y.year)))]
        .sort((a, b) => b - a).slice(0, MAX_YEARS)
      const ll = `lat=${lat}&lon=${lon}`
      const past = new Map<number, DaySeries>()
      let done = 0
      progress.value = `Weather 0/${wanted.length + 2}`
      const queue = [...wanted]
      const worker = async () => {
        while (queue.length) {
          const y = queue.shift()!
          past.set(y, (await call(`${ll}&part=year&year=${y}`)).series)
          progress.value = `Weather ${++done}/${wanted.length + 2}`
        }
      }
      const [observed, forecast] = await Promise.all([
        call(`${ll}&part=year&year=${year}`).then((d) => d.series as DaySeries),
        // The forecast is the improvement, not the foundation: without it the
        // outlook still runs on past years alone.
        call(`${ll}&part=forecast`).catch((e: Error) => ({ error: e.message })),
        ...Array.from({ length: CONCURRENCY }, worker),
      ])
      progress.value = 'Forecasting…'

      let current: DaySeries = observed
      let members: any[] = []
      let bias: Bias | null = null
      let note = ''
      if (forecast && !('error' in forecast)) {
        const recentStart = Date.parse(`${forecast.recentStart}T00:00:00Z`)
        // A window reaching back over New Year starts before day 1; the
        // helpers skip those days.
        const recentDoy = Math.round((recentStart - Date.UTC(year, 0, 1)) / DAY_MS) + 1
        bias = forecastBias(observed, forecast.recent, recentDoy)
        current = currentToDate(observed, forecast.recent, recentDoy, today, bias)
        const offset = today - doyOfIso(forecast.start)
        members = (forecast.members || []).map((m: any) => {
          const fixed = correct(m, bias!)
          // A run cached from yesterday starts a day early; drop the days already past.
          return offset > 0 ? { rain: fixed.rain.slice(offset), tmax: fixed.tmax.slice(offset), tmin: fixed.tmin.slice(offset) } : fixed
        })
      } else {
        note = `Using past years only: ${(forecast as any)?.error || 'no forecast'}`
      }

      const analogYears = [...past.keys()].sort((a, b) => a - b)
      const scenarios = buildScenarios({ current, members, analogs: analogYears.map((y) => past.get(y)!), today })
        .map((s) => indicators(s))

      const out: SpeciesOutlook[] = []
      for (const s of picked) {
        const trigger = fitTrigger(s.past.filter((y) => past.has(y.year)).map((season) => ({ season, series: past.get(season.year)! })))
        if (!trigger) continue
        const fitted = s.past.filter((y) => past.has(y.year))
        out.push({
          name: s.name,
          outlook: forecastSeason(trigger, scenarios, today, s.current),
          trigger,
          basis: describeTrigger(trigger),
          years: trigger.years,
          past: fitted,
        })
      }
      const rank = { in: 0, upcoming: 1, over: 2, unlikely: 3 }
      out.sort((a, b) => rank[a.outlook.status] - rank[b.outlook.status]
        || (a.outlook.onset?.p50 ?? 999) - (b.outlook.onset?.p50 ?? 999))
      outlooks.value = out
      meta.value = { init: (forecast as any)?.init ?? null, bias, note, analogYears }
      if (!out.length) message.value = 'Not enough weather for the past seasons nearby to fit any species.'
    } catch (e: any) {
      message.value = e.message
    } finally {
      loading.value = false
      progress.value = ''
    }
  }

  return { loading, message, outlooks, meta, progress, run }
}
