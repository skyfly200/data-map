// Which enrichment steps a member runs by default.
//
// The catalogue is ee-pipeline.mjs (core steps plus steps drawn from the map's
// layers); this is only the member's choice from it. Stored under one settings
// key so it rides the account sync with the other preferences, and read back
// through cleanStages so a step renamed or removed since it was saved drops out
// rather than failing the next job.

import { STAGES, STAGE_GROUPS, DEFAULT_STAGES, cleanStages } from '../netlify/lib/ee-pipeline.mjs'

const KEY = 'enrich-steps'

export interface StepGroup {
  group: string
  steps: Array<{ key: string, label: string, description: string, fromMap: boolean }>
}

/** The catalogue grouped for a picker, in STAGE_GROUPS order. */
export function stepGroups(): StepGroup[] {
  const entries = Object.entries(STAGES).map(([key, s]: [string, any]) => ({
    key, label: s.label, description: s.description, fromMap: Boolean(s.layer), group: s.group || 'Other',
  }))
  const order = [...STAGE_GROUPS, ...new Set(entries.map((e) => e.group).filter((g) => !STAGE_GROUPS.includes(g)))]
  return order
    .map((group) => ({ group, steps: entries.filter((e) => e.group === group) }))
    .filter((g) => g.steps.length)
}

/** A saved choice, or the standard set when there is none worth using. */
export function resolveDefault(saved: unknown): string[] {
  const clean = cleanStages(saved as any)
  return clean.length ? clean : [...DEFAULT_STAGES]
}

/** Same steps, whatever order they were ticked in. */
export function sameSteps(a: string[] = [], b: string[] = []) {
  const x = cleanStages(a)
  const y = cleanStages(b)
  return x.length === y.length && x.every((k, i) => k === y[i])
}

export function useEnrichSteps() {
  const saved = useState<string[] | null>(KEY, () => null)
  const cloud = safeCloudSync()

  function load() {
    if (!import.meta.client) return
    try {
      const raw = JSON.parse(localStorage.getItem(KEY) || 'null')
      saved.value = Array.isArray(raw) ? cleanStages(raw) : null
    } catch {
      saved.value = null
    }
  }

  function persist() {
    if (!import.meta.client) return
    try {
      if (saved.value) localStorage.setItem(KEY, JSON.stringify(saved.value))
      else localStorage.removeItem(KEY)
      cloud?.schedulePush()
    } catch { /* quota or private mode */ }
  }

  load()

  /** The member's default set, or the standard one. A fresh copy to edit. */
  const myDefault = computed(() => resolveDefault(saved.value))
  const isCustom = computed(() => Boolean(saved.value?.length) && !sameSteps(saved.value!, DEFAULT_STAGES))

  function saveDefault(list: string[]) {
    const clean = cleanStages(list)
    saved.value = clean.length && !sameSteps(clean, DEFAULT_STAGES) ? clean : null
    persist()
  }

  function resetDefault() {
    saved.value = null
    persist()
  }

  return {
    groups: stepGroups(),
    standard: DEFAULT_STAGES as string[],
    myDefault,
    isCustom,
    load,
    saveDefault,
    resetDefault,
  }
}
