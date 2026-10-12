// Fetch wrapper over /.netlify/functions/access-nearest. Caches per cell key
// (keys encode the cell size and shape), looks up best places first, and turns
// errors into a message so the planner keeps working without parking data.

import { ref, shallowRef } from 'vue'
import { keysToFetch, type NearestInfo } from './forayWalk'

export const WALK_BATCH = 1000
/** Most cells looked up per refresh; the rest stay "walk unknown" until they rank higher. */
export const WALK_MAX = 3000

export function useForayWalk() {
  const info = shallowRef(new Map<string, NearestInfo>())
  const busy = ref(false)
  const message = ref('')
  let seq = 0

  async function post(points: [number, number][]) {
    const res = await fetch('/.netlify/functions/access-nearest', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ points }),
    })
    const data = await res.json().catch(() => null)
    if (!res.ok || !data?.ok) throw new Error(typeof data?.error === 'string' ? data.error : `Parking lookup failed (${res.status}).`)
    return data.results as (NearestInfo & { i: number })[]
  }

  /** Look up the cells not yet cached, best-first. */
  async function refresh(cells: { key: string, lat: number, lon: number }[]) {
    const mine = ++seq
    const want = keysToFetch(cells, info.value, WALK_MAX)
    if (!want.length) return
    const byKey = new Map(cells.map((c) => [c.key, c]))
    busy.value = true
    message.value = ''
    try {
      for (let i = 0; i < want.length; i += WALK_BATCH) {
        const keys = want.slice(i, i + WALK_BATCH)
        const results = await post(keys.map((k) => [byKey.get(k)!.lon, byKey.get(k)!.lat]))
        if (mine !== seq) return
        const next = new Map(info.value)
        keys.forEach((k, j) => {
          const r = results[j]
          next.set(k, { parking: r?.parking ?? null, accessible: r?.accessible ?? null, trailhead: r?.trailhead ?? null, road_m: r?.road_m ?? null })
        })
        info.value = next
      }
    } catch (e: any) {
      if (mine === seq) message.value = `Parking and walk-in distances are unavailable (${e.message}).`
    } finally {
      if (mine === seq) busy.value = false
    }
  }

  return { info, busy, message, refresh }
}
