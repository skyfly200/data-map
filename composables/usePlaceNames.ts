// Nearest place names for listed foray places, from Nominatim. One request a
// second (its usage policy), best places first, cached in localStorage so a
// returning user does not look the same places up again.

import { shallowRef } from 'vue'
import { placeKey, placeLabel, reverseUrl } from './placeName'

const STORE = 'foray-place-names'
const MAX_STORED = 500
const GAP_MS = 1100

export function usePlaceNames() {
  const names = shallowRef(new Map<string, string | null>())
  let queue: { lat: number, lon: number, key: string }[] = []
  let running = false

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE) || '{}') || {}
      names.value = new Map(Object.entries(saved))
    } catch { /* ignore */ }
  }
  function persist() {
    try {
      const entries = [...names.value.entries()].slice(-MAX_STORED)
      localStorage.setItem(STORE, JSON.stringify(Object.fromEntries(entries)))
    } catch { /* ignore */ }
  }

  const nameAt = (lat: number, lon: number) => names.value.get(placeKey(lat, lon)) ?? null

  async function run() {
    if (running) return
    running = true
    try {
      while (queue.length) {
        const p = queue.shift()!
        if (names.value.has(p.key)) continue
        let label: string | null = null
        let ok = false
        try {
          const res = await fetch(reverseUrl(p.lat, p.lon), { headers: { Accept: 'application/json' } })
          if (res.ok) { label = placeLabel(await res.json()); ok = true }
        } catch { /* offline or blocked: leave unnamed, try again next time */ }
        if (ok) {
          names.value = new Map(names.value).set(p.key, label)
          persist()
        }
        if (queue.length) await new Promise((r) => setTimeout(r, GAP_MS))
      }
    } finally {
      running = false
    }
  }

  /** Look up names for these places (in order), replacing any earlier queue. */
  function request(cells: { lat: number, lon: number }[]) {
    const seen = new Set<string>()
    queue = []
    for (const c of cells) {
      const key = placeKey(c.lat, c.lon)
      if (names.value.has(key) || seen.has(key)) continue
      seen.add(key)
      queue.push({ lat: c.lat, lon: c.lon, key })
    }
    run()
  }

  return { names, nameAt, request, restore }
}
