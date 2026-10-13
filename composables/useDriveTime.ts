// Fetch wrapper over /.netlify/functions/drive-time: one drive-time area per
// origin and time, cached for the session. Errors become a message and a null
// area, so the planner falls back to straight-line distance.

import { ref, shallowRef } from 'vue'
import type { DriveArea } from './forayDistance'

export function useDriveTime() {
  const area = shallowRef<DriveArea | null>(null)
  const busy = ref(false)
  const message = ref('')
  const cache = new Map<string, DriveArea>()
  let seq = 0

  async function load(origin: { lat: number, lon: number } | null, minutes: number) {
    const mine = ++seq
    message.value = ''
    if (!origin || !minutes) { area.value = null; busy.value = false; return }
    const key = `${origin.lat.toFixed(3)},${origin.lon.toFixed(3)},${minutes}`
    const hit = cache.get(key)
    if (hit) { area.value = hit; busy.value = false; return }
    area.value = null
    busy.value = true
    try {
      const res = await fetch(`/.netlify/functions/drive-time?lat=${origin.lat.toFixed(4)}&lon=${origin.lon.toFixed(4)}&min=${minutes}`)
      const data = await res.json().catch(() => null)
      if (mine !== seq) return
      if (!res.ok || !data?.ok || !data.geom) throw new Error(typeof data?.error === 'string' ? data.error : `status ${res.status}`)
      cache.set(key, data.geom)
      area.value = data.geom
    } catch (e: any) {
      if (mine === seq) message.value = `Drive times are unavailable right now (${e.message}), so only straight-line distance is used.`
    } finally {
      if (mine === seq) busy.value = false
    }
  }

  return { area, busy, message, load }
}
