// Land-access data for the foray planner (WANT-17). Talks to the access
// endpoint (netlify/functions/access.mjs) and degrades to an explicit
// "not loaded" state when it is missing, empty, or the region is not loaded.
// Callers must treat anything but status === 'loaded' as "no access info".

import { computed } from 'vue'
import { ACCESS_STATUS_MESSAGE, bboxInColorado, parseAccessResponse } from './forayPlanner'
import type { AccessArea, AccessStatus } from './forayPlanner'

export const ACCESS_ENDPOINT = '/.netlify/functions/access'

/** Fetch + parse. Never throws; every failure maps to a status. */
export async function fetchAccess(
  bbox: [number, number, number, number],
  fetchFn: typeof fetch = fetch,
): Promise<{ status: AccessStatus, areas: AccessArea[], region: string | null }> {
  if (!bboxInColorado(bbox)) return { status: 'not-loaded', areas: [], region: null }
  try {
    const res = await fetchFn(`${ACCESS_ENDPOINT}?bbox=${bbox.map((n) => n.toFixed(4)).join(',')}`)
    if (!res.ok) return { status: 'unavailable', areas: [], region: null }
    return parseAccessResponse(await res.json())
  } catch {
    return { status: 'unavailable', areas: [], region: null }
  }
}

export function useAccess() {
  const status = useState<AccessStatus>('foray-access-status', () => 'idle')
  const areas = useState<AccessArea[]>('foray-access-areas', () => [])
  const region = useState<string | null>('foray-access-region', () => null)

  async function load(bbox: [number, number, number, number]) {
    status.value = 'loading'
    const r = await fetchAccess(bbox)
    areas.value = r.areas
    region.value = r.region
    status.value = r.status
  }

  const loaded = computed(() => status.value === 'loaded')
  const message = computed(() => ACCESS_STATUS_MESSAGE[status.value])

  return { status, areas, region, loaded, message, load }
}
