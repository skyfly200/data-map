// A member's named sets of allowed areas, private or shared with a club.
// Thin wrapper over /.netlify/functions/access-sets. The endpoint may not be
// deployed yet: that is reported as `unavailable`, not as an error.

import { parseAreaGeoJSON, validateAreaInput, type AreaInput } from './areaGeometry'

export interface AccessSet {
  id: string
  name: string
  scope: 'user' | 'club'
  club_id: string | null
  club_name: string | null
  area_count: number
  role: string
}
export interface AreaFeature {
  type: 'Feature'
  id?: string
  geometry: any
  properties: { id: string, name: string, fee_status: string, collecting: string, notes: string }
}

/** The one place that knows the shape of a set-detail response. */
export function parseSetDetail(data: any): { set: AccessSet | null, areas: AreaFeature[] } {
  const feats = Array.isArray(data?.areas?.features) ? data.areas.features : []
  const areas = feats.map((f: any): AreaFeature => {
    const p = f.properties || {}
    return {
      type: 'Feature', id: f.id ?? p.id, geometry: f.geometry,
      properties: {
        id: String(p.id ?? f.id ?? ''), name: p.name || 'Unnamed area',
        fee_status: p.fee_status || 'unknown', collecting: p.collecting || 'unknown', notes: p.notes || '',
      },
    }
  })
  return { set: data?.set ?? null, areas }
}

export function useAccessSets() {
  const { accessToken } = useAuth()
  const sets = useState<AccessSet[]>('access-sets', () => [])
  const areas = useState<AreaFeature[]>('access-set-areas', () => [])
  const current = useState<AccessSet | null>('access-set-current', () => null)
  const loading = useState<boolean>('access-sets-loading', () => false)
  const error = useState<string>('access-sets-error', () => '')
  const unavailable = useState<boolean>('access-sets-unavailable', () => false)

  async function call<T = any>(query = '', body?: object): Promise<T> {
    const token = await accessToken()
    let res: Response
    try {
      res = await fetch(`/.netlify/functions/access-sets${query}`, {
        method: body ? 'POST' : 'GET',
        headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
        ...(body ? { body: JSON.stringify(body) } : {}),
      })
    } catch {
      throw new Error('Could not reach the server.')
    }
    const data = await res.json().catch(() => null)
    if (res.status === 404 && !data?.ok) { unavailable.value = true; throw new Error('Allowed areas are not available yet.') }
    if (!res.ok || !data?.ok) throw new Error(data?.error || `That request failed (${res.status}).`)
    unavailable.value = false
    return data as T
  }

  async function guarded<T>(fn: () => Promise<T>): Promise<T> {
    error.value = ''
    try { return await fn() } catch (e: any) { error.value = e.message; throw e }
  }

  async function refresh(): Promise<AccessSet[]> {
    loading.value = true
    error.value = ''
    try { sets.value = (await call<{ sets: AccessSet[] }>()).sets || [] }
    catch (e: any) { if (!unavailable.value) error.value = e.message; sets.value = [] }
    finally { loading.value = false }
    return sets.value
  }

  async function open(setId: string) {
    return guarded(async () => {
      const d = parseSetDetail(await call(`?set_id=${encodeURIComponent(setId)}`))
      areas.value = d.areas
      current.value = d.set || sets.value.find((s) => s.id === setId) || null
      return d
    })
  }

  const mutate = (body: object, reopen?: string) => guarded(async () => {
    const d = await call('', body)
    await refresh()
    if (reopen) await open(reopen)
    return d
  })

  const createSet = (name: string, scope: 'user' | 'club', clubId?: string) => {
    const n = name.trim()
    if (!n) return Promise.reject(new Error('Give the set a name.'))
    if (scope === 'club' && !clubId) return Promise.reject(new Error('Choose a club.'))
    return mutate({ action: 'create_set', name: n, scope, ...(scope === 'club' ? { club_id: clubId } : {}) })
  }
  const renameSet = (id: string, name: string) => mutate({ action: 'update_set', set_id: id, id, name: name.trim() })
  const deleteSet = (id: string) => mutate({ action: 'delete_set', set_id: id, id })
  const addArea = (setId: string, a: Partial<AreaInput>) =>
    guarded(async () => mutate({ action: 'add_area', set_id: setId, ...validateAreaInput(a) }, setId))
  const updateArea = (setId: string, areaId: string, a: Partial<AreaInput>) =>
    guarded(async () => mutate({ action: 'update_area', set_id: setId, area_id: areaId, id: areaId, ...validateAreaInput(a) }, setId))
  const deleteArea = (setId: string, areaId: string) =>
    mutate({ action: 'delete_area', set_id: setId, area_id: areaId, id: areaId }, setId)
  const importGeojson = (setId: string, text: string) => guarded(async () => {
    const parsed = parseAreaGeoJSON(text) // throws AreaError before any request
    return mutate({
      action: 'import_geojson', set_id: setId,
      geojson: { type: 'FeatureCollection', features: parsed.areas.map((a) => ({ type: 'Feature', geometry: a.geometry, properties: { name: a.name } })) },
    }, setId).then((d: any) => ({ ...d, skipped: parsed.skipped, count: parsed.areas.length }))
  })
  const createClub = (name: string) => mutate({ action: 'create_club', name: name.trim() })
  const addMember = (clubId: string, email: string) => mutate({ action: 'add_member', club_id: clubId, email: email.trim() })
  const removeMember = (clubId: string, userId: string) => mutate({ action: 'remove_member', club_id: clubId, user_id: userId })

  return {
    sets, areas, current, loading, error, unavailable,
    refresh, open, createSet, renameSet, deleteSet, addArea, updateArea, deleteArea,
    importGeojson, createClub, addMember, removeMember,
  }
}
