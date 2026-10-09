// Fetch wrapper over /.netlify/functions/foray-layers (WANT-17 phases 2, 3, 5).
// Lists the models the viewer can use, mints a layer's tile template, and
// samples a layer at points. Errors become a message, never a throw, so the
// planner keeps working without them.

import { ref } from 'vue'
import type { ForayLayer, ForayModel } from './forayModels'

export function useForayLayers() {
  const { accessToken } = useAuth()
  const models = ref<ForayModel[]>([])
  const listed = ref(false)
  const message = ref('')

  async function call(query = '', body?: object) {
    const token = await accessToken()
    let res: Response
    try {
      res = await fetch(`/.netlify/functions/foray-layers${query}`, {
        method: body ? 'POST' : 'GET',
        headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
        ...(body ? { body: JSON.stringify(body) } : {}),
      })
    } catch {
      throw new Error('Could not reach the server.')
    }
    const data = await res.json().catch(() => null)
    if (res.status === 401) throw new Error('Sign in to use your models here.')
    if (!res.ok || !data?.ok) throw new Error(data?.error || `That request failed (${res.status}).`)
    return data
  }

  async function list() {
    message.value = ''
    try {
      models.value = (await call()).models || []
    } catch (e: any) {
      models.value = []
      message.value = e.message
    } finally {
      listed.value = true
    }
    return models.value
  }

  async function mint(layer: ForayLayer, modelsParam: string): Promise<{ template: string, meta: any } | null> {
    message.value = ''
    try {
      const d = await call(`?layer=${layer}&models=${encodeURIComponent(modelsParam)}`)
      return { template: d.template, meta: d.meta }
    } catch (e: any) {
      message.value = e.message
      return null
    }
  }

  async function sample(layer: ForayLayer, modelsParam: string, points: [number, number][]) {
    message.value = ''
    try {
      return await call('', { layer, models: modelsParam, points })
    } catch (e: any) {
      message.value = e.message
      return null
    }
  }

  return { models, listed, message, list, mint, sample }
}
