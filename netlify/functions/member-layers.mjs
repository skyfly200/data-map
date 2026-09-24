import { adminClient, requireMember } from '../lib/auth.mjs'
import { CustomLayerError, normaliseCustomLayer } from '../lib/ee-custom-layers.mjs'

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

export default async function handler(request) {
  const auth = await requireMember(request, { message: 'Adding custom layers is a membership benefit.' })
  if (!auth.ok) return auth.response

  const client = adminClient()
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)

  const userId = auth.user?.id

  if (request.method === 'GET') {
    const { data, error } = await client
      .from('ee_custom_layers')
      .select('*')
      .eq('created_by', userId)
      .order('name')
    if (error) return json({ ok: false, error: error.message }, 500)
    return json({ ok: true, layers: data || [] })
  }

  if (request.method !== 'POST') return json({ ok: false, error: 'Use GET or POST.' }, 405)

  let body
  try { body = await request.json() } catch { return json({ ok: false, error: 'Send JSON.' }, 400) }

  try {
    if (body.action === 'save-layer') {
      // Members can only edit their own layers.
      if (body.id) {
        const { data: existing } = await client
          .from('ee_custom_layers').select('created_by').eq('id', body.id).single()
        if (!existing) throw new CustomLayerError('Layer not found.')
        if (existing.created_by !== userId) throw new CustomLayerError('You can only edit your own layers.')
      }

      const layer = normaliseCustomLayer(body)
      const row = { ...layer, created_by: userId }

      const { data, error } = body.id
        ? await client.from('ee_custom_layers').update(layer).eq('id', body.id).eq('created_by', userId).select().single()
        : await client.from('ee_custom_layers').insert(row).select().single()
      if (error) {
        throw new Error(/duplicate key|unique/i.test(error.message)
          ? `A layer with the short name "${layer.slug}" already exists.`
          : error.message)
      }
      return json({ ok: true, layer: data })
    }

    if (body.action === 'delete-layer') {
      if (!body.id) throw new Error('Which layer?')
      const { error } = await client
        .from('ee_custom_layers').delete().eq('id', body.id).eq('created_by', userId)
      if (error) throw new Error(error.message)
      return json({ ok: true, deleted: body.id })
    }

    return json({ ok: false, error: 'Unknown action.' }, 400)
  } catch (err) {
    return json({ ok: false, error: err.message }, err instanceof CustomLayerError ? 400 : 500)
  }
}
