// Foray planner model layers (WANT-17 phases 2, 3 and 5).
// Auth: Bearer token. Models are the caller's own plus public ones.
//
//   GET  /.netlify/functions/foray-layers
//     -> { ok, models:[{ id, title, taxon, auc, own, registered, ranges, contributions, usable }] }
//        Models with a finished run. taxon is the source taxon the model was
//        trained on (null when it was trained on a whole dataset). ranges /
//        contributions say whether the habitat score can use it; usable says
//        whether the ensemble can (a finished job, or a registered asset).
//   GET  ...?layer=ensemble|habitat&models=<id>:<weight>,...   (max 6 models)
//     -> { ok, template, meta:{ layer, used, skipped, predictors, legend, mintedAt, cached? } }
//        ensemble: weighted mean of the models' suitability surfaces (each refit,
//        or read from its asset). habitat: weighted share of the top predictors
//        inside the range the finds were made in (no refit). Both 0..1.
//   POST { layer, models: '<id>:<w>,...', points: [[lon, lat], ...] }  (max 1500 points)
//     -> { ok, samples:[{ i, value, outside }], used, skipped }
//        value is the layer at each point (null where masked); outside is true
//        where a habitat predictor falls outside every model's presence range,
//        i.e. a high value there is an extrapolation.
//
// Errors: 400 bad input, 401 no auth, 404 a model not found or not visible,
// 409 nothing usable in the models picked, 502 Earth Engine, 503 not configured.

import { adminClient, requireUser } from '../lib/auth.mjs'
import { ownerViewer } from '../lib/job-queue.mjs'
import { loadSource } from '../lib/job-source.mjs'
import { earthEngineConfigured, mintForayLayer, sampleForayLayer } from '../lib/ee-runner.mjs'
import { effectiveTier } from '../lib/tiers.mjs'
import { FORAY_LAYERS, LayerSpecError, parseModelWeights, parseSamplePoints } from '../lib/foray-layers.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

const filled = (v) => Boolean(v && typeof v === 'object' && Object.keys(v).length)

/**
 * Every model the viewer may use, newest first, with its latest finished run.
 * Contributions and ranges come from the job's result_meta (always written by
 * the runner) and fall back to model_results (registered models, and rows the
 * write-back reached).
 */
export async function visibleModels(client, viewer) {
  let q = client.from('model_configs')
    .select('id, owner_id, title, visibility, created_at, model_runs(job_id, created_at, run_meta), model_results(contributions, predictor_ranges, suitability_asset_path, auc)')
  if (!viewer.admin) q = q.or(`owner_id.eq.${viewer.userId},visibility.eq.public`)
  const { data: configs, error } = await q
  if (error) throw new Error(error.message)

  const latestRun = (c) => [...(c.model_runs || [])].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
  const jobIds = (configs || []).flatMap((c) => latestRun(c).map((r) => r.job_id)).filter(Boolean)
  const jobs = new Map()
  if (jobIds.length) {
    const { data, error: jobErr } = await client.from('ee_jobs')
      .select('id, user_id, kind, status, params, result_meta').in('id', jobIds)
    if (jobErr) throw new Error(jobErr.message)
    for (const j of data || []) jobs.set(String(j.id), j)
  }

  const out = []
  for (const c of configs || []) {
    const runs = latestRun(c)
    const result = (c.model_results || [])[0] || {}
    const registered = runs.some((r) => r.run_meta?.registered === true) && Boolean(result.suitability_asset_path)
    const job = runs.map((r) => jobs.get(String(r.job_id))).find((j) => j && j.kind === 'model' && j.status === 'succeeded') || null
    if (!job && !registered) continue
    const meta = job?.result_meta || {}
    const contributions = filled(meta.contributions) ? meta.contributions : filled(result.contributions) ? result.contributions : null
    const ranges = filled(meta.predictorRanges) ? meta.predictorRanges : filled(result.predictor_ranges) ? result.predictor_ranges : null
    const auc = Number.isFinite(Number(meta.cv?.auc)) ? Number(meta.cv.auc) : Number.isFinite(Number(result.auc)) ? Number(result.auc) : null
    out.push({
      id: c.id, title: c.title, own: c.owner_id === viewer.userId, visibility: c.visibility,
      taxon: job?.params?.source?.taxon || null, auc, registered,
      contributions, ranges,
      job, assetPath: registered ? result.suitability_asset_path : null,
    })
  }
  return out
}

const listRow = (m) => ({
  id: m.id, title: m.title, taxon: m.taxon, auc: m.auc, own: m.own, registered: m.registered,
  ranges: Boolean(m.ranges), contributions: Boolean(m.contributions), usable: Boolean(m.job || m.assetPath),
})

/** The picked models, loaded for the runner. Throws {status:404} for any not visible. */
async function loadPicked(client, viewer, picked, layer, loadFeatures) {
  const all = new Map((await visibleModels(client, viewer)).map((m) => [m.id, m]))
  const out = []
  for (const { id, weight } of picked) {
    const m = all.get(id)
    if (!m) throw Object.assign(new Error(`Model ${id} was not found.`), { status: 404 })
    const row = { id, weight, contributions: m.contributions, ranges: m.ranges, assetPath: m.assetPath }
    if (layer === 'ensemble' && !m.assetPath && m.job) {
      row.spec = m.job.params || {}
      row.features = await loadFeatures(row.spec.source, m.job)
    }
    out.push(row)
  }
  return out
}

export async function handleForayLayers(request, { client, auth, runner, loadFeatures, eeReady }) {
  const viewer = { userId: auth.user?.id ?? null, admin: auth.admin === true }
  const url = new URL(request.url)

  try {
    if (request.method === 'GET' && !url.searchParams.get('layer')) {
      const models = await visibleModels(client, viewer)
      return json({ ok: true, models: models.map(listRow) })
    }

    let layer, rawModels, points
    if (request.method === 'GET') {
      layer = url.searchParams.get('layer')
      rawModels = url.searchParams.get('models')
    } else if (request.method === 'POST') {
      const body = await request.json().catch(() => null)
      if (!body) return json({ ok: false, error: 'Send a JSON body.' }, 400)
      layer = body.layer
      rawModels = Array.isArray(body.models) ? body.models.map((m) => `${m.id}:${m.weight ?? 1}`).join(',') : body.models
      points = parseSamplePoints(body.points)
    } else {
      return json({ ok: false, error: 'Use GET or POST.' }, 405)
    }
    if (!FORAY_LAYERS.includes(layer)) return json({ ok: false, error: `layer must be one of ${FORAY_LAYERS.join(', ')}.` }, 400)
    const picked = parseModelWeights(rawModels)
    if (!eeReady()) return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)

    const models = await loadPicked(client, viewer, picked, layer, loadFeatures)
    if (points) {
      const { samples, used, skipped } = await runner.sample({ layer, models, points })
      return json({ ok: true, samples, used, skipped })
    }
    const cacheKey = `${layer}:${models.map((m) => `${m.id}:${m.weight.toFixed(3)}:${m.job?.id ?? ''}`).join(',')}`
    const { template, meta } = await runner.mint({ layer, models, cacheKey })
    return json({ ok: true, template, meta })
  } catch (err) {
    if (err instanceof LayerSpecError) return json({ ok: false, error: err.message }, 400)
    const status = err?.status || 502
    return json({ ok: false, error: String(err?.message || err).slice(0, 300) }, status)
  }
}

export default async function handler(request) {
  const auth = await requireUser(request)
  if (!auth.ok) return auth.response
  const client = adminClient()
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)
  let admin = !auth.user
  if (auth.user) {
    const { data } = await client.from('profiles').select('tier, member_until').eq('user_id', auth.user.id).maybeSingle()
    admin = effectiveTier(data) === 'admin'
  }
  return handleForayLayers(request, {
    client,
    auth: { ...auth, admin },
    eeReady: earthEngineConfigured,
    runner: { mint: mintForayLayer, sample: sampleForayLayer },
    // Read as the member who owns the model, as model-tiles does.
    loadFeatures: async (source, job) => loadSource(source, { client, viewer: await ownerViewer(job) }),
  })
}
