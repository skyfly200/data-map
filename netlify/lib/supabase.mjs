const DEFAULT_TABLE = 'observations'

export function readSupabaseEnv(env = process.env) {
  return {
    url: env.SUPABASE_URL || env.NEXT_PUBLIC_SUPABASE_URL || null,
    key: env.SUPABASE_KEY || env.SUPABASE_ANON_KEY || env.SUPABASE_SERVICE_ROLE_KEY || null,
    table: env.SUPABASE_TABLE || DEFAULT_TABLE,
    syncToSupabase: ['1', 'true', 'yes', 'y', 'on'].includes(String(env.SUPABASE_SYNC ?? env.SYNC_TO_SUPABASE ?? '0').trim().toLowerCase()),
  }
}

export function normalizeObservationRecord(obs) {
  const coords = Array.isArray(obs?.geojson?.coordinates) ? obs.geojson.coordinates : [null, null]
  const id = obs?.id ?? obs?.inat_id ?? null
  const timestamp = obs?.observed_on ?? obs?.date ?? null
  const species = obs?.taxon?.name ?? obs?.species ?? ''

  return {
    inat_id: id,
    uuid: obs?.uuid ?? null,
    species,
    date: timestamp,
    lon: coords[0] ?? null,
    lat: coords[1] ?? null,
    location: obs?.place_guess ?? null,
    quality_grade: obs?.quality_grade ?? null,
    num_identification_agreements: obs?.num_identification_agreements ?? 0,
    raw_payload: JSON.stringify(obs ?? {}),
    updated_at: new Date().toISOString(),
  }
}

export function shouldRefreshAll(env = process.env) {
  const raw = env.REFRESH_ALL ?? env.INAT_REFRESH_ALL ?? env.FULL_REFRESH ?? '0'
  return ['1', 'true', 'yes', 'y', 'on'].includes(String(raw).trim().toLowerCase())
}

export function buildUpsertPayload(rows, { table = DEFAULT_TABLE } = {}) {
  return {
    table,
    rows: (rows || []).map((row) => normalizeObservationRecord(row)),
    upsert: true,
    onConflict: 'inat_id',
  }
}

export async function syncObservationsToSupabase(rows, overrides = {}) {
  const env = readSupabaseEnv(overrides.env ?? process.env)
  const table = overrides.table ?? env.table ?? DEFAULT_TABLE

  if (!env.url || !env.key) {
    return {
      ok: false,
      skipped: true,
      reason: 'Missing SUPABASE_URL or SUPABASE_KEY',
      table,
      rowCount: rows?.length ?? 0,
    }
  }

  const payload = buildUpsertPayload(rows, { table })
  const endpoint = `${env.url.replace(/\/$/, '')}/rest/v1/${table}?on_conflict=inat_id`

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        apikey: env.key,
        Authorization: `Bearer ${env.key}`,
        'Content-Type': 'application/json',
        Prefer: 'resolution=merge-duplicates',
      },
      body: JSON.stringify(payload.rows),
    })

    const text = await response.text()
    if (!response.ok) {
      return {
        ok: false,
        skipped: false,
        table,
        rowCount: payload.rows.length,
        reason: `Supabase request failed (${response.status}): ${text}`,
      }
    }

    return {
      ok: true,
      skipped: false,
      table,
      rowCount: payload.rows.length,
      responseText: text,
    }
  } catch (error) {
    return {
      ok: false,
      skipped: false,
      table,
      rowCount: payload.rows.length,
      reason: String(error),
    }
  }
}
