// Carry a viewer's settings and saved charts across devices via Supabase.
//
// The design deliberately keeps localStorage as the working copy rather than
// reading from the network on every access: the app must stay fully usable
// signed out, or with Supabase not configured at all, and every existing
// composable already persists to localStorage. So this syncs the STORAGE KEYS
// themselves — snapshot them up, apply them back down — which means a preference
// added later syncs by being added to one list, with no other wiring.
//
// Conflict handling is last-write-wins, with one exception that matters: the
// first sign-in on a device MERGES rather than overwrites, so a viewer who
// configured the app before making an account does not lose that work.

const SETTINGS_KEYS = [
  'appearance',                    // palette, shapes, per-value overrides
  'chart-layout',                  // gallery order + hidden charts
  'map-overlay',                   // overlay mode, cell size, date window
  'units',                         // ft/m, °F/°C
  'map-color-by',
  'map-size-by',
  'observations-show-filtered',
]

const CHARTS_KEY = 'saved-charts'
const LAST_PULL_KEY = 'cloud-sync-pulled-for'

// ─── Pure helpers ────────────────────────────────────────────────────────────
// Exported separately from the composable so the parts that are easy to get
// wrong — the first-sign-in merge, and the mapping between chart objects and
// table rows — can be tested without a Nuxt runtime or a live database.

export { SETTINGS_KEYS, CHARTS_KEY, LAST_PULL_KEY }

/**
 * Settings for a device's first sync with an account.
 *
 * Remote wins key-by-key, because the account is the shared truth; but a key
 * this device has and the account does not is kept and pushed up, so someone
 * who configured the app before signing up does not lose that work.
 */
export function mergeSettings(local = {}, remote = null) {
  return { ...(local || {}), ...(remote || {}) }
}

/** Chart objects → saved_charts rows, position taken from array order. */
export function chartsToRows(charts = [], userId) {
  return charts.map((c, i) => {
    // `id` is dropped: the database assigns it. `title` is promoted to its own
    // column so charts can be listed without parsing the config blob.
    const { id, title, ...config } = c
    return { user_id: userId, config, title: title ?? null, position: i }
  })
}

/** saved_charts rows → chart objects the app renders. */
export function rowsToCharts(rows = []) {
  return (rows || []).map((r) => ({ ...r.config, id: r.id, title: r.title ?? r.config?.title }))
}

/**
 * useCloudSync() where a Nuxt context may not exist.
 *
 * Composables capture this during setup so their persist() — which runs from
 * DOM event handlers, outside any Nuxt context — can still push. Returns null
 * rather than throwing when there is no context, which is the correct outcome:
 * no context means no session means nothing to sync.
 */
export function safeCloudSync() {
  try {
    return useCloudSync()
  } catch {
    return null
  }
}

export function useCloudSync() {
  const { $supabase } = useNuxtApp()
  const { user, isAuthed, configured } = useAuth()

  // 'off' when there is no account to sync with — the app is local-only and
  // that is a valid, complete state, not a degraded one.
  const status = useState('cloud-sync-status', () => 'off')
  const error = useState('cloud-sync-error', () => '')
  const lastSynced = useState('cloud-sync-at', () => null)

  const enabled = computed(() => Boolean(configured && isAuthed.value && $supabase))

  function readLocal(key) {
    if (!import.meta.client) return null
    try {
      const raw = localStorage.getItem(key)
      return raw === null ? null : JSON.parse(raw)
    } catch {
      // Not every key holds JSON (map-color-by is a bare string).
      try { return localStorage.getItem(key) } catch { return null }
    }
  }

  function writeLocal(key, value) {
    if (!import.meta.client || value === null || value === undefined) return
    try {
      localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value))
    } catch { /* quota or private mode */ }
  }

  /** Every synced preference as one plain object. */
  function snapshotSettings() {
    const out = {}
    for (const key of SETTINGS_KEYS) {
      const v = readLocal(key)
      if (v !== null && v !== undefined) out[key] = v
    }
    return out
  }

  function applySettings(settings) {
    if (!settings || typeof settings !== 'object') return
    for (const key of SETTINGS_KEYS) {
      if (key in settings) writeLocal(key, settings[key])
    }
  }

  // ─── Settings ──────────────────────────────────────────────────────────────

  async function pullSettings() {
    if (!enabled.value) return null
    const { data, error: err } = await $supabase
      .from('user_settings').select('settings').eq('user_id', user.value.id).maybeSingle()
    if (err) throw err
    return data?.settings ?? null
  }

  async function pushSettings(settings = null) {
    if (!enabled.value) return
    const payload = settings || snapshotSettings()
    const { error: err } = await $supabase
      .from('user_settings')
      .upsert({ user_id: user.value.id, settings: payload, updated_at: new Date().toISOString() },
              { onConflict: 'user_id' })
    if (err) throw err
  }

  // ─── Saved charts ──────────────────────────────────────────────────────────

  async function pullCharts() {
    if (!enabled.value) return null
    const { data, error: err } = await $supabase
      .from('saved_charts').select('id, config, title, position')
      .eq('user_id', user.value.id).order('position', { ascending: true })
    if (err) throw err
    if (!data) return null
    // The app addresses charts by `id`, so the row id becomes the chart id.
    return rowsToCharts(data)
  }

  /**
   * Replace the account's charts with `charts`.
   *
   * A delete-then-insert keeps the stored order identical to the on-screen
   * order without a per-row diff; the set is small (a handful of charts) and it
   * cannot drift out of sync the way an incremental update can.
   */
  async function pushCharts(charts = []) {
    if (!enabled.value) return []
    const uid = user.value.id
    const { error: delErr } = await $supabase.from('saved_charts').delete().eq('user_id', uid)
    if (delErr) throw delErr
    if (!charts.length) return []

    const rows = chartsToRows(charts, uid)
    const { data, error: err } = await $supabase.from('saved_charts').insert(rows).select('id, config, title, position')
    if (err) throw err
    return rowsToCharts(data)
  }

  // ─── Orchestration ─────────────────────────────────────────────────────────

  /**
   * Bring this device in line with the account.
   *
   * On the FIRST sync for a given user on this device, local work is merged
   * upward: remote settings win key-by-key where both exist, but local-only
   * keys and local-only charts are kept and pushed. After that, remote is the
   * source of truth and simply replaces local.
   */
  async function sync({ force = false } = {}) {
    if (!enabled.value) {
      status.value = 'off'
      return { settings: null, charts: null }
    }
    status.value = 'syncing'
    error.value = ''
    try {
      const firstTime = readLocal(LAST_PULL_KEY) !== user.value.id
      const [remoteSettings, remoteCharts] = await Promise.all([pullSettings(), pullCharts()])

      let settings = remoteSettings
      let charts = remoteCharts

      if (firstTime && !force) {
        // Remote wins per key; anything configured only on this device survives.
        settings = mergeSettings(snapshotSettings(), remoteSettings)
        const localCharts = readLocal(CHARTS_KEY) || []
        if (localCharts.length && !(remoteCharts || []).length) {
          charts = await pushCharts(localCharts)
        }
        await pushSettings(settings)
      }

      if (settings) applySettings(settings)
      if (charts) writeLocal(CHARTS_KEY, charts)
      writeLocal(LAST_PULL_KEY, user.value.id)

      status.value = 'synced'
      lastSynced.value = new Date().toISOString()
      return { settings, charts }
    } catch (e) {
      // A sync failure must never cost the viewer their local state.
      status.value = 'error'
      error.value = e?.message || 'Could not sync with your account.'
      return { settings: null, charts: null }
    }
  }

  // Pushes are debounced: changing a slider should not be one request per pixel.
  let pushTimer = null
  function schedulePush(delay = 1200) {
    if (!enabled.value || !import.meta.client) return
    clearTimeout(pushTimer)
    pushTimer = setTimeout(async () => {
      try {
        await pushSettings()
        lastSynced.value = new Date().toISOString()
        status.value = 'synced'
      } catch (e) {
        status.value = 'error'
        error.value = e?.message || 'Could not save to your account.'
      }
    }, delay)
  }

  return {
    SETTINGS_KEYS, CHARTS_KEY,
    enabled, status, error, lastSynced,
    snapshotSettings, applySettings,
    pullSettings, pushSettings, pullCharts, pushCharts,
    sync, schedulePush,
  }
}
