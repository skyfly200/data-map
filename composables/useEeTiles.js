// Earth Engine layers on the map.
//
// Unlike the reference layers in composables/mapLayers.js, these have no fixed
// URL: the server asks Earth Engine to render the layer and hands back a tile
// template, which the map then treats like any other. So the catalogue is known
// up front and the template is fetched when a layer is switched on.
//
// Templates expire, and an expired one does not error — it serves blank tiles.
// On a fire map blank ground reads as ground that never burned, which is worse
// than an error, so a template carries the time it was fetched and is re-minted
// rather than reused past it.

const CATALOGUE_URL = '/.netlify/functions/ee-tiles'

/** Re-mint after this long regardless. Shorter than the server's own cache. */
const REFRESH_MS = 60 * 60 * 1000

export function useEeTiles() {
  const { accessToken } = useAuth()

  const catalogue = useState('ee-tiles-catalogue', () => [])
  const error = useState('ee-tiles-error', () => '')
  const loading = useState('ee-tiles-loading', () => false)

  // Keyed by layer key + params, matching the server's cache key, so switching
  // a layer off and on does not re-mint.
  const minted = new Map()

  async function loadCatalogue() {
    if (catalogue.value.length) return catalogue.value
    loading.value = true
    error.value = ''
    try {
      const res = await fetch(CATALOGUE_URL)
      const data = await res.json().catch(() => ({}))
      if (!res.ok || !data.ok) throw new Error(messageFrom(data, res.status))
      catalogue.value = data.layers || []
      return catalogue.value
    } catch (e) {
      // Not fatal: the rest of the map works, there are simply no Earth Engine
      // layers to offer. Reported rather than swallowed so "where did the fire
      // layers go" has an answer.
      error.value = e.message
      catalogue.value = []
      return []
    } finally {
      loading.value = false
    }
  }

  const keyFor = (layer, params) => [layer, ...Object.keys(params || {}).sort()
    .map((k) => `${k}=${params[k]}`)].join('|')

  /** A tile template for one layer at one set of parameters. */
  async function template(layer, params = {}) {
    const id = keyFor(layer, params)
    const held = minted.get(id)
    if (held && Date.now() - held.at < REFRESH_MS) return held

    const token = await accessToken()
    const query = new URLSearchParams({ layer, ...params })
    const res = await fetch(`${CATALOGUE_URL}?${query}`, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(messageFrom(data, res.status))

    const entry = { ...data, at: Date.now() }
    minted.set(id, entry)
    return entry
  }

  return { catalogue, error, loading, loadCatalogue, template, keyFor }
}
