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

/**
 * How long a tile request may be before it goes in a body instead.
 *
 * Well under the point where any browser, proxy or CDN starts truncating or
 * refusing. Almost every layer is far below it; the one that is not is the soil
 * taxonomy selection, which can name several hundred classes at once.
 */
const MAX_URL = 1800

export function useEeTiles() {
  const { accessToken } = useAuth()

  const catalogue = useState('ee-tiles-catalogue', () => [])
  const error = useState('ee-tiles-error', () => '')
  const loading = useState('ee-tiles-loading', () => false)
  // layer key → its class table. Shared, because two layers over the same asset
  // would otherwise each read the same four hundred names.
  const classTables = useState('ee-tiles-classes', () => ({}))

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
    const auth = token ? { authorization: `Bearer ${token}` } : {}
    const query = new URLSearchParams({ layer, ...params })
    const asGet = `${CATALOGUE_URL}?${query}`

    // A GET while it fits, a POST when it does not. Selecting every class of
    // the soil taxonomy raster is a few thousand characters of parameters,
    // which is past what a URL can be relied on to carry — and a selection that
    // fails somewhere above two hundred classes, at a size nobody can predict,
    // is worse than one that cannot be made. The server reads both the same way.
    const res = asGet.length <= MAX_URL
      ? await fetch(asGet, { headers: auth })
      : await fetch(CATALOGUE_URL, {
        method: 'POST',
        headers: { ...auth, 'content-type': 'application/json' },
        body: JSON.stringify({ layer, ...params }),
      })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(messageFrom(data, res.status))

    const entry = { ...data, at: Date.now() }
    minted.set(id, entry)
    return entry
  }

  /**
   * A layer's class table, for the layers whose classes are too many for a key.
   *
   * Static — it is a property of a published asset — so it is cached for the
   * life of the page and shared between every component that asks. The four
   * hundred great groups of the soil taxonomy raster are the only user so far,
   * and they are exactly why this is a separate request rather than part of the
   * catalogue that every viewer loads.
   */
  async function classes(layer) {
    if (classTables.value[layer]) return classTables.value[layer]
    const token = await accessToken()
    const res = await fetch(`${CATALOGUE_URL}?classes=${encodeURIComponent(layer)}`, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(messageFrom(data, res.status))
    classTables.value = { ...classTables.value, [layer]: data }
    return data
  }

  return { catalogue, error, loading, loadCatalogue, template, keyFor, classes }
}
