// Encode the current view into the URL, and restore it from one.
//
// Sharing a screenshot of this app loses everything that made the view worth
// showing — which filters were on, what the marks were coloured by, which
// overlay was up, where the map was pointed. This puts that state in the query
// string so a link reproduces the view, and everything else (QR, social, email,
// embed) is just that link in a different wrapper.
//
// Keys are short because they end up in a QR code, where every character costs
// modules. Anything left at its default is omitted rather than written out, so
// a plain view still yields a plain URL.

import { computed } from 'vue'

// param → [read from state, write into state]. Kept as one table so the two
// directions cannot drift apart.
const NUMERIC = new Set(['z', 'cell', 'd', 'w', 'rad', 'yr', 'mo', 'wk'])

export function useShareState() {
  const { selectedDataset, speciesFilter, setSpeciesFilter, showFiltered, setShowFiltered } = useObservations()
  const { filters, setFilter } = useFilters()
  const overlays = useMapOverlays()
  const appearance = useAppearance()

  /**
   * Current view as a flat param object. Only non-default values are included.
   * `mapView` is passed in by the map (Leaflet owns the centre and zoom).
   */
  function collect({ mapView = null, colorBy = '', sizeBy = '', extra = null } = {}) {
    const q = {}
    const f = filters.value

    if (mapView?.center) {
      q.c = `${mapView.center.lat.toFixed(4)},${mapView.center.lng.toFixed(4)}`
      q.z = String(mapView.zoom)
    }
    if (colorBy && colorBy !== 'cluster') q.color = colorBy
    if (sizeBy) q.size = sizeBy

    if (overlays.mode.value) {
      q.ov = overlays.mode.value
      if (overlays.cellSize.value !== 0.05) q.cell = String(overlays.cellSize.value)
      if (overlays.mode.value === 'season' || overlays.mode.value === 'hotspots') {
        q.d = String(overlays.seasonDay.value)
        q.w = String(overlays.seasonWindow.value)
      }
    }

    // Species can be a long list; it is the filter most worth sharing, so it
    // goes in whole rather than being truncated into something misleading.
    if (speciesFilter.value?.length) q.sp = speciesFilter.value.join('|')
    if (showFiltered.value) q.wf = '1'

    for (const [key, param] of [['country', 'co'], ['state', 'st'], ['county', 'cty'],
      ['year', 'yr'], ['month', 'mo'], ['week', 'wk'],
      ['dateFrom', 'from'], ['dateTo', 'to']]) {
      if (f[key]) q[param] = String(f[key])
    }
    if (f.center && f.radiusKm) {
      q.near = `${Number(f.center.lat).toFixed(4)},${Number(f.center.lng).toFixed(4)}`
      q.rad = String(f.radiusKm)
    }

    if (appearance.paletteKey.value !== 'default') q.pal = appearance.paletteKey.value
    if (selectedDataset.value && !selectedDataset.value.endsWith('observations.geojson')) {
      q.ds = selectedDataset.value
    }

    // Params the calling view owns — a built chart's configuration, say. They go
    // in last but never overwrite the view state above, so a chart link still
    // carries the filters that produced the data behind the chart.
    for (const [key, value] of Object.entries(extra || {})) {
      if (value !== null && value !== undefined && value !== '' && q[key] === undefined) {
        q[key] = String(value)
      }
    }
    return q
  }

  /**
   * Apply share params onto the live state. Unknown or malformed values are
   * skipped rather than throwing — a hand-edited or truncated link should still
   * open the app, just with less of the view restored.
   *
   * Returns the map view to apply, if the link carried one.
   */
  function apply(query = {}) {
    const num = (v) => {
      const n = Number(v)
      return Number.isFinite(n) ? n : null
    }

    if (query.ov && overlays.OVERLAY_MODES.some((m) => m.key === query.ov)) {
      overlays.mode.value = query.ov
    }
    if (query.cell !== undefined) {
      const c = num(query.cell)
      if (c && overlays.CELL_SIZES.some((x) => x.value === c)) overlays.cellSize.value = c
    }
    if (query.d !== undefined) {
      const d = num(query.d)
      if (d !== null && d >= 1 && d <= 366) overlays.seasonDay.value = d
    }
    if (query.w !== undefined) {
      const w = num(query.w)
      if (w !== null && w >= 1 && w <= 182) overlays.seasonWindow.value = w
    }

    if (query.sp) setSpeciesFilter(String(query.sp).split('|').filter(Boolean))
    if (query.wf === '1') setShowFiltered(true)

    for (const [key, param] of [['country', 'co'], ['state', 'st'], ['county', 'cty'],
      ['dateFrom', 'from'], ['dateTo', 'to']]) {
      if (query[param]) setFilter(key, String(query[param]))
    }
    for (const [key, param] of [['year', 'yr'], ['month', 'mo'], ['week', 'wk']]) {
      const v = num(query[param])
      if (v !== null) setFilter(key, v)
    }
    if (query.near && query.rad) {
      const [lat, lng] = String(query.near).split(',').map(Number)
      const rad = num(query.rad)
      if (Number.isFinite(lat) && Number.isFinite(lng) && rad) {
        filters.value = { ...filters.value, center: { lat, lng }, radiusKm: rad }
      }
    }

    if (query.pal && appearance.PALETTES.some((p) => p.key === query.pal)) {
      appearance.paletteKey.value = query.pal
    }

    let view = null
    if (query.c) {
      const [lat, lng] = String(query.c).split(',').map(Number)
      const z = num(query.z)
      if (Number.isFinite(lat) && Number.isFinite(lng)) {
        view = { center: [lat, lng], zoom: z !== null && z >= 1 && z <= 20 ? z : 10 }
      }
    }
    return {
      view,
      colorBy: typeof query.color === 'string' ? query.color : null,
      sizeBy: typeof query.size === 'string' ? query.size : null,
    }
  }

  /** Absolute URL for the current view. `path` defaults to the current route. */
  function buildUrl(state = {}, path = null) {
    const origin = import.meta.client ? window.location.origin : ''
    const route = path || (import.meta.client ? window.location.pathname : '/')
    const params = new URLSearchParams(collect(state))
    const qs = params.toString()
    return `${origin}${route}${qs ? `?${qs}` : ''}`
  }

  /** The same link with ?embed=1, which strips the site chrome for an iframe. */
  function buildEmbedUrl(state = {}, path = null) {
    const url = new URL(buildUrl(state, path), import.meta.client ? window.location.origin : 'https://example.com')
    url.searchParams.set('embed', '1')
    return url.toString()
  }

  function buildEmbedCode(state = {}, path = null, { width = '100%', height = 520 } = {}) {
    const src = buildEmbedUrl(state, path)
    return `<iframe src="${src}" width="${width}" height="${height}" `
      + 'style="border:1px solid #ddd;border-radius:8px" loading="lazy" '
      + 'title="Mushroom observations"></iframe>'
  }

  // True when the app is being rendered inside someone else's page.
  const isEmbed = computed(() => {
    if (!import.meta.client) return false
    return new URLSearchParams(window.location.search).get('embed') === '1'
  })

  return { collect, apply, buildUrl, buildEmbedUrl, buildEmbedCode, isEmbed, NUMERIC }
}
