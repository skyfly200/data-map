import { shallowRef, watch } from 'vue'
import { hasValue } from '~/composables/useObservations'
import { fmtNum, FIELD_LABEL } from '~/composables/useMapPointStyle'

export function useMapSelection({
  mapRef, LRef, geoLayerRef, filteredData, chunks,
  focusObservation, setFocusObservation,
  coloring, colorBy, sizeBy,
  heatmapMode, heatmapMeta, heatmapCellAt,
  markerStyle, showPoints,
}) {
  const selected = shallowRef(null)
  const selectedLatLng = shallowRef(null)

  let fittedOnce = false
  let suppressFit = false
  let seenChunkVersion = 0

  const esc = (v) => String(v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

  function pointTooltip(feature) {
    const p = feature?.properties || {}
    const co = feature?.geometry?.coordinates
    const rows = []
    const title = p.species || 'Observation'
    if (p.date) rows.push(['Observed', p.date])

    const c = coloring.value
    if (c && typeof c.labelOf === 'function') {
      const v = c.labelOf(p)
      if (hasValue(v) && v !== title) {
        const dim = String(c.title || '')
        const val = String(v)
        if (dim && val.toLowerCase().startsWith(dim.toLowerCase())) rows.push(['', val])
        else rows.push([dim, val])
      }
    } else if (colorBy.value && hasValue(p[colorBy.value])) {
      rows.push([FIELD_LABEL[colorBy.value] || colorBy.value, fmtNum(p[colorBy.value])])
    }
    if (sizeBy.value && hasValue(p[sizeBy.value])) {
      rows.push([`${FIELD_LABEL[sizeBy.value] || sizeBy.value} (size)`, fmtNum(p[sizeBy.value])])
    }

    if (heatmapMode.value && co) {
      const cell = heatmapCellAt(co[1], co[0])
      if (cell) {
        const m = heatmapMode.value
        const meta = heatmapMeta.value
        const label = meta?.label || 'Heatmap'
        const value = meta?.kind === 'field'
          ? `${meta.circular ? `${Math.round(cell.value)}°` : fmtNum(cell.value)} (${cell.samples} obs)`
          : m === 'common' || m === 'land_cover' ? (cell.label || ', ')
            : m === 'season' || m === 'hotspots'
              ? `${Math.round((cell.n ? cell.inWindow / cell.n : 0) * 100)}% of ${cell.n} finds`
              : m === 'richness' ? `${cell.species.size} species`
                : m === 'wind' ? `${Math.round(cell.aspectDeg ?? 0)}°`
                  : `${cell.n} observations`
        rows.push([label, value])
      }
    }

    return `<strong>${esc(title)}</strong>`
      + rows.map(([k, v]) => `<span class="ot-row">${k ? `<span class="ot-k">${esc(k)}</span>` : ''}${esc(v)}</span>`).join('')
  }

  function selectFeature(feature) {
    if (!feature) return null
    const co = feature.geometry?.coordinates
    const thinned = chunks.available.value && !feature.__full
    const base = { ...feature.properties, __thinned: thinned }
    return co ? { ...base, lon: co[0], lat: co[1] } : base
  }

  function renderPoints(geo) {
    const map = mapRef.value
    const L = LRef.value
    if (!map || !L || !geo) return
    if (geoLayerRef.value) { geoLayerRef.value.remove(); geoLayerRef.value = null }
    if (!suppressFit) selected.value = null

    const layer = L.geoJSON(geo, {
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, markerStyle(feature.properties)),
    }).addTo(map)

    layer.bindTooltip((lyr) => pointTooltip(lyr.feature),
                      { direction: 'top', sticky: true, className: 'obs-tip' })
    layer.on('click', (e) => {
      const feature = e.layer?.feature
      if (!feature) return
      selected.value = selectFeature(feature)
      const co = feature.geometry?.coordinates
      selectedLatLng.value = co ? [co[1], co[0]] : null
    })

    if (!showPoints.value) layer.remove()

    const bounds = layer.getBounds()
    if (bounds.isValid() && !suppressFit) {
      map.fitBounds(bounds.pad(0.1), { animate: false })
      fittedOnce = true
    }
    suppressFit = false
    geoLayerRef.value = layer
  }

  watch(filteredData, (geo) => {
    if (chunks.version.value !== seenChunkVersion) {
      seenChunkVersion = chunks.version.value
      if (fittedOnce) suppressFit = true
    }
    renderPoints(geo)
  })

  function applyFocus(target) {
    const map = mapRef.value
    if (!target || !map) return
    const lon = Number(target.lon), lat = Number(target.lat)
    const feats = filteredData.value?.features || []
    const match = (target.uuid && feats.find((f) => f.properties?.uuid === target.uuid))
      || feats.find((f) => {
        const co = f.geometry?.coordinates
        return co && Math.abs(co[0] - lon) < 1e-6 && Math.abs(co[1] - lat) < 1e-6
      })
    if (match) selected.value = selectFeature(match)
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      selectedLatLng.value = [lat, lon]
      suppressFit = true
      map.setView([lat, lon], 15)
    }
    setFocusObservation(null)
  }
  watch(focusObservation, (t) => t && applyFocus(t))

  function pinIcon() {
    const L = LRef.value
    return L.divIcon({
      className: 'obs-pin', iconSize: [28, 40], iconAnchor: [14, 38], tooltipAnchor: [0, -34],
      html: `<svg viewBox="0 0 24 34" width="28" height="40" aria-hidden="true">
        <path d="M12 0C5.4 0 0 5.3 0 11.9 0 20.6 12 34 12 34s12-13.4 12-22.1C24 5.3 18.6 0 12 0z"
              fill="#e34948" stroke="#fff" stroke-width="1.5"/>
        <circle cx="12" cy="12" r="4.5" fill="#fff"/></svg>`,
    })
  }

  let selectedMarker = null
  watch(selectedLatLng, (ll) => {
    const map = mapRef.value
    const L = LRef.value
    if (!map || !L) return
    if (selectedMarker) { selectedMarker.remove(); selectedMarker = null }
    if (ll) selectedMarker = L.marker(ll, { icon: pinIcon(), interactive: false, zIndexOffset: 1000 }).addTo(map)
  })
  watch(selected, (s) => { if (!s) selectedLatLng.value = null })

  return {
    selected, selectedLatLng,
    selectFeature, renderPoints, applyFocus, pointTooltip,
    setSuppressFit: (v) => { suppressFit = v },
    setFittedOnce: (v) => { fittedOnce = v },
  }
}
