<template>
  <div class="map-shell">
    <div ref="mapEl" class="map"></div>

    <div v-if="loadError" class="overlay error">{{ loadError }}</div>
    <div v-else-if="!loaded" class="overlay">Loading observations…</div>

    <div v-if="loaded && legend.length" class="legend">
      <div class="legend-title">Environmental cluster</div>
      <div v-for="item in legend" :key="item.key" class="legend-row">
        <span class="swatch" :style="{ background: item.color }"></span>
        <span>{{ item.label }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import 'leaflet/dist/leaflet.css'
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { FIELDS, UNCLUSTERED, colorFor, hasValue, inatUrl, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const { data, load } = useObservations()
const { elevLabel } = useUnits()

const mapEl = ref(null)
const loaded = ref(false)
const loadError = ref('')
const legend = ref([])
let map

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ))
}

function popupHtml(p) {
  const parts = []
  if (hasValue(p.elevation)) {
    parts.push(`<tr><th>Elevation</th><td>${escapeHtml(elevLabel(p.elevation))}</td></tr>`)
  }
  for (const [key, label, fmt] of FIELDS) {
    if (hasValue(p[key])) parts.push(`<tr><th>${label}</th><td>${escapeHtml(fmt(p[key]))}</td></tr>`)
  }
  const rows = parts.join('')
  const species = p.species ? `<em>${escapeHtml(p.species)}</em>` : 'Observation'
  const url = inatUrl(p)
  const link = url ? `<a href="${url}" target="_blank" rel="noopener">View on iNaturalist ↗</a>` : ''
  return `<div class="popup"><strong>${species}</strong>`
    + (rows ? `<table>${rows}</table>` : '')
    + (link ? `<div class="popup-link">${link}</div>` : '')
    + '</div>'
}

onMounted(async () => {
  try {
    // Wait for the container to be in the DOM, then load Leaflet client-side.
    await nextTick()
    if (!mapEl.value) throw new Error('map container not ready')
    const L = (await import('leaflet')).default

    map = L.map(mapEl.value, { scrollWheelZoom: true }).setView([39.5, -105.7], 7)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 18,
    }).addTo(map)

    await load()
    const geo = data.value
    if (!geo) throw new Error('no data')

    const seen = new Set()
    const layer = L.geoJSON(geo, {
      pointToLayer: (feature, latlng) => {
        const c = feature.properties.cluster
        if (c !== null && c !== undefined) seen.add(c)
        return L.circleMarker(latlng, {
          radius: 6, weight: 1, color: '#222',
          fillColor: colorFor(c), fillOpacity: 0.85,
        })
      },
      // Function content is re-evaluated each open, so popups reflect the
      // current elevation unit without rebuilding the layer.
      onEachFeature: (feature, lyr) => lyr.bindPopup(() => popupHtml(feature.properties)),
    }).addTo(map)

    const bounds = layer.getBounds()
    if (bounds.isValid()) map.fitBounds(bounds.pad(0.1))

    legend.value = [...seen].sort((a, b) => a - b)
      .map((c) => ({ key: c, color: colorFor(c), label: `Cluster ${c}` }))
    if (geo.features.some((f) => f.properties.cluster === null || f.properties.cluster === undefined)) {
      legend.value.push({ key: 'none', color: UNCLUSTERED, label: 'Unclustered' })
    }
    loaded.value = true
  } catch (err) {
    loadError.value = `Could not load observations (${err.message}).`
  }
})

onBeforeUnmount(() => { if (map) map.remove() })
</script>

<style scoped>
.map-shell { position: relative; width: 100%; height: 100%; }
.map { width: 100%; height: 100%; }

.overlay {
  position: absolute; inset: 0; display: grid; place-items: center;
  background: rgba(255, 255, 255, 0.7); font: 500 15px/1.4 system-ui, sans-serif;
  color: #333; z-index: 500; pointer-events: none;
}
.overlay.error { color: #b00020; }

.legend {
  position: absolute; bottom: 18px; right: 12px; z-index: 500;
  background: rgba(255, 255, 255, 0.94); border: 1px solid #ddd; border-radius: 8px;
  padding: 10px 12px; font: 13px/1.4 system-ui, sans-serif; color: #222;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.legend-title { font-weight: 600; margin-bottom: 6px; }
.legend-row { display: flex; align-items: center; gap: 8px; }
.swatch { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #222; }

:deep(.popup table) { border-collapse: collapse; margin-top: 6px; }
:deep(.popup th) { text-align: left; padding: 1px 10px 1px 0; color: #666; font-weight: 500; }
:deep(.popup td) { text-align: right; }
:deep(.popup-link) { margin-top: 8px; }
:deep(.popup-link a) { color: #2b7a3d; font-weight: 600; text-decoration: none; }
:deep(.popup-link a:hover) { text-decoration: underline; }
</style>
