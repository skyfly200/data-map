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
import L from 'leaflet'
import { onBeforeUnmount, onMounted, ref } from 'vue'

// Qualitative palette (colour-blind friendly), indexed by cluster id.
const PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759',
                 '#b07aa1', '#76b7b2', '#ff9da7', '#9c755f']
const UNCLUSTERED = '#9aa0a6'

// property key -> [label, formatter]
const FIELDS = [
  ['date', 'Observed', (v) => v],
  ['elevation', 'Elevation', (v) => `${Math.round(v)} m`],
  ['land_cover_label', 'Land cover', (v) => v],
  ['ndvi', 'NDVI', num3],
  ['soil_moisture', 'Soil moisture', num3],
  ['solar_exposure', 'Solar exposure', num2],
  ['wind_exposure', 'Wind exposure', num2],
  ['water_retention', 'Water retention', num2],
  ['slope', 'Slope', (v) => `${num1(v)}°`],
]

function num1(v) { return Number(v).toFixed(1) }
function num2(v) { return Number(v).toFixed(2) }
function num3(v) { return Number(v).toFixed(3) }

const mapEl = ref(null)
const loaded = ref(false)
const loadError = ref('')
const legend = ref([])
let map

function colorFor(cluster) {
  if (cluster === null || cluster === undefined || Number.isNaN(cluster)) return UNCLUSTERED
  return PALETTE[cluster % PALETTE.length]
}

function popupHtml(p) {
  const rows = FIELDS
    .filter(([key]) => p[key] !== null && p[key] !== undefined && p[key] !== '')
    .map(([key, label, fmt]) => `<tr><th>${label}</th><td>${fmt(p[key])}</td></tr>`)
    .join('')
  const species = p.species ? `<em>${p.species}</em>` : 'Observation'
  return `<div class="popup"><strong>${species}</strong>`
    + (rows ? `<table>${rows}</table>` : '') + '</div>'
}

onMounted(async () => {
  map = L.map(mapEl.value, { scrollWheelZoom: true }).setView([39.5, -105.7], 7)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18,
  }).addTo(map)

  try {
    const res = await fetch('/data/observations.geojson')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const geo = await res.json()

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
      onEachFeature: (feature, lyr) => lyr.bindPopup(popupHtml(feature.properties)),
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
</style>
