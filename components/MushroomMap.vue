<template>
  <div class="map-shell">
    <div ref="mapEl" class="map"></div>

    <div v-if="loadError" class="overlay error">{{ loadError }}</div>
    <div v-else-if="!loaded" class="overlay">Loading observations…</div>

    <!-- Thematic layer selector -->
    <div v-if="loaded" class="controls">
      <div class="colorby">
        <label for="colorby-sel">Color by</label>
        <select id="colorby-sel" v-model="colorBy">
          <optgroup label="Category">
            <option v-for="o in colorOptions.category" :key="o.key" :value="o.key">{{ o.label }}</option>
          </optgroup>
          <optgroup v-if="colorOptions.numeric.length" label="Numeric">
            <option v-for="o in colorOptions.numeric" :key="o.key" :value="o.key">{{ o.label }}</option>
          </optgroup>
        </select>
      </div>
      <label class="toggle">
        <input type="checkbox" v-model="showFiltered" />
        Include excluded water / non-terrestrial rows
      </label>
    </div>

    <!-- Legend (categorical swatches or a sequential gradient) -->
    <div v-if="loaded && coloring" class="legend">
      <div class="legend-title">{{ coloring.title }}</div>
      <template v-if="coloring.type === 'categorical'">
        <div v-for="item in coloring.legend" :key="item.label" class="legend-row">
          <span class="swatch" :style="{ background: item.color }"></span>
          <span>{{ item.label }}</span>
        </div>
      </template>
      <template v-else>
        <div class="gradient" :style="{ background: `linear-gradient(90deg, ${RAMP[0]}, ${RAMP[1]})` }"></div>
        <div class="gradient-scale"><span>{{ fmtNum(coloring.min) }}</span><span>{{ fmtNum(coloring.max) }}</span></div>
      </template>
    </div>

    <!-- Detail drawer with the weather lead-up -->
    <transition name="slide">
      <aside v-if="selected" class="drawer">
        <button class="close" aria-label="Close" @click="selected = null">×</button>
        <h3><em>{{ selected.species || 'Observation' }}</em></h3>
        <dl class="meta">
          <div v-if="selected.date"><dt>Observed</dt><dd>{{ selected.date }}</dd></div>
          <div v-if="selected.location"><dt>Location</dt><dd>{{ selected.location }}</dd></div>
          <div v-if="hasValue(selected.elevation)"><dt>Elevation</dt><dd>{{ elevLabel(selected.elevation) }}</dd></div>
          <div v-if="hasValue(selected.land_cover_label)"><dt>Land cover</dt><dd>{{ selected.land_cover_label }}</dd></div>
          <div v-if="hasValue(selected.cluster)"><dt>Cluster</dt><dd><span class="chip" :style="{ background: colorFor(selected.cluster) }">{{ selected.cluster }}</span></dd></div>
        </dl>
        <LeadUpCharts :p="selected" />
        <a v-if="inatUrl(selected)" :href="inatUrl(selected)" target="_blank" rel="noopener" class="inat">View on iNaturalist ↗</a>
      </aside>
    </transition>
  </div>
</template>

<script setup>
import 'leaflet/dist/leaflet.css'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { PALETTE, UNCLUSTERED, colorFor, hasValue, inatUrl, useObservations } from '~/composables/useObservations'
import { ALL_CATEGORY, ALL_NUMERIC } from '~/composables/useChartFields'
import { useUnits } from '~/composables/useUnits'

const { data, filteredData, load, showFiltered, setShowFiltered, focusObservation, setFocusObservation } = useObservations()
const { elevLabel } = useUnits()

const mapEl = ref(null)
const loaded = ref(false)
const loadError = ref('')
const colorBy = ref('cluster')
const selected = ref(null)
let map, geoLayer, L

const RAMP = ['#e8f1fb', '#0b3d91'] // sequential light → dark blue

// Field labels + which keys are categorical, drawn from the shared chart
// registry so the map and the Explore builder stay in sync.
const CATEGORY_KEYS = new Set(ALL_CATEGORY.map((f) => f.key))
const FIELD_LABEL = Object.fromEntries([...ALL_CATEGORY, ...ALL_NUMERIC].map((f) => [f.key, f.label]))

// Offer only the dimensions that actually carry data in the current dataset,
// so an un-enriched layer (e.g. NDVI still empty) doesn't yield an all-grey map.
const colorOptions = computed(() => {
  const feats = filteredData.value?.features || []
  const present = (list) => list.filter((f) => feats.some((ft) => hasValue(ft.properties[f.key])))
  return { category: present(ALL_CATEGORY), numeric: present(ALL_NUMERIC) }
})

// If a dataset switch drops the active dimension's data, fall back to the first
// option still available (cluster, in practice).
watch(colorOptions, (opts) => {
  const keys = [...opts.category, ...opts.numeric].map((o) => o.key)
  if (keys.length && !keys.includes(colorBy.value)) colorBy.value = keys[0]
})

function fmtNum(v) { return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2) }

function hexLerp(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t))
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`
}

// Build the colour function + legend for the current "color by" dimension.
const coloring = computed(() => {
  const feats = filteredData.value?.features || []
  const key = colorBy.value
  const title = FIELD_LABEL[key] || key

  // Cluster keeps its own stable palette + an explicit "Unclustered" bucket.
  if (key === 'cluster') {
    const seen = new Set()
    let hasNull = false
    for (const f of feats) {
      const c = f.properties.cluster
      if (hasValue(c)) seen.add(c); else hasNull = true
    }
    const legend = [...seen].sort((a, b) => a - b).map((c) => ({ label: `Cluster ${c}`, color: colorFor(c) }))
    if (hasNull) legend.push({ label: 'Unclustered', color: UNCLUSTERED })
    return { type: 'categorical', title, colorFn: (p) => colorFor(p.cluster), legend }
  }

  // Any other categorical dimension (land cover, species, …): assign palette
  // colours to the distinct values present.
  if (CATEGORY_KEYS.has(key)) {
    const cats = [...new Set(feats.map((f) => f.properties[key]).filter(hasValue))]
    const map2 = new Map(cats.map((v, i) => [v, PALETTE[i % PALETTE.length]]))
    return {
      type: 'categorical', title,
      colorFn: (p) => (hasValue(p[key]) ? map2.get(p[key]) : UNCLUSTERED),
      legend: cats.map((v) => ({ label: String(v), color: map2.get(v) })),
    }
  }

  const vals = feats.map((f) => f.properties[key]).filter(hasValue).map(Number)
  const min = vals.length ? Math.min(...vals) : 0
  const max = vals.length ? Math.max(...vals) : 1
  return {
    type: 'sequential', title, min, max,
    colorFn: (p) => {
      const v = p[key]
      if (!hasValue(v)) return UNCLUSTERED
      return hexLerp(RAMP[0], RAMP[1], (Number(v) - min) / ((max - min) || 1))
    },
  }
})

// Re-style markers when the colouring changes.
watch(coloring, (c) => {
  if (geoLayer) geoLayer.eachLayer((l) => l.setStyle({ fillColor: c.colorFn(l.feature.properties) }))
})

// Rebuild the point layer whenever the dataset changes (e.g. species switch).
function renderPoints(geo) {
  if (!map || !L || !geo) return
  if (geoLayer) { geoLayer.remove(); geoLayer = null }
  selected.value = null

  geoLayer = L.geoJSON(geo, {
    pointToLayer: (feature, latlng) => L.circleMarker(latlng, {
      radius: 6, weight: 1, color: '#222',
      fillColor: coloring.value.colorFn(feature.properties), fillOpacity: 0.85,
    }),
    onEachFeature: (feature, lyr) => {
      lyr.bindTooltip(feature.properties.species || 'Observation', { direction: 'top' })
      lyr.on('click', () => { selected.value = feature.properties })
    },
  }).addTo(map)

  const bounds = geoLayer.getBounds()
  if (bounds.isValid()) map.fitBounds(bounds.pad(0.1))
}

watch(filteredData, (geo) => renderPoints(geo))

// "Open on map" from a chart: select the matching observation and pan to it.
function applyFocus(target) {
  if (!target || !map) return
  const lon = Number(target.lon), lat = Number(target.lat)
  const feats = filteredData.value?.features || []
  const match = (target.uuid && feats.find((f) => f.properties?.uuid === target.uuid))
    || feats.find((f) => {
      const co = f.geometry?.coordinates
      return co && Math.abs(co[0] - lon) < 1e-6 && Math.abs(co[1] - lat) < 1e-6
    })
  if (match) selected.value = match.properties
  if (Number.isFinite(lat) && Number.isFinite(lon)) {
    map.setView([lat, lon], Math.max(map.getZoom() || 0, 12))
  }
  setFocusObservation(null) // consume so a later revisit doesn't re-trigger
}
watch(focusObservation, (t) => t && applyFocus(t))

onMounted(async () => {
  try {
    await nextTick()
    if (!mapEl.value) throw new Error('map container not ready')
    L = (await import('leaflet')).default

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 19,
    })
    const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenTopoMap (CC-BY-SA)', maxZoom: 17,
    })
    const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Imagery © Esri', maxZoom: 19,
    })

    map = L.map(mapEl.value, { scrollWheelZoom: true, layers: [osm] }).setView([39.5, -105.7], 7)
    L.control.layers(
      { 'Street (OSM)': osm, 'Terrain (OpenTopoMap)': topo, 'Satellite (Esri)': sat },
      {}, { position: 'topright', collapsed: true },
    ).addTo(map)

    await load()
    if (!data.value) throw new Error('no data')
    renderPoints(filteredData.value)
    loaded.value = true
    // If arriving via "Open on map" from a chart, focus that observation now.
    if (focusObservation.value) applyFocus(focusObservation.value)
  } catch (err) {
    loadError.value = `Could not load map (${err.message}).`
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

.controls {
  position: absolute; top: 12px; left: 12px; z-index: 500; display: flex; gap: 10px; align-items: center;
  flex-wrap: wrap;
}
.colorby {
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 7px 10px; font: 13px system-ui, sans-serif; display: flex; gap: 8px; align-items: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.colorby label { color: var(--muted); font-weight: 600; }
.colorby select { border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-size: 13px; }
.toggle {
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 7px 10px; font: 13px system-ui, sans-serif; display: inline-flex; gap: 8px; align-items: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.toggle input { accent-color: #2a78d6; }

.legend {
  position: absolute; bottom: 18px; right: 12px; z-index: 500;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 10px 12px; font: 13px/1.4 system-ui, sans-serif; color: #222; min-width: 120px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.legend-title { font-weight: 600; margin-bottom: 6px; }
.legend-row { display: flex; align-items: center; gap: 8px; }
.swatch { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #222; flex: 0 0 auto; }
.gradient { height: 12px; border-radius: 3px; border: 1px solid #ccc; }
.gradient-scale { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; }

.drawer {
  position: absolute; top: 0; right: 0; z-index: 600; width: 320px; max-width: 86%;
  height: 100%; background: var(--surface); box-shadow: -2px 0 12px rgba(0, 0, 0, 0.2);
  padding: 16px 18px; overflow-y: auto; font: 14px/1.45 system-ui, sans-serif;
}
.drawer h3 { margin: 0 26px 10px 0; font-size: 1.05rem; }
.close {
  position: absolute; top: 8px; right: 10px; border: 0; background: transparent;
  font-size: 1.5rem; line-height: 1; color: var(--muted); cursor: pointer;
}
.meta { margin: 0 0 14px; display: grid; gap: 5px; }
.meta div { display: grid; grid-template-columns: 84px 1fr; gap: 8px; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; }
.chip { display: inline-block; min-width: 20px; padding: 0 7px; border-radius: 10px; color: #fff; font-weight: 600; text-align: center; }
.inat { display: inline-block; margin-top: 14px; color: #2b7a3d; font-weight: 600; text-decoration: none; }
.inat:hover { text-decoration: underline; }

.slide-enter-active, .slide-leave-active { transition: transform 0.2s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }
</style>
