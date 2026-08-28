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
      <div v-if="colorOptions.numeric.length" class="colorby">
        <label for="sizeby-sel">Size by</label>
        <select id="sizeby-sel" v-model="sizeBy">
          <option value="">Uniform</option>
          <option v-for="o in colorOptions.numeric" :key="o.key" :value="o.key">{{ o.label }}</option>
        </select>
      </div>
      <button class="locate" :class="{ busy: locating }" :title="locateError || 'Show my location'"
              @click="locateMe">
        <span class="dot-icon"></span>{{ locating ? 'Locating…' : 'My location' }}
      </button>
      <LiveClusterControls />
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
        <div v-if="observationInfo && observationInfo.photos && observationInfo.photos.length" class="photos">
          <img v-for="p in observationInfo.photos" :src="p.url" :alt="'Observation photo'" class="obs-photo" />
        </div>
        <div v-if="observationInfo && observationInfo.description" class="description">
          {{ observationInfo.description }}
        </div>
        <LeadUpCharts :p="selected" />
        <a v-if="inatUrl(selected)" :href="inatUrl(selected)" target="_blank" rel="noopener" class="inat">View on iNaturalist ↗</a>
      </aside>
    </transition>
  </div>
</template>

<script setup>
import 'leaflet/dist/leaflet.css'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { PALETTE, UNCLUSTERED, categoryColor, colorFor, hasValue, inatUrl, fetchObservationDetails, useObservations } from '~/composables/useObservations'
import { ALL_CATEGORY, ALL_NUMERIC } from '~/composables/useChartFields'
import { useUnits } from '~/composables/useUnits'

const { data, filteredData, load, showFiltered, setShowFiltered, focusObservation, setFocusObservation } = useObservations()
const { elevLabel, elevValue, tempValue, unit, tempUnit } = useUnits()
const live = useLiveClusters()

const mapEl = ref(null)
const loaded = ref(false)
const loadError = ref('')
// Remember the "Color by" dimension per viewer.
const COLORBY_KEY = 'map-color-by'
const colorBy = ref('cluster')
if (import.meta.client) {
  const saved = localStorage.getItem(COLORBY_KEY)
  if (saved) colorBy.value = saved
}
watch(colorBy, (v) => { if (import.meta.client) localStorage.setItem(COLORBY_KEY, v) })
// Remember the "Size by" dimension per viewer.
const SIZEBY_KEY = 'map-size-by'
const sizeBy = ref('')
if (import.meta.client) {
  const saved = localStorage.getItem(SIZEBY_KEY)
  if (saved !== null) sizeBy.value = saved
}
watch(sizeBy, (v) => { if (import.meta.client) localStorage.setItem(SIZEBY_KEY, v) })
const selected = ref(null)
const selectedLatLng = ref(null)
const locating = ref(false)
const locateError = ref('')
let map, geoLayer, L, userLayer, selectedMarker

// Holds enriched observation info (photos, description, etc.) fetched from iNaturalist API
const observationInfo = ref(null)
watch(selected, async (s) => {
  if (s) {
    const id = s.inat_id ?? s.uuid
    observationInfo.value = await fetchObservationDetails(id)
  } else {
    observationInfo.value = null
  }
})

const RAMP = ['#e8f1fb', '#0b3d91'] // sequential light → dark blue

// Field labels + which keys are categorical, drawn from the shared chart
// registry so the map and the Explore builder stay in sync.
const CATEGORY_KEYS = new Set(ALL_CATEGORY.map((f) => f.key))
const FIELD_LABEL = Object.fromEntries([...ALL_CATEGORY, ...ALL_NUMERIC].map((f) => [f.key, f.label]))

// Offer only the dimensions that actually carry data in the current dataset,
// so an un-enriched layer (e.g. NDVI still empty) doesn't yield an all-grey map.
const colorOptions = computed(() => {
  const feats = filteredData.value?.features || []
  const present = (list) => list.filter((f) => (
    f.key === 'live_cluster' ? live.active.value : feats.some((ft) => hasValue(ft.properties[f.key]))
  ))
  return { category: present(ALL_CATEGORY), numeric: present(ALL_NUMERIC) }
})

// If a dataset switch drops the active dimension's data, fall back to the first
// option still available (cluster, in practice).
watch(colorOptions, (opts) => {
  const keys = [...opts.category, ...opts.numeric].map((o) => o.key)
  if (keys.length && !keys.includes(colorBy.value)) colorBy.value = keys[0]
  // Drop a size field that the new dataset doesn't carry.
  if (sizeBy.value && !opts.numeric.some((o) => o.key === sizeBy.value)) sizeBy.value = ''
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

  // Live (in-browser) clusters: values come from the reactive assignment map,
  // not a property. Same stable palette as the pipeline clusters.
  if (key === 'live_cluster') {
    const seen = new Set()
    let hasNull = false
    for (const f of feats) {
      const lab = live.labelFor(f.properties)
      if (hasValue(lab)) seen.add(lab); else hasNull = true
    }
    const legend = [...seen].sort().map((lab) => ({ label: lab, color: categoryColor('live_cluster', lab) }))
    if (hasNull) legend.push({ label: 'Unclustered', color: UNCLUSTERED })
    return { type: 'categorical', title, colorFn: (p) => categoryColor('live_cluster', live.labelFor(p)), legend }
  }

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
  // colours to the distinct values present, most frequent first. The legend is
  // capped (a dataset can have hundreds of species) with a "+N more" row.
  if (CATEGORY_KEYS.has(key)) {
    const counts = new Map()
    for (const f of feats) {
      const v = f.properties[key]
      if (hasValue(v)) counts.set(v, (counts.get(v) || 0) + 1)
    }
    const cats = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v)
    const LEGEND_CAP = 12
    // Stable per-value colours, so a species/year/class matches its colour in
    // the charts. Legend shows the most frequent values first.
    const legend = cats.slice(0, LEGEND_CAP).map((v) => ({ label: String(v), color: categoryColor(key, v) }))
    if (cats.length > LEGEND_CAP) legend.push({ label: `+${cats.length - LEGEND_CAP} more`, color: UNCLUSTERED })
    return {
      type: 'categorical', title,
      colorFn: (p) => (hasValue(p[key]) ? categoryColor(key, p[key]) : UNCLUSTERED),
      legend,
    }
  }

  // Numeric (sequential). Elevation and temperature follow the ft/m and °F/°C
  // settings, so the gradient scale matches the units shown elsewhere.
  const meta = ALL_NUMERIC.find((f) => f.key === key) || {}
  const conv = meta.unit === 'elev' ? elevValue : meta.unit === 'temp' ? tempValue : (v) => Number(v)
  const unitSuffix = meta.unit === 'elev' ? ` (${unit.value})` : meta.unit === 'temp' ? ` (°${tempUnit.value})` : ''
  const vals = feats.map((f) => f.properties[key]).filter(hasValue).map((v) => conv(Number(v)))
  const min = vals.length ? Math.min(...vals) : 0
  const max = vals.length ? Math.max(...vals) : 1
  return {
    type: 'sequential', title: title + unitSuffix, min, max,
    colorFn: (p) => {
      const raw = p[key]
      if (!hasValue(raw)) return UNCLUSTERED
      return hexLerp(RAMP[0], RAMP[1], (conv(Number(raw)) - min) / ((max - min) || 1))
    },
  }
})

// Size points by a numeric field (radius), or a uniform size when "none".
const sizeScale = computed(() => {
  if (!sizeBy.value) return null
  const feats = filteredData.value?.features || []
  const vals = feats.map((f) => f.properties[sizeBy.value]).filter(hasValue).map(Number)
  if (!vals.length) return null
  return { lo: Math.min(...vals), hi: Math.max(...vals) }
})
function radiusFor(props) {
  const s = sizeScale.value
  if (!s) return 6
  const v = props[sizeBy.value]
  if (!hasValue(v)) return 3
  return 4 + 9 * ((Number(v) - s.lo) / ((s.hi - s.lo) || 1)) // 4 … 13
}

// Re-style markers when the colouring or sizing changes.
watch([coloring, sizeScale], ([c]) => {
  if (geoLayer) geoLayer.eachLayer((l) => {
    l.setStyle({ fillColor: c.colorFn(l.feature.properties) })
    l.setRadius(radiusFor(l.feature.properties))
  })
})

// When focusing an observation, the next re-render must not refit/clear it.
let suppressFit = false

// Rebuild the point layer whenever the dataset changes (e.g. species switch).
function renderPoints(geo) {
  if (!map || !L || !geo) return
  if (geoLayer) { geoLayer.remove(); geoLayer = null }
  if (!suppressFit) selected.value = null

  geoLayer = L.geoJSON(geo, {
    pointToLayer: (feature, latlng) => L.circleMarker(latlng, {
      radius: radiusFor(feature.properties), weight: 1, color: '#222',
      fillColor: coloring.value.colorFn(feature.properties), fillOpacity: 0.85,
    }),
    onEachFeature: (feature, lyr) => {
      lyr.bindTooltip(feature.properties.species || 'Observation', { direction: 'top' })
      lyr.on('click', () => {
        selected.value = feature.properties
        const co = feature.geometry?.coordinates
        selectedLatLng.value = co ? [co[1], co[0]] : null
      })
    },
  }).addTo(map)

  const bounds = geoLayer.getBounds()
  // Non-animated: an in-flight fit animation would block a subsequent zoom-in to
  // a focused observation (Leaflet ignores zoom changes mid-animation).
  if (bounds.isValid() && !suppressFit) map.fitBounds(bounds.pad(0.1), { animate: false })
  suppressFit = false // one-shot
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
    selectedLatLng.value = [lat, lon]
    // Zoom in on the observation (not just pan). Stop any in-flight fit-to-data
    // animation first, or it would complete and override this zoom.
    suppressFit = true
    map.setView([lat, lon], 15)
  }
  setFocusObservation(null) // consume so a later revisit doesn't re-trigger
}
watch(focusObservation, (t) => t && applyFocus(t))

// A location pin marks the currently-selected observation (from a click or from
// "Open on map"), and clears when the detail drawer is closed.
function pinIcon() {
  return L.divIcon({
    className: 'obs-pin', iconSize: [28, 40], iconAnchor: [14, 38], tooltipAnchor: [0, -34],
    html: `<svg viewBox="0 0 24 34" width="28" height="40" aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.3 0 11.9 0 20.6 12 34 12 34s12-13.4 12-22.1C24 5.3 18.6 0 12 0z"
            fill="#e34948" stroke="#fff" stroke-width="1.5"/>
      <circle cx="12" cy="12" r="4.5" fill="#fff"/></svg>`,
  })
}
watch(selectedLatLng, (ll) => {
  if (!map || !L) return
  if (selectedMarker) { selectedMarker.remove(); selectedMarker = null }
  if (ll) selectedMarker = L.marker(ll, { icon: pinIcon(), interactive: false, zIndexOffset: 1000 }).addTo(map)
})
// Closing the drawer (selected → null) removes the pin.
watch(selected, (s) => { if (!s) selectedLatLng.value = null })

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

    // Zoom control on the bottom-left so it never overlaps the top-left
    // "Color by" control (previously it clipped the label).
    map = L.map(mapEl.value, { scrollWheelZoom: true, zoomControl: false, layers: [osm] }).setView([39.5, -105.7], 7)
    L.control.zoom({ position: 'bottomleft' }).addTo(map)
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

// Show a dot at the viewer's location (browser geolocation, opt-in per click).
function locateMe() {
  if (!map || !L) return
  if (!('geolocation' in navigator)) {
    locateError.value = 'Geolocation not supported by this browser.'
    return
  }
  locating.value = true
  locateError.value = ''
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      locating.value = false
      const { latitude: lat, longitude: lon, accuracy } = pos.coords
      if (userLayer) { userLayer.remove(); userLayer = null }
      userLayer = L.layerGroup([
        // Accuracy halo + a solid "you are here" dot.
        L.circle([lat, lon], { radius: accuracy || 0, color: '#2a78d6', weight: 1, fillOpacity: 0.12 }),
        L.circleMarker([lat, lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#2a78d6', fillOpacity: 1 })
          .bindTooltip('You are here', { direction: 'top' }),
      ]).addTo(map)
      map.setView([lat, lon], Math.max(map.getZoom() || 0, 11))
    },
    (err) => {
      locating.value = false
      locateError.value = err.code === err.PERMISSION_DENIED
        ? 'Location permission denied.'
        : 'Could not get your location.'
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
  )
}

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

.locate {
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 7px 10px; font: 600 13px system-ui, sans-serif; color: #333; cursor: pointer;
  display: inline-flex; gap: 7px; align-items: center; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.locate:hover { background: #fff; }
.locate.busy { opacity: 0.7; cursor: progress; }
.locate .dot-icon {
  width: 11px; height: 11px; border-radius: 50%; background: #2a78d6; border: 2px solid #fff;
  box-shadow: 0 0 0 1px #2a78d6; flex: 0 0 auto;
}

.legend {
  position: absolute; bottom: 18px; right: 12px; z-index: 500;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 10px 12px; font: 13px/1.4 system-ui, sans-serif; color: #222; min-width: 120px;
  max-width: 46vw; max-height: 44vh; overflow-y: auto;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.legend-title { font-weight: 600; margin-bottom: 6px; position: sticky; top: 0; }
.legend-row { display: flex; align-items: center; gap: 8px; }
.legend-row span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.swatch { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #222; flex: 0 0 auto; }
.gradient { height: 12px; border-radius: 3px; border: 1px solid #ccc; }
.gradient-scale { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; }

/* Mobile: tighten the on-map controls and legend so they don't swallow the map. */
@media (max-width: 640px) {
  .controls { top: 8px; left: 8px; right: 8px; gap: 6px; }
  .colorby, .toggle { padding: 5px 8px; font-size: 12px; }
  .toggle { flex: 1 1 100%; }
  .legend {
    bottom: 8px; right: 8px; left: auto; max-width: 62vw; max-height: 34vh;
    padding: 8px 10px; font-size: 12px;
  }
}

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
.photos { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.obs-photo { max-width: 100%; height: auto; border-radius: 4px; }
.description { margin-top: 8px; white-space: pre-wrap; }

.slide-enter-active, .slide-leave-active { transition: transform 0.2s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }
</style>
