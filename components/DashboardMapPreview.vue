<template>
  <div class="dash-widget map-preview">
    <div class="widget-head">
      <h3 class="widget-title"><span aria-hidden="true">🗺️ </span>Map preview</h3>
      <NuxtLink :to="mapLink" class="widget-link">Full map ›</NuxtLink>
    </div>

    <div ref="container" class="mp-container">
      <div ref="mapEl" class="mp-map"></div>
      <div v-if="dotCount" class="mp-badge" :title="`${dotCount.toLocaleString()} observations shown`">
        {{ fmtCount(dotCount) }}
      </div>
    </div>

    <p class="mp-caption">
      {{ caption }}
      <NuxtLink :to="mapLink" class="mp-open">Open ›</NuxtLink>
    </p>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useFilters } from '~/composables/useFilters'
import { useMaxEnt } from '~/composables/useMaxEnt'

const { filteredData, load, pending } = useObservations()
const { filters } = useFilters()
const { models, fetchModels } = useMaxEnt()

const mapEl = ref(null)
let map = null
let markers = null

// Fixed view on Colorado's Front Range (Colorado Springs to Fort Collins,
// foothills to the plains edge). Fitting to the data zoomed out to everything
// loaded. zoomSnap below lets a short phone-width tile land between whole zoom
// levels instead of dropping back to the whole state.
const FRONT_RANGE = [[38.8, -105.5], [40.6, -104.7]]

const dotCount = computed(() => filteredData.value?.features?.length || 0)

const latestModel = computed(() =>
  [...(models.value || [])]
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))[0] || null
)

const mapLink = computed(() => {
  const m = latestModel.value
  return m ? `/map?layer=maxent:${m.id}` : '/map'
})

const caption = computed(() => {
  const f = filters.value
  const parts = []
  if (f.search) parts.push(`"${f.search}"`)
  if (f.taxon) parts.push(f.taxon)
  if (f.country) parts.push(f.country)
  if (f.state) parts.push(f.state)
  if (f.year) parts.push(f.year)
  return parts.length ? parts.join(' · ') : 'All observations'
})

function fmtCount(n) {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

async function initMap() {
  if (!import.meta.client || !mapEl.value) return
  const L = (await import('leaflet')).default
  await import('leaflet/dist/leaflet.css')

  map = L.map(mapEl.value, {
    zoomControl: false,
    attributionControl: false,
    dragging: false,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    tap: false,
    zoomSnap: 0.25,
  })

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors',
  }).addTo(map)

  markers = L.layerGroup().addTo(map)
  // Guard against zero-size container on first mount.
  try {
    map.invalidateSize({ animate: false })
    map.fitBounds(FRONT_RANGE, { padding: [0, 0] })
  } catch { map.setView([39.7, -105.1], 8) }
  renderDots(L)
}

function renderDots(L) {
  if (!markers || !map) return
  markers.clearLayers()
  // Sample only what falls in the fixed view, so the cap isn't spent off-screen.
  const view = map.getBounds()
  const inView = []
  for (const f of filteredData.value?.features || []) {
    const [lng, lat] = f.geometry?.coordinates || []
    if (lat != null && lng != null && view.contains([lat, lng])) inView.push([lat, lng])
  }
  if (!inView.length) return

  const MAX_DOTS = 2000
  const step = inView.length > MAX_DOTS ? Math.ceil(inView.length / MAX_DOTS) : 1

  for (let i = 0; i < inView.length; i += step) {
    L.circleMarker(inView[i], {
      radius: 3,
      color: '#2a78d6',
      fillColor: '#2a78d6',
      fillOpacity: 0.55,
      weight: 0,
    }).addTo(markers)
  }
}

onMounted(async () => {
  await Promise.all([load(), fetchModels()])
  await initMap()
})

watch(filteredData, async () => {
  if (!map) return
  const L = (await import('leaflet')).default
  renderDots(L)
})

onBeforeUnmount(() => {
  map?.remove()
  map = null
})
</script>

<style scoped>
.map-preview { height: 100%; display: flex; flex-direction: column; gap: 0.4rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }

.mp-container { position: relative; flex: 1; min-height: 180px; border-radius: 8px; overflow: hidden; border: 1px solid var(--border-soft, #eee); }
.mp-map { width: 100%; height: 100%; min-height: 180px; }

.mp-badge {
  position: absolute; bottom: 6px; right: 6px;
  background: rgba(0,0,0,0.55); color: #fff;
  font-size: 0.7rem; border-radius: 999px; padding: 0.1rem 0.45rem;
  font-variant-numeric: tabular-nums; pointer-events: none;
}

.mp-caption {
  margin: 0; font-size: 0.68rem; color: var(--muted, #999);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mp-open { color: var(--accent, #2a78d6); text-decoration: none; margin-left: 0.4rem; }
.mp-open:hover { text-decoration: underline; }
</style>
