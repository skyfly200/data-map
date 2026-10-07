<template>
  <div ref="el" class="foray-map" role="application" aria-label="Foray score map"></div>
</template>

<script setup>
// Leaflet map for the foray planner. Every scored cell is drawn coloured by
// `cell.color`: as a dot when zoomed out (a ~2 km square is a pixel at state
// zoom) and as its square when zoomed in. The ranked places get numbered pins
// that match the list. Click either to select. Client-only (Leaflet needs the DOM).
import { markRaw, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  cells: { type: Array, default: () => [] },
  /** Ranked cells, in list order; numbered pins and the initial fit. */
  ranked: { type: Array, default: () => [] },
  selectedKey: { type: String, default: '' },
})
const emit = defineEmits(['select'])

const SQUARES_FROM_ZOOM = 11
const SELECT_ZOOM = 12

const el = ref(null)
let L = null
let map = null
let layer = null
let fittedFor = ''

const opacity = (t) => 0.3 + 0.6 * (Number(t) || 0)

function draw() {
  if (!map || !L) return
  if (layer) { layer.remove(); layer = null }
  if (!props.cells.length) return
  layer = L.layerGroup()
  const squares = map.getZoom() >= SQUARES_FROM_ZOOM
  for (const c of props.cells) {
    const sel = c.key === props.selectedKey
    const style = {
      color: sel ? '#111' : c.color, weight: sel ? 3 : 1,
      fillColor: c.color, fillOpacity: opacity(c.t),
    }
    const shape = squares
      ? L.polygon(c.polygon, style)
      : L.circleMarker([c.lat, c.lon], { ...style, radius: 3 + 4 * (Number(c.t) || 0) })
    shape.on('click', () => emit('select', c.key))
    if (c.tip) shape.bindTooltip(c.tip, { sticky: true })
    layer.addLayer(shape)
  }
  props.ranked.forEach((c, i) => {
    const sel = c.key === props.selectedKey
    const pin = L.marker([c.lat, c.lon], {
      icon: L.divIcon({ className: '', html: `<span class="foray-pin${sel ? ' sel' : ''}">${i + 1}</span>`, iconSize: [24, 24], iconAnchor: [12, 12] }),
      zIndexOffset: sel ? 1000 : 500 - i,
      keyboard: false,
    })
    pin.on('click', () => emit('select', c.key))
    if (c.tip) pin.bindTooltip(`#${i + 1} · ${c.tip}`)
    layer.addLayer(pin)
  })
  layer.addTo(map)
}

// Fit to the ranked places (not every sampled cell) whenever that set changes,
// so the view opens on the places the list is about.
function fit() {
  if (!map || !L) return
  const pts = (props.ranked.length ? props.ranked : props.cells).map((c) => [c.lat, c.lon])
  const sig = pts.map((p) => p.join()).join('|')
  if (!pts.length || sig === fittedFor) return
  fittedFor = sig
  map.fitBounds(L.latLngBounds(pts), { padding: [32, 32], maxZoom: SELECT_ZOOM })
}

onMounted(async () => {
  await import('leaflet/dist/leaflet.css')
  L = (await import('leaflet')).default
  map = markRaw(L.map(el.value, { zoomControl: true }).setView([39.0, -105.5], 7))
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 18,
  }).addTo(map)
  map.on('zoomend', draw)
  fit()
  draw()
})

watch(() => [props.cells, props.ranked], () => { fit(); draw() })
watch(() => props.selectedKey, (key) => {
  const c = props.cells.find((x) => x.key === key)
  if (c && map) map.setView([c.lat, c.lon], Math.max(map.getZoom(), SELECT_ZOOM))
  draw()
})

onBeforeUnmount(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
.foray-map { width: 100%; height: 100%; min-height: 280px; background: var(--surface-2); border-radius: 8px; }
</style>

<style>
.foray-pin {
  display: grid; place-items: center; width: 24px; height: 24px; border-radius: 50%;
  background: #fff; color: #1b3a1f; border: 2px solid #1b5e20; font: 700 12px/1 system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}
.foray-pin.sel { background: #1b5e20; color: #fff; border-color: #fff; transform: scale(1.2); }
</style>
