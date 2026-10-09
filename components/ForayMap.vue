<template>
  <div ref="el" class="foray-map" role="application" aria-label="Foray score map"></div>
</template>

<script setup>
// Leaflet map for the foray planner. Every scored cell is drawn coloured by
// `cell.color`: as a dot when zoomed out (a ~2 km square is a pixel at state
// zoom) and as its square when zoomed in. The ranked places get numbered pins
// that match the list. Click either to select. Client-only (Leaflet needs the DOM).
//
// Optional: a model layer drawn under the cells (`overlay`, a tile template
// from foray-layers), and lettered "where few have looked" places
// (`opportunities`). The map reports its view as `view` so the page can lay
// out candidate cells for exactly what is on screen.
import { markRaw, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  cells: { type: Array, default: () => [] },
  /** Ranked cells, in list order; numbered pins and the initial fit. */
  ranked: { type: Array, default: () => [] },
  selectedKey: { type: String, default: '' },
  /** Tile template for a model layer, drawn under the cells; null for none. */
  overlay: { type: String, default: null },
  overlayOpacity: { type: Number, default: 0.6 },
  /** Under-sampled places, in list order: [{ key, lat, lon, polygon }]. */
  opportunities: { type: Array, default: () => [] },
})
const emit = defineEmits(['select', 'view'])

const SQUARES_FROM_ZOOM = 11
const SELECT_ZOOM = 12

const el = ref(null)
let L = null
let map = null
let layer = null
let overlayLayer = null
let fittedFor = ''

const OPPORTUNITY_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

const opacity = (t) => 0.3 + 0.6 * (Number(t) || 0)

function draw() {
  if (!map || !L) return
  if (layer) { layer.remove(); layer = null }
  if (!props.cells.length && !props.opportunities.length) return
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
  props.opportunities.forEach((c, i) => {
    const sel = c.key === props.selectedKey
    const outline = L.polygon(c.polygon, { color: '#6a1b9a', weight: sel ? 3 : 2, dashArray: '4 3', fill: false })
    outline.on('click', () => emit('select', c.key))
    layer.addLayer(outline)
    const pin = L.marker([c.lat, c.lon], {
      icon: L.divIcon({ className: '', html: `<span class="foray-pin opp${sel ? ' sel' : ''}">${OPPORTUNITY_LETTERS[i] || '?'}</span>`, iconSize: [24, 24], iconAnchor: [12, 12] }),
      zIndexOffset: sel ? 1000 : 300 - i,
      keyboard: false,
    })
    pin.on('click', () => emit('select', c.key))
    pin.bindTooltip(`${OPPORTUNITY_LETTERS[i] || '?'} · few finds, promising habitat`)
    layer.addLayer(pin)
  })
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

function drawOverlay() {
  if (!map || !L) return
  if (overlayLayer) { overlayLayer.remove(); overlayLayer = null }
  if (!props.overlay) return
  overlayLayer = L.tileLayer(props.overlay, { opacity: props.overlayOpacity, maxZoom: 18, zIndex: 2 }).addTo(map)
}

function emitView() {
  if (!map) return
  const b = map.getBounds()
  emit('view', { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth(), zoom: map.getZoom() })
}

onMounted(async () => {
  await import('leaflet/dist/leaflet.css')
  L = (await import('leaflet')).default
  map = markRaw(L.map(el.value, { zoomControl: true }).setView([39.0, -105.5], 7))
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 18,
  }).addTo(map)
  map.on('zoomend', draw)
  map.on('moveend', emitView)
  fit()
  drawOverlay()
  draw()
  emitView()
})

watch(() => [props.overlay, props.overlayOpacity], drawOverlay)
watch(() => props.opportunities, draw)

watch(() => [props.cells, props.ranked], () => { fit(); draw() })
watch(() => props.selectedKey, (key) => {
  const c = props.cells.find((x) => x.key === key) || props.opportunities.find((x) => x.key === key)
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
.foray-pin.opp { color: #4a148c; border-color: #6a1b9a; border-style: dashed; }
.foray-pin.opp.sel { background: #6a1b9a; color: #fff; border-color: #fff; border-style: solid; }
</style>
