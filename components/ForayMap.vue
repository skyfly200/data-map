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
  /** Distance origin {lat, lon, label}; drawn as a blue dot with the radius ring. */
  origin: { type: Object, default: null },
  radiusMi: { type: Number, default: 0 },
  /** When true, the next map tap emits `pick` with its [lat, lon] instead of selecting. */
  picking: { type: Boolean, default: false },
})
const emit = defineEmits(['select', 'pick'])

const SQUARES_FROM_ZOOM = 11
const SELECT_ZOOM = 12

const el = ref(null)
let L = null
let map = null
let layer = null
let originLayer = null
let fittedFor = ''

const opacity = (t) => 0.3 + 0.6 * (Number(t) || 0)

// While picking a start point, a tap on a cell or pin picks that spot instead.
function onShapeClick(e, key) {
  L.DomEvent.stopPropagation(e)
  if (props.picking) emit('pick', [e.latlng.lat, e.latlng.lng])
  else emit('select', key)
}

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
    shape.on('click', (e) => onShapeClick(e, c.key))
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
    pin.on('click', (e) => onShapeClick(e, c.key))
    if (c.tip) pin.bindTooltip(`#${i + 1} · ${c.tip}`)
    layer.addLayer(pin)
  })
  layer.addTo(map)
}

function drawOrigin() {
  if (!map || !L) return
  if (originLayer) { originLayer.remove(); originLayer = null }
  const o = props.origin
  if (!o) return
  originLayer = L.layerGroup()
  if (props.radiusMi) {
    originLayer.addLayer(L.circle([o.lat, o.lon], {
      radius: props.radiusMi * 1609.344, color: '#2a78d6', weight: 1.5, dashArray: '6 4', fill: false, interactive: false,
    }))
  }
  originLayer.addLayer(L.circleMarker([o.lat, o.lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#2a78d6', fillOpacity: 1 })
    .bindTooltip(o.label || 'Start', { direction: 'top' }))
  originLayer.addTo(map)
}

// Fit to the ranked places (not every sampled cell) whenever that set changes,
// so the view opens on the places the list is about.
function fit() {
  if (!map || !L) return
  const pts = (props.ranked.length ? props.ranked : props.cells).map((c) => [c.lat, c.lon])
  const o = props.origin
  const sig = pts.map((p) => p.join()).join('|') + (o ? `@${o.lat},${o.lon},${props.radiusMi}` : '')
  if (sig === fittedFor) return
  fittedFor = sig
  if (o) pts.push([o.lat, o.lon])
  // Nothing within reach: show the whole radius ring so the empty area is visible.
  if (o && props.radiusMi && !props.ranked.length) {
    map.fitBounds(L.latLng(o.lat, o.lon).toBounds(props.radiusMi * 1609.344 * 2), { padding: [16, 16] })
    return
  }
  if (!pts.length) return
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
  map.on('click', (e) => { if (props.picking) emit('pick', [e.latlng.lat, e.latlng.lng]) })
  fit()
  draw()
  drawOrigin()
})

watch(() => [props.cells, props.ranked], () => { fit(); draw() })
watch(() => [props.origin, props.radiusMi], () => { drawOrigin(); fit() })
watch(() => props.picking, (on) => { if (el.value) el.value.classList.toggle('picking', on) })

/** Current map centre as [lat, lon], or null before the map exists. */
function getCenter() {
  if (!map) return null
  const c = map.getCenter()
  return [c.lat, c.lng]
}
defineExpose({ getCenter })
watch(() => props.selectedKey, (key) => {
  const c = props.cells.find((x) => x.key === key)
  if (c && map) map.setView([c.lat, c.lon], Math.max(map.getZoom(), SELECT_ZOOM))
  draw()
})

onBeforeUnmount(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
.foray-map { width: 100%; height: 100%; min-height: 280px; background: var(--surface-2); border-radius: 8px; }
.foray-map.picking, .foray-map.picking :deep(.leaflet-interactive) { cursor: crosshair; }
</style>

<style>
.foray-pin {
  display: grid; place-items: center; width: 24px; height: 24px; border-radius: 50%;
  background: #fff; color: #1b3a1f; border: 2px solid #1b5e20; font: 700 12px/1 system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}
.foray-pin.sel { background: #1b5e20; color: #fff; border-color: #fff; transform: scale(1.2); }
</style>
