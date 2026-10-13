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
// out candidate cells for exactly what is on screen. A cell with
// `predicted: true` (graded by the model blend, few or no finds) is drawn
// with a dashed outline and a light fill so it reads as a guess.
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
  /** Distance origin {lat, lon, label}; drawn as a blue dot with the radius ring. */
  origin: { type: Object, default: null },
  radiusMi: { type: Number, default: 0 },
  /** Drive-time area from the origin (GeoJSON MultiPolygon), drawn as a shaded outline; null for none. */
  driveArea: { type: Object, default: null },
  /** When true, the next map tap emits `pick` with its [lat, lon] instead of selecting. */
  picking: { type: Boolean, default: false },
})
const emit = defineEmits(['select', 'view', 'pick'])

const SQUARES_FROM_ZOOM = 11
const SELECT_ZOOM = 12

const el = ref(null)
let L = null
let map = null
let layer = null
let overlayLayer = null
let originLayer = null
let fittedFor = ''

const OPPORTUNITY_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
  if (!props.cells.length && !props.opportunities.length) return
  layer = L.layerGroup()
  const squares = map.getZoom() >= SQUARES_FROM_ZOOM
  for (const c of props.cells) {
    const sel = c.key === props.selectedKey
    const style = {
      color: sel ? '#111' : c.color, weight: sel ? 3 : c.predicted ? 2 : 1,
      fillColor: c.color, fillOpacity: c.predicted ? 0.15 : opacity(c.t),
      ...(c.predicted ? { dashArray: '4 3' } : {}),
    }
    const shape = squares
      ? L.polygon(c.polygon, style)
      : L.circleMarker([c.lat, c.lon], { ...style, radius: 3 + 4 * (Number(c.t) || 0) })
    shape.on('click', (e) => onShapeClick(e, c.key))
    if (c.tip) shape.bindTooltip(c.tip, { sticky: true })
    layer.addLayer(shape)
  }
  props.opportunities.forEach((c, i) => {
    const sel = c.key === props.selectedKey
    const outline = L.polygon(c.polygon, { color: '#6a1b9a', weight: sel ? 3 : 2, dashArray: '4 3', fill: false })
    outline.on('click', (e) => onShapeClick(e, c.key))
    layer.addLayer(outline)
    const pin = L.marker([c.lat, c.lon], {
      icon: L.divIcon({ className: '', html: `<span class="foray-pin opp${sel ? ' sel' : ''}">${OPPORTUNITY_LETTERS[i] || '?'}</span>`, iconSize: [24, 24], iconAnchor: [12, 12] }),
      zIndexOffset: sel ? 1000 : 300 - i,
      keyboard: false,
    })
    pin.on('click', (e) => onShapeClick(e, c.key))
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
  if (props.driveArea?.coordinates?.length) {
    originLayer.addLayer(L.geoJSON({ type: 'Feature', geometry: props.driveArea, properties: {} }, {
      style: { color: '#7b4fd6', weight: 1.5, fillColor: '#7b4fd6', fillOpacity: 0.06 }, interactive: false,
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
  const sig = pts.map((p) => p.join()).join('|') + (o ? `@${o.lat},${o.lon},${props.radiusMi},${props.driveArea ? 'd' : ''}` : '')
  if (sig === fittedFor) return
  fittedFor = sig
  if (o) pts.push([o.lat, o.lon])
  // Nothing within reach: show the drive-time area or radius ring so the empty area is visible.
  if (o && props.driveArea && !props.ranked.length) {
    const b = L.geoJSON(props.driveArea).getBounds()
    if (b.isValid()) { map.fitBounds(b, { padding: [16, 16] }); return }
  }
  if (o && props.radiusMi && !props.ranked.length) {
    map.fitBounds(L.latLng(o.lat, o.lon).toBounds(props.radiusMi * 1609.344 * 2), { padding: [16, 16] })
    return
  }
  if (!pts.length) return
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
  map.on('click', (e) => { if (props.picking) emit('pick', [e.latlng.lat, e.latlng.lng]) })
  fit()
  drawOverlay()
  draw()
  emitView()
  drawOrigin()
})

watch(() => [props.overlay, props.overlayOpacity], drawOverlay)
watch(() => props.opportunities, draw)

watch(() => [props.cells, props.ranked], () => { fit(); draw() })
watch(() => [props.origin, props.radiusMi, props.driveArea], () => { drawOrigin(); fit() })
watch(() => props.picking, (on) => { if (el.value) el.value.classList.toggle('picking', on) })

/** Current map centre as [lat, lon], or null before the map exists. */
function getCenter() {
  if (!map) return null
  const c = map.getCenter()
  return [c.lat, c.lng]
}
defineExpose({ getCenter })
watch(() => props.selectedKey, (key) => {
  const c = props.cells.find((x) => x.key === key) || props.opportunities.find((x) => x.key === key)
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
.foray-pin.opp { color: #4a148c; border-color: #6a1b9a; border-style: dashed; }
.foray-pin.opp.sel { background: #6a1b9a; color: #fff; border-color: #fff; border-style: solid; }
</style>
