<template>
  <div class="area-map-wrap">
    <div ref="el" class="area-map" role="application"
         aria-label="Map of allowed areas. Use the Draw buttons below to add a polygon."></div>
    <div v-if="drawing" class="draw-bar" role="group" aria-label="Drawing tools">
      <span class="draw-hint" aria-live="polite">{{ pts.length }} point{{ pts.length === 1 ? '' : 's' }} — tap the map to add</span>
      <button type="button" class="btn" @click="addCenter">Add centre point</button>
      <button type="button" class="btn" :disabled="!pts.length" @click="undo">Undo</button>
      <button type="button" class="btn primary" :disabled="pts.length < 3" @click="finish">Finish</button>
      <button type="button" class="btn" @click="cancel">Cancel</button>
    </div>
  </div>
</template>

<script setup>
// Leaflet map for allowed areas: shows a set's polygons, click to select, and a
// small click-to-add-vertices drawing mode (no draw plugin). Client-only.
import { markRaw, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  areas: { type: Array, default: () => [] },
  selectedId: { type: String, default: '' },
  drawing: { type: Boolean, default: false },
  label: { type: String, default: 'Your area' },
})
const emit = defineEmits(['select', 'drawn', 'cancel'])

const el = ref(null)
const pts = ref([])
let L = null
let map = null
let group = null
let draft = null
let fittedFor = ''

function style(selected) {
  return { color: selected ? '#e8a33d' : '#3d8b5f', weight: selected ? 3 : 2, fillOpacity: 0.25 }
}

function draw() {
  if (!map || !L) return
  if (group) group.remove()
  group = L.featureGroup()
  for (const a of props.areas) {
    const id = a.properties?.id || a.id
    const layer = L.geoJSON(a.geometry, { style: () => style(id === props.selectedId) })
    layer.bindTooltip(`${a.properties?.name || 'Area'} · ${props.label}`, { sticky: true })
    layer.on('click', () => { if (!props.drawing) emit('select', id) })
    group.addLayer(layer)
  }
  group.addTo(map)
  const key = props.areas.map((a) => a.properties?.id || a.id).join(',')
  if (key !== fittedFor && props.areas.length) {
    fittedFor = key
    map.fitBounds(group.getBounds(), { padding: [24, 24], maxZoom: 14 })
  }
}

function drawDraft() {
  if (!map || !L) return
  if (draft) { draft.remove(); draft = null }
  if (!pts.value.length) return
  draft = (pts.value.length > 2 ? L.polygon(pts.value, { color: '#e8a33d', dashArray: '4' }) : L.polyline(pts.value, { color: '#e8a33d', dashArray: '4' }))
  draft.addTo(map)
}

const addPoint = (ll) => { pts.value = [...pts.value, [ll.lat, ll.lng]]; drawDraft() }
const addCenter = () => map && addPoint(map.getCenter())
const undo = () => { pts.value = pts.value.slice(0, -1); drawDraft() }
function reset() { pts.value = []; drawDraft() }
function finish() { const ring = pts.value; reset(); emit('drawn', ring) }
function cancel() { reset(); emit('cancel') }

onMounted(async () => {
  await import('leaflet/dist/leaflet.css')
  L = (await import('leaflet')).default
  map = markRaw(L.map(el.value, { zoomControl: true }).setView([39.0, -105.5], 7))
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 18,
  }).addTo(map)
  map.on('click', (e) => { if (props.drawing) addPoint(e.latlng) })
  draw()
})
onBeforeUnmount(() => { if (map) { map.remove(); map = null } })

watch(() => [props.areas, props.selectedId], draw)
watch(() => props.drawing, (d) => {
  if (!d) reset()
  if (el.value) el.value.style.cursor = d ? 'crosshair' : ''
})
</script>

<style scoped>
.area-map { height: 360px; border: 1px solid var(--border); border-radius: 10px; }
.draw-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 8px; }
.draw-hint { font-size: 0.82rem; color: var(--muted); flex: 1 1 100%; }
.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 12px; font: inherit; font-size: 0.84rem; cursor: pointer; min-height: 40px; }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
@media (max-width: 600px) { .area-map { height: 300px; } }
</style>
