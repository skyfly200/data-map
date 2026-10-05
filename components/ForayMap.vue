<template>
  <div ref="el" class="foray-map" role="application" aria-label="Foray score map"></div>
</template>

<script setup>
// Leaflet map for the foray planner: one polygon per scored cell, coloured by
// `cell.color`, click to select. Client-only (Leaflet needs the DOM).
import { markRaw, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  cells: { type: Array, default: () => [] },
  selectedKey: { type: String, default: '' },
  opacity: { type: Number, default: 0.65 },
})
const emit = defineEmits(['select'])

const el = ref(null)
let L = null
let map = null
let layer = null
let fitted = false
const polys = new Map()

function draw() {
  if (!map || !L) return
  if (layer) { layer.remove(); layer = null }
  polys.clear()
  if (!props.cells.length) return
  layer = L.layerGroup()
  for (const c of props.cells) {
    const poly = L.polygon(c.polygon, {
      color: '#111', weight: c.key === props.selectedKey ? 3 : 0.5,
      fillColor: c.color, fillOpacity: props.opacity,
    })
    poly.on('click', () => emit('select', c.key))
    poly.bindTooltip(`${Math.round(c.score * 100)}% foray score · ${c.n} finds`, { sticky: true })
    polys.set(c.key, poly)
    layer.addLayer(poly)
  }
  layer.addTo(map)
  if (!fitted) {
    fitted = true
    map.fitBounds(L.latLngBounds(props.cells.flatMap((c) => c.polygon)), { padding: [24, 24], maxZoom: 11 })
  }
}

onMounted(async () => {
  await import('leaflet/dist/leaflet.css')
  L = (await import('leaflet')).default
  map = markRaw(L.map(el.value, { zoomControl: true }).setView([39.0, -105.5], 7))
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 18,
  }).addTo(map)
  draw()
})

watch(() => [props.cells, props.opacity], draw)
watch(() => props.selectedKey, (key) => {
  draw()
  const c = props.cells.find((x) => x.key === key)
  if (c && map) map.panTo([c.lat, c.lon])
})

onBeforeUnmount(() => { if (map) { map.remove(); map = null } })
</script>

<style scoped>
.foray-map { width: 100%; height: 100%; min-height: 320px; background: var(--surface-2); border-radius: 8px; }
</style>
