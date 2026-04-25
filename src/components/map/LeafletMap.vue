<template>
  <div ref="mapContainer" :style="{ height, width: '100%' }" />
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import L from 'leaflet'

const props = defineProps({
  observations: { type: Array, default: () => [] },
  height: { type: String, default: '400px' }
})

const CLUSTER_COLORS = [
  '#e41a1c', // red
  '#377eb8', // blue
  '#4daf4a', // green
  '#984ea3', // purple
  '#ff7f00', // orange
  '#a65628', // brown
  '#f781bf', // pink
  '#999999'  // grey (unclustered)
]

const mapContainer = ref(null)
let map = null
let markersLayer = null

function buildMarkers() {
  if (!map) return
  if (markersLayer) {
    markersLayer.clearLayers()
  }

  const circles = props.observations.map(obs => {
    const color = obs.cluster !== null && obs.cluster !== undefined
      ? CLUSTER_COLORS[obs.cluster % CLUSTER_COLORS.length]
      : CLUSTER_COLORS[CLUSTER_COLORS.length - 1]

    const popup = [
      `<strong>${obs.taxon_name || 'Unknown'}</strong>`,
      obs.observed_on ? `Date: ${obs.observed_on}` : '',
      obs.place_guess ? obs.place_guess : '',
      obs.elevation ? `Elevation: ${Math.round(obs.elevation)} m` : '',
      obs.ndvi ? `NDVI: ${obs.ndvi.toFixed(3)}` : '',
      obs.cluster !== null ? `Cluster: ${obs.cluster}` : ''
    ].filter(Boolean).join('<br>')

    return L.circleMarker([obs.lat, obs.lon], {
      radius: 7,
      fillColor: color,
      color: '#ffffff',
      weight: 1.5,
      opacity: 1,
      fillOpacity: 0.85
    }).bindPopup(popup)
  })

  markersLayer = L.layerGroup(circles).addTo(map)

  if (circles.length > 0) {
    const bounds = L.latLngBounds(props.observations.map(o => [o.lat, o.lon]))
    map.fitBounds(bounds, { padding: [30, 30], maxZoom: 12 })
  }
}

onMounted(() => {
  map = L.map(mapContainer.value).setView([39.5, -105.5], 6)

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
  }).addTo(map)

  buildMarkers()
})

onUnmounted(() => {
  if (map) {
    map.remove()
    map = null
  }
})

watch(() => props.observations, buildMarkers, { deep: true })
</script>
