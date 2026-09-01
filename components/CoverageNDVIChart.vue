<template>
  <div class="chart">
    <Vega v-if="spec" :spec="spec" />
    <p v-else>Loading NDVI data...</p>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import vueVega from 'vue-vega'
const { Vega } = vueVega

const props = defineProps({
  plusCodes: { type: String, default: '' }
})

const spec = ref(null)

async function loadData(codes) {
  if (!codes) { spec.value = null; return }
  const code = codes.split(';')[0].trim()
  try {
    const res = await fetch(`/data/ndvi/${code}.geojson`)
    if (!res.ok) throw new Error('not found')
    const geo = await res.json()
    // Assume geojson features have a property "ndvi" numeric.
    const values = geo.features.map(f => ({ ndvi: f.properties.ndvi }))
    spec.value = {
      $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
      description: 'NDVI heatmap',
      data: { values },
      mark: 'rect',
      encoding: {
        // Simple heatmap by index (x) and feature index (y)
        x: { field: 'index', type: 'ordinal', title: '' },
        y: { field: 'index2', type: 'ordinal', title: '' },
        color: { field: 'ndvi', type: 'quantitative', scale: { scheme: 'green' } }
      }
    }
  } catch (e) {
    spec.value = null
  }
}

watch(() => props.plusCodes, (newVal) => {
  loadData(newVal)
}, { immediate: true })
</script>

<style scoped>
.chart { min-height: 200px; }
</style>

