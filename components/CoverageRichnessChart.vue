<template>
  <div class="chart">
    <Vega v-if="spec" :spec="spec" />
    <p v-else>Loading richness data...</p>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { Vega } from 'vue-vega'

const props = defineProps({
  plusCodes: { type: String, default: '' }
})

const spec = ref(null)

async function loadData(codes) {
  if (!codes) { spec.value = null; return }
  const code = codes.split(';')[0].trim()
  try {
    const res = await fetch(`/data/richness/${code}.geojson`)
    if (!res.ok) throw new Error('not found')
    const geo = await res.json()
    // Assume each feature has a numeric "richness" property
    const values = geo.features.map((f, i) => ({ index: i, richness: f.properties.richness }))
    spec.value = {
      $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
      description: 'Species richness heatmap',
      data: { values },
      mark: 'rect',
      encoding: {
        x: { field: 'index', type: 'ordinal', title: '' },
        y: { value: 0, type: 'ordinal' },
        color: { field: 'richness', type: 'quantitative', scale: { scheme: 'viridis' } }
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

