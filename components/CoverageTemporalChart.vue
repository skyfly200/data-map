<template>
  <div class="chart">
    <Vega v-if="spec" :spec="spec" />
    <p v-else>Loading temporal data...</p>
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
  // Assume semicolon‑separated plus‑codes, take first for demo
  const code = codes.split(';')[0].trim()
  try {
    const res = await fetch(`/data/temporal/${code}.json`)
    if (!res.ok) throw new Error('not found')
    const data = await res.json()
    // Expect data array of {date: 'YYYY‑MM‑DD', value: number}
    spec.value = {
      $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
      description: 'Temporal coverage',
      data: { values: data },
      mark: 'line',
      encoding: {
        x: { field: 'date', type: 'temporal', title: 'Date' },
        y: { field: 'value', type: 'quantitative', title: 'Coverage' }
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

