<template>
  <div class="home-view">
    <div class="map-controls pa-3">
      <v-select
        v-model="selectedDatasetId"
        :items="completedDatasets"
        item-title="name"
        item-value="id"
        label="Dataset"
        density="compact"
        hide-details
        variant="outlined"
        bg-color="white"
        style="max-width: 320px"
        @update:modelValue="onDatasetChange"
      />
      <ClusterLegend v-if="observations.length" :cluster-counts="clusterCounts" class="ml-3" />
    </div>

    <LeafletMap :observations="observations" height="calc(100vh - 64px)" />

    <v-overlay v-if="loading" contained class="align-center justify-center">
      <v-progress-circular indeterminate size="64" />
    </v-overlay>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useStore } from 'vuex'
import LeafletMap from '@/components/map/LeafletMap.vue'
import ClusterLegend from '@/components/map/ClusterLegend.vue'

const store = useStore()

const selectedDatasetId = ref(null)
const loading = computed(() => store.state.observations.loading)

const completedDatasets = computed(() => store.getters['datasets/completedDatasets'])

const observations = computed(() =>
  selectedDatasetId.value
    ? store.getters['observations/forDataset'](selectedDatasetId.value)
    : []
)

const clusterCounts = computed(() => {
  const counts = {}
  observations.value.forEach(o => {
    if (o.cluster !== null) counts[o.cluster] = (counts[o.cluster] || 0) + 1
  })
  return counts
})

async function onDatasetChange(id) {
  if (id) await store.dispatch('observations/fetchObservations', id)
}

onMounted(async () => {
  await store.dispatch('datasets/fetchDatasets')
  if (completedDatasets.value.length) {
    selectedDatasetId.value = completedDatasets.value[0].id
    await store.dispatch('observations/fetchObservations', selectedDatasetId.value)
  }
})
</script>

<style scoped>
.home-view {
  position: relative;
  height: calc(100vh - 64px);
}

.map-controls {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 1000;
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.92);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
</style>
