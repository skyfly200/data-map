<template>
  <v-container class="py-6" v-if="dataset">
    <v-btn variant="text" prepend-icon="mdi-arrow-left" :to="'/datasets'" class="mb-4">
      All Datasets
    </v-btn>

    <v-row>
      <!-- Info card -->
      <v-col cols="12" md="4">
        <v-card class="mb-4">
          <v-card-title>{{ dataset.name }}</v-card-title>
          <v-card-subtitle v-if="dataset.description">{{ dataset.description }}</v-card-subtitle>
          <v-card-text>
            <v-chip :color="statusColor" class="mb-3" label>{{ dataset.status }}</v-chip>
            <div class="text-body-2 text-medium-emphasis mb-2">
              {{ dataset.observation_count || 0 }} observations
            </div>
            <div class="text-body-2 text-medium-emphasis mb-2">
              Created {{ formatDate(dataset.created_at) }}
            </div>
            <v-divider class="my-3" />
            <div class="text-overline mb-1">Pipeline Config</div>
            <div class="text-body-2">
              <strong>Taxon:</strong> {{ dataset.config.taxon_name }}<br>
              <strong>Quality:</strong> {{ dataset.config.quality_grade }}<br>
              <strong>Center:</strong> {{ dataset.config.lat }}, {{ dataset.config.lng }}<br>
              <strong>Radius:</strong> {{ dataset.config.radius }} km<br>
              <strong>Clusters:</strong> {{ dataset.config.n_clusters }}
            </div>
          </v-card-text>
        </v-card>

        <!-- Cluster breakdown -->
        <v-card v-if="clusterCounts && Object.keys(clusterCounts).length">
          <v-card-title class="text-body-1">Clusters</v-card-title>
          <v-card-text class="pa-2">
            <ClusterLegend :cluster-counts="clusterCounts" vertical />
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Map / pipeline status -->
      <v-col cols="12" md="8">
        <PipelineStatus
          v-if="dataset.status !== 'complete'"
          :dataset="dataset"
          @complete="onComplete"
        />

        <v-card v-if="dataset.status === 'complete'">
          <LeafletMap :observations="observations" height="500px" />
        </v-card>
      </v-col>
    </v-row>
  </v-container>

  <v-container v-else class="py-6">
    <v-progress-circular v-if="loading" indeterminate />
    <v-alert v-else type="error">Dataset not found.</v-alert>
  </v-container>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useStore } from 'vuex'
import LeafletMap from '@/components/map/LeafletMap.vue'
import ClusterLegend from '@/components/map/ClusterLegend.vue'
import PipelineStatus from '@/components/datasets/PipelineStatus.vue'

const route = useRoute()
const store = useStore()

const loading = computed(() => store.state.datasets.loading)
const dataset = computed(() => store.getters['datasets/byId'](route.params.id))

const observations = computed(() =>
  store.getters['observations/forDataset'](route.params.id)
)

const clusterCounts = computed(() => {
  const counts = {}
  observations.value.forEach(o => {
    if (o.cluster !== null) counts[o.cluster] = (counts[o.cluster] || 0) + 1
  })
  return counts
})

const statusColor = computed(() => ({
  pending: 'default',
  running: 'warning',
  complete: 'success',
  error: 'error'
}[dataset.value?.status] || 'default'))

function formatDate(iso) {
  return new Date(iso).toLocaleDateString()
}

async function onComplete() {
  await store.dispatch('datasets/fetchDatasets')
  await store.dispatch('observations/fetchObservations', route.params.id)
}

onMounted(async () => {
  await store.dispatch('datasets/fetchDatasets')
  if (dataset.value?.status === 'complete') {
    await store.dispatch('observations/fetchObservations', route.params.id)
  }
})
</script>
