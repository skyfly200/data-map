<template>
  <v-container class="py-6">
    <div class="d-flex align-center mb-6">
      <h1 class="text-h5 font-weight-bold">Datasets</h1>
      <v-spacer />
      <CreateDatasetDialog @created="onCreated" />
    </div>

    <v-progress-linear v-if="loading" indeterminate class="mb-4" />

    <v-alert v-if="error" type="error" class="mb-4">{{ error }}</v-alert>

    <v-row v-if="datasets.length">
      <v-col v-for="dataset in datasets" :key="dataset.id" cols="12" md="6" lg="4">
        <DatasetCard
          :dataset="dataset"
          @deleted="onDeleted"
          @statusChange="onStatusChange"
        />
      </v-col>
    </v-row>

    <v-empty-state
      v-else-if="!loading"
      icon="mdi-mushroom-outline"
      title="No datasets yet"
      text="Create your first dataset to start collecting and analyzing mushroom observations."
    />
  </v-container>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useStore } from 'vuex'
import DatasetCard from '@/components/datasets/DatasetCard.vue'
import CreateDatasetDialog from '@/components/datasets/CreateDatasetDialog.vue'

const store = useStore()

const datasets = computed(() => store.state.datasets.datasets)
const loading = computed(() => store.state.datasets.loading)
const error = computed(() => store.state.datasets.error)

function onCreated() {
  // Dataset already added to store by createDataset action
}

function onDeleted(id) {
  // Already removed from store by deleteDataset action
}

function onStatusChange(updated) {
  store.commit('datasets/updateDataset', updated)
}

onMounted(() => store.dispatch('datasets/fetchDatasets'))
</script>
