<template>
  <v-card v-if="!compact" variant="outlined">
    <v-card-title class="text-body-1">Pipeline Running</v-card-title>
    <v-card-text>
      <v-stepper :model-value="currentStep" alt-labels flat>
        <v-stepper-header>
          <v-stepper-item
            title="Fetch"
            subtitle="iNaturalist"
            value="1"
            :complete="currentStep > 1"
            :color="currentStep >= 1 ? 'green-darken-3' : undefined"
          />
          <v-divider />
          <v-stepper-item
            title="Enrich"
            subtitle="Weather + elevation"
            value="2"
            :complete="currentStep > 2"
            :color="currentStep >= 2 ? 'green-darken-3' : undefined"
          />
          <v-divider />
          <v-stepper-item
            title="Cluster"
            subtitle="k-means"
            value="3"
            :complete="currentStep > 3"
            :color="currentStep >= 3 ? 'green-darken-3' : undefined"
          />
        </v-stepper-header>
      </v-stepper>

      <v-alert v-if="live.status === 'error'" type="error" class="mt-3">
        {{ live.error_message || 'Pipeline failed.' }}
      </v-alert>
    </v-card-text>
  </v-card>

  <!-- Compact inline version for DatasetCard -->
  <div v-else class="d-flex align-center gap-2">
    <v-progress-linear
      :model-value="progressPct"
      color="green-darken-3"
      height="6"
      rounded
      class="flex-grow-1"
    />
    <span class="text-caption text-medium-emphasis">{{ stageLabel }}</span>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { api } from '@/plugins/api'

const props = defineProps({
  dataset: { type: Object, required: true },
  compact: { type: Boolean, default: false }
})

const emit = defineEmits(['complete'])

const live = ref({ ...props.dataset })

const STAGE_STEP = { idle: 0, inat: 1, enrich: 2, cluster: 3, done: 4, error: 0 }
const STAGE_LABELS = { idle: 'Queued', inat: 'Fetching…', enrich: 'Enriching…', cluster: 'Clustering…', done: 'Done', error: 'Error' }

const currentStep = computed(() => String(STAGE_STEP[live.value.stage] || 0))
const stageLabel = computed(() => STAGE_LABELS[live.value.stage] || live.value.stage)
const progressPct = computed(() => (STAGE_STEP[live.value.stage] || 0) / 3 * 100)

let timer = null

async function poll() {
  try {
    const updated = await api.getDataset(props.dataset.id)
    live.value = updated
    if (updated.status === 'complete' || updated.status === 'error') {
      stopPolling()
      emit('complete', updated)
    }
  } catch {
    // transient error — keep polling
  }
}

function stopPolling() {
  if (timer) { clearInterval(timer); timer = null }
}

onMounted(() => {
  if (live.value.status === 'complete' || live.value.status === 'error') return
  poll()
  timer = setInterval(poll, 2000)
})

onUnmounted(stopPolling)
</script>
