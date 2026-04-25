<template>
  <v-card :to="`/datasets/${dataset.id}`">
    <v-card-title class="d-flex align-center">
      <span class="text-truncate">{{ dataset.name }}</span>
      <v-spacer />
      <v-chip :color="statusColor" size="small" label>{{ dataset.status }}</v-chip>
    </v-card-title>

    <v-card-subtitle v-if="dataset.description" class="pt-1">
      {{ dataset.description }}
    </v-card-subtitle>

    <v-card-text>
      <div class="text-body-2 text-medium-emphasis">
        <v-icon icon="mdi-leaf" size="small" class="mr-1" />
        {{ dataset.config.taxon_name }}
        &nbsp;·&nbsp;
        <v-icon icon="mdi-map-marker-radius" size="small" class="mr-1" />
        {{ dataset.config.radius }} km radius
        &nbsp;·&nbsp;
        <v-icon icon="mdi-eye" size="small" class="mr-1" />
        {{ dataset.observation_count || 0 }} obs
      </div>

      <PipelineStatus
        v-if="dataset.status === 'running'"
        :dataset="liveDataset"
        compact
        class="mt-3"
        @complete="onComplete"
      />
    </v-card-text>

    <v-card-actions>
      <v-spacer />
      <v-btn
        icon="mdi-delete"
        variant="text"
        color="error"
        size="small"
        @click.prevent="confirmDelete"
      />
    </v-card-actions>
  </v-card>

  <v-dialog v-model="deleteDialog" max-width="400">
    <v-card>
      <v-card-title>Delete dataset?</v-card-title>
      <v-card-text>
        This will permanently delete <strong>{{ dataset.name }}</strong> and all its observations.
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn @click="deleteDialog = false">Cancel</v-btn>
        <v-btn color="error" :loading="deleting" @click="doDelete">Delete</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useStore } from 'vuex'
import PipelineStatus from './PipelineStatus.vue'

const props = defineProps({ dataset: { type: Object, required: true } })
const emit = defineEmits(['deleted', 'statusChange'])

const store = useStore()
const deleteDialog = ref(false)
const deleting = ref(false)
const liveDataset = ref(props.dataset)

const statusColor = computed(() => ({
  pending: 'default',
  running: 'warning',
  complete: 'success',
  error: 'error'
}[liveDataset.value.status] || 'default'))

function confirmDelete() { deleteDialog.value = true }

async function doDelete() {
  deleting.value = true
  await store.dispatch('datasets/deleteDataset', props.dataset.id)
  emit('deleted', props.dataset.id)
  deleting.value = false
  deleteDialog.value = false
}

function onComplete(updated) {
  liveDataset.value = updated
  emit('statusChange', updated)
}
</script>
