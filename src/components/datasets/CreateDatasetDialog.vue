<template>
  <v-dialog v-model="dialog" max-width="560" persistent>
    <template #activator="{ props: activatorProps }">
      <v-btn v-bind="activatorProps" color="green-darken-3" prepend-icon="mdi-plus">
        New Dataset
      </v-btn>
    </template>

    <v-card>
      <v-card-title>New Dataset</v-card-title>

      <v-card-text>
        <v-form ref="form" v-model="valid">
          <v-text-field
            v-model="name"
            label="Name"
            :rules="[v => !!v || 'Required']"
            variant="outlined"
            density="compact"
            class="mb-3"
          />
          <v-text-field
            v-model="description"
            label="Description (optional)"
            variant="outlined"
            density="compact"
            class="mb-4"
          />

          <v-divider class="mb-4" />
          <div class="text-overline mb-3">iNaturalist Query</div>

          <v-row dense>
            <v-col cols="8">
              <v-text-field
                v-model="config.taxon_name"
                label="Taxon name"
                hint="e.g. morchella, amanita, boletus"
                persistent-hint
                variant="outlined"
                density="compact"
              />
            </v-col>
            <v-col cols="4">
              <v-select
                v-model="config.quality_grade"
                label="Quality"
                :items="['research', 'needs_id', 'casual']"
                variant="outlined"
                density="compact"
              />
            </v-col>
          </v-row>

          <v-row dense class="mt-2">
            <v-col cols="6">
              <v-text-field
                v-model.number="config.lat"
                label="Center latitude"
                type="number"
                variant="outlined"
                density="compact"
              />
            </v-col>
            <v-col cols="6">
              <v-text-field
                v-model.number="config.lng"
                label="Center longitude"
                type="number"
                variant="outlined"
                density="compact"
              />
            </v-col>
          </v-row>

          <div class="text-body-2 mb-1 mt-2">Radius: {{ config.radius }} km</div>
          <v-slider
            v-model="config.radius"
            min="10"
            max="1000"
            step="10"
            thumb-label
            hide-details
            color="green-darken-3"
            class="mb-3"
          />

          <div class="text-body-2 mb-1">Max observations per page: {{ config.per_page }}</div>
          <v-slider
            v-model="config.per_page"
            min="50"
            max="200"
            step="50"
            thumb-label
            hide-details
            color="green-darken-3"
            class="mb-3"
          />

          <div class="text-body-2 mb-1">Clusters (k): {{ config.n_clusters }}</div>
          <v-slider
            v-model="config.n_clusters"
            min="2"
            max="10"
            step="1"
            thumb-label
            hide-details
            color="green-darken-3"
          />
        </v-form>
      </v-card-text>

      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="dialog = false">Cancel</v-btn>
        <v-btn
          color="green-darken-3"
          :loading="loading"
          :disabled="!valid"
          @click="submit"
        >
          Create & Run
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useStore } from 'vuex'

const emit = defineEmits(['created'])
const store = useStore()

const dialog = ref(false)
const form = ref(null)
const valid = ref(false)
const loading = ref(false)

const name = ref('')
const description = ref('')
const config = reactive({
  taxon_name: 'morchella',
  quality_grade: 'research',
  lat: 39.5,
  lng: -105.5,
  radius: 500,
  per_page: 200,
  n_clusters: 4
})

async function submit() {
  const { valid: ok } = await form.value.validate()
  if (!ok) return
  loading.value = true
  try {
    const dataset = await store.dispatch('datasets/createDataset', {
      name: name.value,
      description: description.value,
      config: { ...config }
    })
    emit('created', dataset)
    dialog.value = false
    name.value = ''
    description.value = ''
  } finally {
    loading.value = false
  }
}
</script>
