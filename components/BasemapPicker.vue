<template>
  <div class="bp">
    <label v-for="b in layers" :key="b.key" class="lay-row">
      <input type="radio" :name="group" :value="b.key" :checked="active === b.key"
             @change="$emit('pick', b.key)" />
      <span>{{ b.name }}</span>
    </label>
  </div>
</template>

<script setup>
// The basemap radios, on their own so they can live in two places: a dropdown
// on the control bar on a wide screen, and inside the layer window on a phone,
// where eight buttons across the bar was two too many.

import { useId } from 'vue'

defineProps({
  layers: { type: Array, default: () => [] },
  active: { type: String, default: '' },
})
defineEmits(['pick'])

// A radio group needs a name, and two copies of this on one page sharing one
// name would make the hidden copy's selection fight the visible one's.
const group = `basemap-${useId()}`
</script>

<style scoped>
.bp { display: grid; }
.lay-row {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 2px; font-size: 0.82rem; cursor: pointer; line-height: 1.3;
}
.lay-row:hover { color: var(--text); }
.lay-row input { flex: 0 0 auto; margin: 0; }
.lay-row span { flex: 1 1 auto; }

@media (max-width: 720px) {
  /* Real touch targets where this is a list you tap rather than click. */
  .lay-row { padding: 7px 2px; }
}
</style>
