<template>
  <div class="re">
    <div class="re-preview" :style="{ background: gradientCss(stops) }" :title="`${stops.length} stops`"></div>

    <div class="re-stops">
      <!-- One swatch per stop, with the add button in the gap it would fill.
           Putting + between two stops rather than at the end is what makes
           "add a stop here" mean somewhere — a ramp is a sequence, and where a
           colour goes is most of the edit. -->
      <template v-for="(hex, i) in stops" :key="i">
        <span class="re-stop">
          <input type="color" :value="hex" :aria-label="`Stop ${i + 1} of ${stops.length}`"
                 @input="onSet(i, $event.target.value)" />
          <button v-if="stops.length > MIN_STOPS" type="button" class="re-drop"
                  :title="`Remove stop ${i + 1}`" :aria-label="`Remove stop ${i + 1}`"
                  @click="onRemove(i)">×</button>
        </span>
        <button v-if="i < stops.length - 1 && stops.length < MAX_STOPS" type="button"
                class="re-add" :title="`Add a stop between ${i + 1} and ${i + 2}`"
                :aria-label="`Add a stop between ${i + 1} and ${i + 2}`"
                @click="onAdd(i)">+</button>
      </template>
    </div>

    <div class="re-acts">
      <button type="button" class="re-text" title="Flip the ramp end to end"
              @click="onReverse">⇄ Reverse</button>
      <span class="re-count">{{ stops.length }} of {{ MAX_STOPS }}</span>
    </div>
  </div>
</template>

<script setup>
// Editing a colour ramp of any length.
//
// Shared by the heatmap ramp and the point gradient, which were separate
// problems only because one of them did not exist: the heatmap had two fixed
// pickers and points had nothing at all. Both are the same question — what
// colours does this scale run through — so both get the same control.
//
// Every edit goes through the pure helpers in composables/ramps.js and emits a
// whole new ramp. Nothing here mutates what it was given, so the parent decides
// when a change is persisted.

import { computed } from 'vue'
import {
  MAX_STOPS, MIN_STOPS, addStop, gradientCss, normaliseStops, removeStop, reverseStops, setStop,
} from '~/composables/ramps'

const props = defineProps({
  modelValue: { type: Array, default: null },
  // Used when the value is unusable or absent, so the editor never opens on an
  // empty ramp and make somebody build one from nothing.
  fallback: { type: Array, default: () => ['#e8f1fb', '#0b3d91'] },
})
const emit = defineEmits(['update:modelValue'])

const stops = computed(() => normaliseStops(props.modelValue) || normaliseStops(props.fallback)
  || ['#e8f1fb', '#0b3d91'])

const onSet = (i, hex) => emit('update:modelValue', setStop(stops.value, i, hex))
const onAdd = (i) => emit('update:modelValue', addStop(stops.value, i))
const onRemove = (i) => emit('update:modelValue', removeStop(stops.value, i))
const onReverse = () => emit('update:modelValue', reverseStops(stops.value))
</script>

<style scoped>
.re { display: grid; gap: 6px; }

.re-preview {
  height: 14px; border-radius: 4px; border: 1px solid var(--border);
}

.re-stops { display: flex; align-items: center; gap: 2px; flex-wrap: wrap; }

.re-stop { position: relative; display: inline-flex; }
.re-stop input[type="color"] {
  width: 30px; height: 26px; padding: 0; border: 1px solid var(--border);
  border-radius: 4px; background: none; cursor: pointer;
}

/* Sits on the swatch rather than beside it: a row of eight stops each with a
   neighbouring button is a row of sixteen controls, and which × belongs to
   which colour stops being obvious. */
.re-drop {
  position: absolute; top: -5px; right: -5px; width: 14px; height: 14px;
  border-radius: 50%; border: 1px solid var(--border); background: var(--bg);
  color: var(--muted); font-size: 0.6rem; line-height: 1; cursor: pointer; padding: 0;
}
.re-drop:hover { color: #b3492f; border-color: #b3492f; }

.re-add {
  width: 16px; height: 26px; padding: 0; border: 1px dashed var(--border);
  border-radius: 3px; background: none; color: var(--muted); cursor: pointer;
  font-size: 0.75rem; line-height: 1;
}
.re-add:hover { color: var(--text); border-color: var(--muted); }

.re-acts { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.re-text {
  background: none; border: none; padding: 0; cursor: pointer; font: inherit;
  font-size: 0.72rem; color: var(--accent, #3d8b5f); text-decoration: underline;
}
.re-count { font-size: 0.68rem; color: var(--muted); font-variant-numeric: tabular-nums; }
</style>
