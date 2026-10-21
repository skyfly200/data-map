<template>
  <div class="predictor-picker">
    <div class="presets">
      <button type="button" class="chip" :class="{ on: matches(standard) }" @click="set(standard)">
        Standard ({{ standard.length }})
      </button>
      <button type="button" class="chip" :class="{ on: matches(allKeys) }" @click="set(allKeys)">
        Everything ({{ allKeys.length }})
      </button>
      <span class="count">{{ modelValue.length }} picked</span>
    </div>

    <fieldset v-for="g in groups" :key="g.group" class="group">
      <legend>{{ g.group }}</legend>
      <label v-for="p in g.items" :key="p.key" class="item">
        <input type="checkbox" :checked="modelValue.includes(p.key)" @change="toggle(p.key, $event)" />
        <span>{{ p.label }}<small v-if="scores && scores[p.key] != null" class="score">{{ scores[p.key].toFixed(1) }}%</small></span>
      </label>
    </fieldset>
  </div>
</template>

<script setup lang="ts">
import { MAXENT_PREDICTORS, PREDICTOR_KEYS, PREDICTOR_GROUPS, DEFAULT_PREDICTORS } from '~/netlify/lib/maxent.mjs'

// One list of layers for a model. Training samples each picked layer at the
// observations and at the background points itself, so nothing here needs an
// enrichment job first; every extra layer is one more image in that sample.
const props = defineProps<{ modelValue: string[], scores?: Record<string, number> | null }>()
const emit = defineEmits<{ (e: 'update:modelValue', v: string[]): void }>()

const standard = DEFAULT_PREDICTORS as string[]
const allKeys = PREDICTOR_KEYS as string[]
const groups = PREDICTOR_GROUPS.map((g: { group: string, keys: string[] }) => ({
  group: g.group,
  items: g.keys.map((key) => ({ key, label: (MAXENT_PREDICTORS as any)[key].label })),
}))

// Registry order, so the same picks are always the same list.
const ordered = (list: string[]) => allKeys.filter((k) => list.includes(k))
const matches = (list: string[]) => {
  const a = ordered(props.modelValue)
  const b = ordered(list)
  return a.length === b.length && a.every((k, i) => k === b[i])
}
const set = (list: string[]) => emit('update:modelValue', [...list])

function toggle(key: string, ev: Event) {
  const on = (ev.target as HTMLInputElement).checked
  emit('update:modelValue', ordered(on ? [...props.modelValue, key] : props.modelValue.filter((k) => k !== key)))
}
</script>

<style scoped>
.predictor-picker { display: grid; gap: 10px; flex: 1; min-width: 0; }
.presets { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 999px;
  padding: 4px 10px; font: inherit; font-size: 0.78rem; cursor: pointer; }
.chip.on { border-color: var(--accent, #3d8b5f); color: var(--accent, #3d8b5f); font-weight: 600; }
.count { font-size: 0.75rem; color: var(--muted); margin-left: auto; }
.group { border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px 10px; margin: 0;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 6px; min-width: 0; }
.group legend { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); padding: 0 4px; }
.item { display: flex; gap: 7px; align-items: flex-start; font-size: 0.8rem; min-width: 0; }
.item input { margin-top: 2px; flex: none; }
.score { margin-left: 6px; font-size: 0.7rem; color: var(--muted); }
</style>
