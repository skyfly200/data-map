<template>
  <div class="step-picker">
    <div class="presets">
      <button type="button" class="chip" :class="{ on: matches(myDefault) }" @click="set(myDefault)">
        {{ isCustom ? 'My default' : 'Standard' }} ({{ myDefault.length }})
      </button>
      <button v-if="isCustom" type="button" class="chip" :class="{ on: matches(standard) }" @click="set(standard)">
        Standard ({{ standard.length }})
      </button>
      <button type="button" class="chip" :class="{ on: matches(allKeys) }" @click="set(allKeys)">
        Everything ({{ allKeys.length }})
      </button>
      <span class="count">{{ modelValue.length }} picked</span>
    </div>

    <fieldset v-for="g in groups" :key="g.group" class="group">
      <legend>{{ g.group }}</legend>
      <label v-for="s in g.steps" :key="s.key" class="step">
        <input type="checkbox" :checked="modelValue.includes(s.key)" @change="toggle(s.key, $event)" />
        <span>
          <strong>{{ s.label }}<small v-if="s.fromMap" class="tag" title="Sampled from the same image this map layer draws">map layer</small></strong>
          <em>{{ s.description }}</em>
        </span>
      </label>
    </fieldset>

    <p class="save">
      <template v-if="matches(myDefault)">
        These are your default steps.
        <button v-if="isCustom" type="button" class="linkish" @click="resetDefault()">Go back to the standard set</button>
      </template>
      <template v-else>
        <button type="button" class="linkish" :disabled="!modelValue.length" @click="saveDefault(modelValue)">
          Make these my default
        </button>
        <span class="muted">so new jobs start with them</span>
      </template>
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { STAGE_KEYS, cleanStages } from '~/netlify/lib/ee-pipeline.mjs'
import { sameSteps } from '~/composables/useEnrichSteps'

const props = defineProps<{ modelValue: string[] }>()
const emit = defineEmits<{ (e: 'update:modelValue', v: string[]): void }>()

const { groups, standard, myDefault, isCustom, saveDefault, resetDefault } = useEnrichSteps()
const allKeys = STAGE_KEYS as string[]

const matches = (list: string[]) => sameSteps(props.modelValue, list)
const set = (list: string[]) => emit('update:modelValue', [...list])

function toggle(key: string, ev: Event) {
  const on = (ev.target as HTMLInputElement).checked
  const next = on ? [...props.modelValue, key] : props.modelValue.filter((k) => k !== key)
  emit('update:modelValue', cleanStages(next))
}
</script>

<style scoped>
.step-picker { display: grid; gap: 10px; flex: 1; min-width: 0; }
.presets { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 999px;
  padding: 4px 10px; font: inherit; font-size: 0.78rem; cursor: pointer; }
.chip.on { border-color: var(--accent, #3d8b5f); color: var(--accent, #3d8b5f); font-weight: 600; }
.count { font-size: 0.75rem; color: var(--muted); margin-left: auto; }
.group { border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px 10px; margin: 0;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; min-width: 0; }
.group legend { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); padding: 0 4px; }
.step { display: flex; gap: 7px; align-items: flex-start; font-size: 0.8rem; min-width: 0; }
.step input { margin-top: 2px; flex: none; }
.step strong { display: block; font-weight: 600; }
.step em { display: block; font-style: normal; color: var(--muted); font-size: 0.75rem; }
.tag { margin-left: 6px; font-size: 0.65rem; font-weight: 500; padding: 1px 5px; border-radius: 4px;
  border: 1px solid var(--border); color: var(--muted); vertical-align: middle; }
.save { margin: 0; font-size: 0.78rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }
.linkish { background: none; border: none; color: var(--accent, #3d8b5f); font: inherit; font-size: 0.8rem;
  cursor: pointer; padding: 0; text-decoration: underline; }
.linkish:disabled { opacity: 0.5; cursor: default; }
.muted { color: var(--muted); }
</style>
