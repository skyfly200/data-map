<template>
  <div class="tri" role="radiogroup" :aria-label="label" :style="{ '--n': options.length, '--i': index }">
    <span class="tri-thumb" aria-hidden="true"></span>
    <button v-for="(o, i) in options" :key="o.value" type="button" role="radio" class="tri-opt"
            :class="{ on: i === index }" :aria-checked="i === index" :tabindex="i === index ? 0 : -1" :title="o.title || o.label"
            @click="choose(i)" @keydown="onKey($event, i)">{{ o.label }}</button>
  </div>
</template>

<script setup>
// A three-position switch (works for any small number of options): a sliding
// thumb under the chosen label. Arrow keys move it, as in a radio group.
import { computed } from 'vue'

const props = defineProps({
  modelValue: { type: [String, Number, Boolean], default: null },
  /** [{ value, label, title? }] */
  options: { type: Array, required: true },
  label: { type: String, default: '' },
})
const emit = defineEmits(['update:modelValue'])

const index = computed(() => Math.max(0, props.options.findIndex((o) => o.value === props.modelValue)))

function choose(i) {
  const o = props.options[i]
  if (o) emit('update:modelValue', o.value)
}
function onKey(e, i) {
  const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0
  if (!step) return
  e.preventDefault()
  const next = (i + step + props.options.length) % props.options.length
  choose(next)
  e.currentTarget.parentElement?.querySelectorAll('.tri-opt')[next]?.focus()
}
</script>

<style scoped>
.tri {
  position: relative; display: inline-grid; grid-template-columns: repeat(var(--n), 1fr);
  border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); padding: 2px; min-width: 0;
}
.tri-thumb {
  position: absolute; top: 2px; bottom: 2px; left: 2px; width: calc((100% - 4px) / var(--n));
  transform: translateX(calc(100% * var(--i))); border-radius: 999px; background: var(--accent);
  transition: transform 0.18s ease; pointer-events: none;
}
.tri-opt {
  position: relative; z-index: 1; font: inherit; font-size: 0.78rem; padding: 0.3rem 0.7rem; border: 0; background: none;
  color: var(--text); cursor: pointer; border-radius: 999px; white-space: nowrap;
}
.tri-opt.on { color: var(--accent-ink, #fff); font-weight: 600; }
.tri-opt:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
@media (prefers-reduced-motion: reduce) { .tri-thumb { transition: none; } }
</style>
