<template>
  <NuxtLink v-if="doc" class="help" :to="href" :title="hoverText" :aria-label="`What does ${doc.title} do?`">?</NuxtLink>
</template>

<script setup>
import { computed } from 'vue'
import { docFor, docHref, docSummary } from '~/composables/optionDocs'

// A control's link into the reference. The hover text is the option's own
// one-line summary, so the tooltip and the guide cannot disagree — and the ?
// itself goes to the full entry for the cases a sentence cannot cover.
const props = defineProps({
  // An id from composables/optionDocs.js.
  option: { type: String, required: true },
  // A shortcut key to mention, when the control has one.
  keys: { type: String, default: '' },
})

const doc = computed(() => docFor(props.option))
const href = computed(() => docHref(props.option))

const shortcuts = useShortcuts()
const hoverText = computed(() => {
  const base = `${docSummary(props.option)}  Click for details.`
  return props.keys ? shortcuts.withKey(base, props.keys) : base
})
</script>

<style scoped>
/* Deliberately quiet: one of these sits beside a lot of controls, and a row of
   loud question marks would compete with the controls themselves. */
.help {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; flex: 0 0 auto;
  border: 1px solid var(--border); border-radius: 50%;
  font-size: 9px; font-weight: 700; line-height: 1;
  color: var(--muted); background: transparent; text-decoration: none;
  cursor: help; vertical-align: middle;
}
.help:hover, .help:focus-visible {
  color: var(--accent-ink); background: var(--accent); border-color: var(--accent);
}
</style>
