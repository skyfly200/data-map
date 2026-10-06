<template>
  <div class="al">
    <label class="al-row al-main">
      <input v-model="a.enabled.value" type="checkbox" />
      <span>Show access areas</span>
      <span v-if="a.loading.value" class="al-spin" aria-label="Loading access data" role="status"></span>
    </label>

    <template v-if="a.enabled.value">
      <p v-if="a.notice.value" class="al-notice" role="status">{{ a.notice.value }}</p>

      <fieldset class="al-fs">
        <legend>Color by</legend>
        <div class="al-seg">
          <label v-for="o in ACCESS_ATTRS" :key="o.key" class="al-seg-opt" :class="{ on: a.attr.value === o.key }">
            <input v-model="a.attr.value" type="radio" name="access-attr" :value="o.key" />
            <span>{{ o.label }}</span>
          </label>
        </div>
      </fieldset>

      <fieldset class="al-fs">
        <legend>Show only</legend>
        <label class="al-row"><input v-model="a.switches.free" type="checkbox" /><span>Free</span></label>
        <label class="al-row"><input v-model="a.switches.public" type="checkbox" /><span>Public land</span></label>
        <label class="al-row"><input v-model="a.switches.collecting" type="checkbox" /><span>Collecting allowed</span></label>
        <label class="al-row al-sub" :class="{ off: !a.switches.collecting }">
          <input v-model="a.switches.includeLikely" type="checkbox" :disabled="!a.switches.collecting" />
          <span>Include likely allowed (estimated)</span>
        </label>
        <p class="al-hint">Unknown values are hidden by these filters, never assumed free, public or allowed.</p>
      </fieldset>

      <fieldset class="al-fs">
        <legend>Sources</legend>
        <label v-for="s in ACCESS_SOURCES" :key="s.key" class="al-row">
          <input v-model="a.sources[s.key]" type="checkbox" />
          <span>{{ s.label }}</span>
          <small class="al-count">{{ a.counts.value[s.key] }}</small>
        </label>
        <p v-if="!signedIn" class="al-hint">Sign in to see My areas and Club areas.</p>
      </fieldset>

      <div class="al-legend" aria-label="Access legend">
        <div v-for="(it, i) in legend" :key="i" class="al-leg-row">
          <span class="al-sw" :class="{ outline: it.kind === 'outline' }"
                :style="swatch(it)"></span>
          <span>{{ it.label }}</span>
        </div>
      </div>
      <p class="al-hint">{{ DISCLAIMER }}</p>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { ACCESS_ATTRS, ACCESS_SOURCES, legendFor } from '~/composables/accessLayer'
import { DISCLAIMER } from '~/composables/forayPlanner'

// Controls + legend for the map's Access overlay. State lives in
// useMapAccessLayer; this only edits it.
const props = defineProps({
  access: { type: Object, required: true },
  signedIn: { type: Boolean, default: false },
})
const a = props.access
const legend = computed(() => legendFor(a.attr.value))
function swatch(it) {
  return it.kind === 'outline'
    ? { borderColor: it.color, borderStyle: it.dashed ? 'dashed' : 'solid' }
    : { background: it.color, borderStyle: it.dashed ? 'dashed' : 'solid', borderColor: it.color }
}
</script>

<style scoped>
.al { display: grid; gap: 8px; font-size: 0.82rem; min-width: 0; }
.al-row { display: flex; align-items: center; gap: 8px; min-height: 28px; cursor: pointer; line-height: 1.3; }
.al-row input { flex: 0 0 auto; margin: 0; accent-color: var(--accent, #2b7a3d); }
.al-main { font-weight: 600; }
.al-sub { padding-left: 22px; }
.al-sub.off { opacity: 0.55; }
.al-count { margin-left: auto; color: var(--muted, #777); font-variant-numeric: tabular-nums; }
.al-fs { border: 0; padding: 0; margin: 0; min-width: 0; }
.al-fs legend {
  padding: 0; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted, #777); font-weight: 700; margin-bottom: 2px;
}
.al-seg { display: flex; border: 1px solid var(--border, #ddd); border-radius: 6px; overflow: hidden; }
.al-seg-opt {
  flex: 1 1 0; min-width: 0; text-align: center; cursor: pointer; padding: 6px 4px;
  font-size: 0.72rem; border-left: 1px solid var(--border, #ddd); background: var(--surface, #fff);
  color: var(--muted, #666); position: relative;
}
.al-seg-opt:first-child { border-left: 0; }
.al-seg-opt input { position: absolute; opacity: 0; inset: 0; margin: 0; cursor: pointer; }
.al-seg-opt.on { background: var(--accent, #2b7a3d); color: #fff; font-weight: 600; }
.al-seg-opt:focus-within { outline: 2px solid var(--accent, #2b7a3d); outline-offset: -2px; }
.al-notice {
  margin: 0; padding: 6px 8px; border-radius: 6px; font-size: 0.74rem; line-height: 1.35;
  background: #fff4d6; color: #5c4400; border: 1px solid #e8c660;
}
.al-hint { margin: 2px 0 0; color: var(--muted, #777); font-size: 0.72rem; line-height: 1.35; }
.al-legend { display: grid; gap: 4px; }
.al-leg-row { display: flex; align-items: center; gap: 8px; font-size: 0.74rem; }
.al-sw { flex: 0 0 auto; width: 22px; height: 14px; border-radius: 3px; border: 2px solid; opacity: 0.9; }
.al-sw.outline { background: transparent; border-width: 3px; }
@keyframes al-spin { to { transform: rotate(360deg); } }
.al-spin {
  width: 12px; height: 12px; border: 2px solid var(--border, #ccc); border-top-color: var(--accent, #2b7a3d);
  border-radius: 50%; animation: al-spin 0.7s linear infinite;
}

@media (max-width: 720px) {
  .al-row { min-height: 44px; }
  .al-row input { width: 22px; height: 22px; }
  .al-seg-opt { padding: 12px 4px; font-size: 0.78rem; }
}
</style>

<style>
/* Leaflet popups live outside this component's DOM, so these are global. */
.acc-pop { font-size: 0.82rem; line-height: 1.35; max-width: 250px; }
.acc-pop-tag { font-size: 0.72rem; font-weight: 700; color: #6a1b9a; margin: 1px 0 3px; }
.acc-pop dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 8px; margin: 4px 0; }
.acc-pop dt { color: #666; }
.acc-pop dd { margin: 0; font-weight: 600; }
.acc-pop-note { margin: 4px 0 0; font-size: 0.72rem; color: #555; }
</style>
