<template>
  <PopoverMenu icon="⤓" label="Export" title="Download the suitability surface">
    <div class="ex">
      <p class="ex-count">
        <strong>GeoTIFF</strong>
        <span class="ex-sub">one band, suitability 0 to 1, WGS 84</span>
      </p>

      <div class="ex-formats" role="radiogroup" aria-label="Resolution">
        <label v-for="s in options" :key="s.metres" class="ex-fmt" :class="{ off: s.tooBig }">
          <input v-model="scale" type="radio" :value="s.metres" :disabled="s.tooBig" />
          <span>
            <strong>{{ s.label }}</strong>
            <em>{{ s.tooBig ? 'Too large for one download at this size' : s.note }}</em>
          </span>
        </label>
      </div>

      <p class="ex-note">
        Earth Engine draws the file fresh, which can take a minute. The link it
        gives expires, so it is fetched when you press the button.
      </p>
      <p v-if="error" class="ex-error">{{ error }}</p>

      <button class="btn primary wide" :disabled="busy" @click="run">
        {{ busy ? 'Preparing…' : 'Download GeoTIFF' }}
      </button>
    </div>
  </PopoverMenu>
</template>

<script setup>
// The export for a model job. Data jobs use ExportMenu (CSV / GeoJSON); a model
// is a raster, so the only real choice is its resolution.
import { computed, ref } from 'vue'
import {
  MAX_EXPORT_PIXELS, MODEL_EXPORT_SCALES, defaultExportScale, exportPixels,
} from '~/composables/modelExport'

const props = defineProps({
  /** The model's region ({ west, south, east, north }), when known. */
  region: { type: Object, default: null },
  /** (scale) => Promise<string>, resolving to the download URL. */
  getUrl: { type: Function, required: true },
})

const options = computed(() => MODEL_EXPORT_SCALES.map((s) => {
  const px = exportPixels(props.region, s.metres)
  return { ...s, tooBig: px != null && px > MAX_EXPORT_PIXELS }
}))

const scale = ref(defaultExportScale(props.region))
const busy = ref(false)
const error = ref('')

async function run() {
  busy.value = true
  error.value = ''
  try {
    const url = await props.getUrl(scale.value)
    // A plain navigation: Earth Engine serves the file as an attachment, so
    // the page stays where it is.
    const a = document.createElement('a')
    a.href = url
    a.rel = 'noopener'
    document.body.appendChild(a)
    a.click()
    a.remove()
  } catch (e) {
    error.value = e?.message || 'The download could not be prepared.'
  } finally {
    busy.value = false
  }
}
</script>

<style scoped>
.ex { display: grid; gap: 10px; min-width: 240px; max-width: 300px; }
.ex-count { margin: 0; font-size: 0.85rem; }
.ex-sub { display: block; color: var(--muted); font-size: 0.76rem; }
.ex-formats { display: grid; gap: 6px; }
.ex-fmt { display: flex; gap: 7px; align-items: flex-start; font-size: 0.8rem; }
.ex-fmt.off { opacity: 0.55; }
.ex-fmt strong { display: block; font-weight: 600; }
.ex-fmt em { display: block; font-style: normal; color: var(--muted); font-size: 0.74rem; }
.ex-note { margin: 0; font-size: 0.73rem; color: var(--muted); }
.ex-error { margin: 0; font-size: 0.78rem; color: #b3492f; }
.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.82rem; cursor: pointer; }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
.btn.wide { width: 100%; }
</style>
