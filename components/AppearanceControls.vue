<template>
  <div class="appearance">
    <button class="ap-btn" :class="{ on: open }" :title="`Palette, shapes and point styling${overrideCount ? ` — ${overrideCount} override(s)` : ''}`"
            @click="open = !open">
      <span class="swatches">
        <span v-for="c in activeColors.slice(0, 4)" :key="c" class="mini" :style="{ background: c }"></span>
      </span>
      Appearance<span v-if="overrideCount" class="badge">{{ overrideCount }}</span>
    </button>

    <div v-if="open" class="ap-panel">
      <div class="ap-row">
        <label for="ap-palette">Palette</label>
        <select id="ap-palette" v-model="paletteKey" @change="persist">
          <option v-for="p in PALETTES" :key="p.key" :value="p.key">{{ p.label }}</option>
        </select>
      </div>
      <div class="ap-preview">
        <span v-for="c in activeColors" :key="c" class="mini" :style="{ background: c }" :title="c"></span>
      </div>

      <div class="ap-row">
        <label for="ap-shapes">Shapes</label>
        <select id="ap-shapes" v-model="shapeSetKey" @change="persist">
          <option v-for="s in SHAPE_SETS" :key="s.key" :value="s.key">{{ s.label }}</option>
        </select>
      </div>
      <div class="ap-preview">
        <svg v-for="s in activeShapes" :key="s" class="glyph" viewBox="-7 -7 14 14" :aria-label="s">
          <path :d="shapePath(s, 0, 0, 5)" fill="currentColor" />
        </svg>
      </div>

      <div class="ap-row">
        <label for="ap-radius">Point size <span class="val">{{ pointRadius }}px</span> <HelpLink option="appearance-point-size" /></label>
        <input id="ap-radius" v-model.number="pointRadius" type="range" min="1" max="10" step="0.5" @change="persist" />
      </div>
      <div class="ap-row">
        <label for="ap-opacity">Opacity <span class="val">{{ Math.round(pointOpacity * 100) }}%</span> <HelpLink option="appearance-point-opacity" /></label>
        <input id="ap-opacity" v-model.number="pointOpacity" type="range" min="0.1" max="1" step="0.05" @change="persist" />
      </div>
      <div class="ap-row">
        <label class="ap-check" for="ap-outline"
               title="The dark ring around each dot. It separates overlapping finds, but over a dense patch the rings merge into a grey mass.">
          <input id="ap-outline" v-model="pointOutline" type="checkbox" @change="persist" />
          Outline map dots
        </label>
        <HelpLink option="appearance-point-outline" />
      </div>

      <!-- Per-value overrides for whatever categories are on screen -->
      <template v-if="field && values.length">
        <div class="ap-sub">
          <span>{{ fieldLabel || field }}</span>
          <span class="ap-hint">click a swatch to recolour</span>
        </div>
        <div class="ap-values">
          <div v-for="(v, i) in values.slice(0, VALUE_CAP)" :key="v" class="ap-value">
            <input type="color" class="ap-color" :value="toHex(categoryColor(field, v))"
                   :aria-label="`Colour for ${v}`"
                   @input="setColor(field, v, $event.target.value)" />
            <select class="ap-shape" :value="shapeOverrides[overrideKey(field, v)] || ''"
                    :aria-label="`Shape for ${v}`"
                    @change="$event.target.value ? setShape(field, v, $event.target.value) : clearShape(field, v)">
              <option value="">auto</option>
              <option v-for="s in ALL_SHAPES" :key="s" :value="s">{{ s }}</option>
            </select>
            <span class="ap-label" :title="String(v)">{{ v }}</span>
            <button v-if="hasOverride(field, v)" class="ap-clear" title="Back to automatic"
                    @click="clearColor(field, v); clearShape(field, v)">↺</button>
          </div>
        </div>
        <p v-if="values.length > VALUE_CAP" class="ap-hint">
          Showing the {{ VALUE_CAP }} most common of {{ values.length }}.
        </p>
      </template>

      <button class="ap-reset" @click="reset">Reset appearance</button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { categoryColor } from '~/composables/useObservations'
import { overrideKey, useAppearance } from '~/composables/useAppearance'

const props = defineProps({
  // The category dimension currently being coloured, and the values present in
  // it — supplied by the host view so the override list matches what is on
  // screen rather than every value in the dataset.
  field: { type: String, default: '' },
  fieldLabel: { type: String, default: '' },
  values: { type: Array, default: () => [] },
})

// Long tails (hundreds of species) make the panel unusable; the host passes
// values most-common-first, so the cap keeps the ones worth recolouring.
const VALUE_CAP = 24

const open = ref(false)
const {
  PALETTES, SHAPE_SETS, ALL_SHAPES,
  paletteKey, shapeSetKey, pointRadius, pointOpacity, pointOutline,
  activeColors, activeShapes, shapeOverrides, overrideCount,
  persist, reset, setColor, clearColor, setShape, clearShape, hasOverride,
} = useAppearance()

// <input type="color"> only accepts #rrggbb, so shorthand and named colours
// have to be normalised or the swatch silently shows black.
function toHex(color) {
  if (typeof color !== 'string') return '#000000'
  const c = color.trim()
  if (/^#[0-9a-f]{6}$/i.test(c)) return c
  if (/^#[0-9a-f]{3}$/i.test(c)) return `#${c[1]}${c[1]}${c[2]}${c[2]}${c[3]}${c[3]}`
  return '#000000'
}

// Same generator the scatter charts use, so the preview glyphs match the marks.
function shapePath(shape, cx, cy, r) {
  const a = r * 0.6
  switch (shape) {
    case 'square': return `M${cx - r},${cy - r}h${2 * r}v${2 * r}h${-2 * r}z`
    case 'triangle': return `M${cx},${cy - r}L${cx + r},${cy + r}L${cx - r},${cy + r}z`
    case 'diamond': return `M${cx},${cy - r}L${cx + r},${cy}L${cx},${cy + r}L${cx - r},${cy}z`
    case 'cross': return `M${cx - a},${cy - r}h${2 * a}v${r - a}h${r - a}v${2 * a}h${-(r - a)}v${r - a}h${-2 * a}v${-(r - a)}h${-(r - a)}v${-2 * a}h${r - a}z`
    case 'wye': return `M${cx - a},${cy + r}l${a},${-r}l${-r},${-a}l${a * 0.6},${-a * 0.9}l${r - a * 0.6},${a}l${r - a * 0.6},${-a}l${a * 0.6},${a * 0.9}l${-r},${a}l${a},${r}z`
    default: return `M${cx - r},${cy}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0z`
  }
}
</script>

<style scoped>
.appearance { position: relative; }

.ap-btn {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  border-radius: 6px; padding: 5px 10px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
}
.ap-btn:hover, .ap-btn.on { background: var(--surface-2); }
.swatches { display: inline-flex; gap: 2px; }
.mini { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.badge {
  background: var(--accent); color: var(--accent-ink); border-radius: 8px;
  padding: 0 5px; font-size: 0.7rem; margin-left: 2px;
}

.ap-panel {
  position: absolute; top: calc(100% + 6px); left: 0; z-index: 900;
  width: 300px; max-height: 70vh; overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 12px; font-size: 0.82rem;
}
.ap-row { display: flex; flex-direction: column; gap: 3px; margin-bottom: 8px; }
.ap-row label { color: var(--muted); font-weight: 600; }
.ap-row .val { color: var(--text); font-weight: 400; }
.ap-row select, .ap-row input[type="range"] { width: 100%; }
.ap-row .ap-check { display: flex; align-items: center; gap: 7px; cursor: pointer; }
.ap-row .ap-check input { width: auto; margin: 0; accent-color: var(--accent); }
.ap-row select {
  background: var(--input-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 5px; padding: 4px 6px;
}
.ap-preview { display: flex; flex-wrap: wrap; gap: 3px; margin: -4px 0 10px; color: var(--muted); }
.glyph { width: 14px; height: 14px; }

.ap-sub {
  display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
  border-top: 1px solid var(--border-soft); padding-top: 8px; margin-bottom: 6px; font-weight: 600;
}
.ap-hint { color: var(--muted); font-weight: 400; font-size: 0.74rem; margin: 4px 0 0; }
.ap-values { display: grid; gap: 4px; }
.ap-value { display: grid; grid-template-columns: 26px 62px 1fr 20px; gap: 5px; align-items: center; }
.ap-color {
  width: 26px; height: 22px; padding: 0; border: 1px solid var(--border);
  border-radius: 4px; background: none; cursor: pointer;
}
.ap-shape {
  background: var(--input-bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 4px; font-size: 0.72rem; padding: 2px 3px;
}
.ap-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ap-clear {
  border: 0; background: transparent; color: var(--muted); cursor: pointer;
  font-size: 0.85rem; padding: 0; line-height: 1;
}
.ap-clear:hover { color: var(--text); }

.ap-reset {
  margin-top: 10px; width: 100%; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 6px; padding: 5px; font-size: 0.8rem; cursor: pointer;
}
.ap-reset:hover { background: var(--surface-3); }
</style>
