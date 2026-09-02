<template>
  <div class="appearance">
    <button class="ap-btn" :class="{ on: open, 'icon-only': iconOnly }"
            :title="`Palette, point and overlay styling${overrideCount ? ` — ${overrideCount} override(s)` : ''}`"
            :aria-label="iconOnly ? 'Style' : null" @click="open = !open">
      <!-- The swatches are the icon: they say what the palette currently is,
           which a paint-pot glyph could not. -->
      <span class="swatches">
        <span v-for="c in activeColors.slice(0, 4)" :key="c" class="mini" :style="{ background: c }"></span>
      </span>
      <template v-if="!iconOnly">Style</template>
      <span v-if="overrideCount" class="badge">{{ overrideCount }}</span>
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

      <!-- Overlay ramp: the grid overlays have their own colours, and the key
           on the map reads from the same place. -->
      <div class="ap-row">
        <label for="ap-ramp">Overlay colours <HelpLink option="appearance-overlay-ramp" /></label>
        <select id="ap-ramp" v-model="overlayRampKey" @change="onRampChange">
          <option v-for="r in RAMP_PRESETS" :key="r.key" :value="r.key">{{ r.label }}</option>
          <option value="custom">Custom…</option>
        </select>
      </div>
      <div class="ap-ramp-preview" :style="{ background: `linear-gradient(90deg, ${rampPreview[0]}, ${rampPreview[1]})` }"></div>
      <div v-if="overlayRampKey === 'custom'" class="ap-ramp-pick">
        <label>
          Low
          <input type="color" :value="rampPreview[0]" aria-label="Low end of the overlay ramp"
                 @input="setRampEnd(0, $event.target.value)" />
        </label>
        <label>
          High
          <input type="color" :value="rampPreview[1]" aria-label="High end of the overlay ramp"
                 @input="setRampEnd(1, $event.target.value)" />
        </label>
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
            <span class="ap-label" :title="String(v)">{{ v }}</span>
            <button v-if="hasOverride(field, v)" class="ap-clear" title="Back to automatic"
                    @click="clearColor(field, v); clearShape(field, v)">↺</button>
          </div>
        </div>
        <p v-if="values.length > VALUE_CAP" class="ap-hint">
          Showing the {{ VALUE_CAP }} most common of {{ values.length }}.
        </p>
      </template>

      <div class="ap-buttons">
        <button class="ap-shuffle" title="Deal the palette out differently. Same colours, different assignment — for when two species land on shades you cannot tell apart."
                @click="shuffleColors">🎨 Shuffle colours</button>
        <button class="ap-reset" @click="reset">Reset</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { categoryColor } from '~/composables/useObservations'
import { overrideKey, useAppearance } from '~/composables/useAppearance'

const props = defineProps({
  // The category dimension currently being coloured, and the values present in
  // it — supplied by the host view so the override list matches what is on
  // screen rather than every value in the dataset.
  field: { type: String, default: '' },
  fieldLabel: { type: String, default: '' },
  values: { type: Array, default: () => [] },
  // Icon-only, for the map's control bar.
  iconOnly: { type: Boolean, default: false },
})

// Long tails (hundreds of species) make the panel unusable; the host passes
// values most-common-first, so the cap keeps the ones worth recolouring.
const VALUE_CAP = 24

// The grid overlays draw from their own ramp, chosen here and keyed on the map.
const overlays = useMapOverlays()
const { RAMP_PRESETS, overlayRampKey, overlayRampCustom, rampFor } = overlays
// Previewed against the density ramp, which is the one a reader meets first.
const rampPreview = computed(() => rampFor('density'))

function onRampChange() {
  // Seed a custom pair from whatever was on screen, so the pickers do not open
  // on black and force the viewer to rebuild a ramp from nothing.
  if (overlayRampKey.value === 'custom' && !overlayRampCustom.value) {
    overlayRampCustom.value = [...rampPreview.value]
  }
  overlays.persist()
}

function setRampEnd(i, hex) {
  const next = [...(overlayRampCustom.value || rampPreview.value)]
  next[i] = hex
  overlayRampCustom.value = next
  overlays.persist()
}

const open = ref(false)
// Shapes are no longer offered here: colour already carries category identity
// on this map, and a second encoding of the same thing added a control without
// adding information. The shape encoding still exists for charts that ask for it
// explicitly — it just runs on its defaults rather than being configured here.
const {
  PALETTES,
  paletteKey, pointRadius, pointOpacity, pointOutline,
  activeColors, overrideCount,
  persist, reset, shuffleColors, setColor, clearColor, clearShape, hasOverride,
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

</script>

<style scoped>
.appearance { position: relative; }

.ap-btn {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  border-radius: 6px; padding: 5px 10px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
}
.ap-btn:hover, .ap-btn.on { background: var(--surface-2); }
.ap-btn.icon-only { width: 34px; height: 34px; padding: 0; justify-content: center; }
.ap-btn.icon-only .swatches { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; }
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


.ap-ramp-preview { height: 12px; border-radius: 3px; border: 1px solid var(--border); margin: -4px 0 8px; }
.ap-ramp-pick { display: flex; gap: 12px; margin-bottom: 10px; }
.ap-ramp-pick label { display: flex; align-items: center; gap: 6px; color: var(--muted); font-weight: 600; }
.ap-ramp-pick input { width: 34px; height: 24px; padding: 0; border: 1px solid var(--border); border-radius: 4px; background: none; }
.ap-buttons { display: flex; gap: 8px; margin-top: 10px; }
.ap-shuffle {
  flex: 1 1 auto; border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 6px; padding: 7px 10px; font-size: 0.8rem; font-weight: 600; cursor: pointer;
}
.ap-shuffle:hover { background: var(--surface-3); }
.ap-reset {
  flex: 0 0 auto; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 6px; padding: 7px 12px; font-size: 0.8rem; cursor: pointer;
}
.ap-reset:hover { background: var(--surface-3); }
</style>
