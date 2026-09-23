<template>
  <div class="map-shell" :class="{ 'drawer-open': selected }">
    <div ref="mapEl" class="map"></div>

    <div v-if="loadError" class="overlay error">{{ loadError }}</div>
    <div v-else-if="!loaded" class="overlay">Loading observations…</div>

    <!-- A first-run tour of the map, shown once the observations are in so it
         introduces a working map rather than a loading one. Self-gates on
         localStorage; dispatch 'map-tour-open' to replay it. -->
    <MapTour v-if="loaded" />

    <!-- Over the map rather than modal, because whether a layer is worth having
         on is a question you answer by looking at the map. -->
    <LayerManager
      :open="showLayers" :groups="overlayGroups" :active="activeOverlays"
      :order="overlayOrder" :opacity="layerOpacity"
      :blend="layerBlend" :stack-blend="stackBlend" :solo="soloKey"
      @blend="setLayerBlend" @solo="setSolo"
      @toggle="toggleOverlayByKey" @opacity="setLayerOpacity" @move="moveOverlay"
      @clear="clearOverlays" @close="showLayers = false"
    >
      <!-- On a phone this window is the whole of "what the map is made of":
           the ground underneath, the layers over it, and the grid over those.
           On a wide screen each of those keeps its own button on the bar. -->
      <template v-if="compact" #top>
        <details class="lm-extra">
          <summary>Basemap <em>{{ activeBaseName }}</em></summary>
          <BasemapPicker :layers="baseLayers" :active="activeBase" @pick="setBase" />
        </details>
        <details class="lm-extra" :open="!!heatmapMode">
          <summary>Heatmap <em>{{ heatmapMode ? heatmapMeta.label : 'None' }}</em></summary>
          <HeatmapControls :tip-text="heatmapTip" />
        </details>
      </template>
    </LayerManager>

    <!-- What the map says about one spot you picked, rather than about a
         record someone else made. The observations answer "what was found
         here"; this answers "what is this place like", which is the question
         you have when scouting ground nobody has posted from. -->
    <div v-if="pin" class="pin-panel">
      <div class="pin-head">
        <strong>Dropped point</strong>
        <button class="pin-x" title="Remove this point" @click="clearPin">×</button>
      </div>
      <button class="pin-coords" :title="copied ? 'Copied' : 'Copy coordinates'" @click="copyPin">
        {{ pin.lat.toFixed(5) }}, {{ pin.lon.toFixed(5) }}
        <span class="pin-copy">{{ copied ? 'copied' : 'copy' }}</span>
      </button>
      <dl class="pin-facts">
        <div>
          <dt>Plus code</dt>
          <dd>
            <button class="pin-mini" :title="copied ? 'Copied' : 'Copy plus code'"
                    @click="copyText(pinPlusCode)">{{ pinPlusCode }}</button>
          </dd>
        </div>
        <div>
          <dt>Elevation</dt>
          <dd>{{ pinElevationText }}</dd>
        </div>
        <div v-if="pinCell">
          <dt>{{ heatmapMeta.label }}</dt>
          <dd>{{ pinCellValue }}</dd>
        </div>
        <div v-if="pinCell">
          <dt>Finds in this cell</dt>
          <dd>{{ pinCell.n.toLocaleString() }}</dd>
        </div>
        <div v-if="pinNearest">
          <dt>Nearest find</dt>
          <dd>{{ pinNearest.label }}</dd>
        </div>
      </dl>
      <p v-if="!pinCell && heatmapMode" class="pin-note">
        No cell here. The heatmap is built from observations, so ground nobody
        has recorded from is blank rather than zero.
      </p>
      <p v-if="!pinNearest" class="pin-note">No loaded observations nearby.</p>

      <!-- What the active Earth Engine layers say at this exact spot, read on
           demand because each layer is one Earth Engine call. -->
      <div v-if="activeEeLayers.length" class="pin-sample">
        <button class="pin-sample-btn" :disabled="pinSampling" @click="samplePinLayers">
          {{ pinSampling ? 'Sampling…' : (pinSamples ? 'Sample again' : 'Sample layers here') }}
        </button>
        <dl v-if="pinSamples && pinSamples.length" class="pin-facts sampled">
          <div v-for="s in pinSamples" :key="s.key">
            <dt>{{ s.name }}</dt>
            <dd>
              <span v-if="s.color" class="pin-sw" :style="{ background: s.color }"></span>{{ sampleText(s) }}
            </dd>
          </div>
        </dl>
        <p v-if="pinSampleError" class="pin-note">{{ pinSampleError }}</p>
      </div>
    </div>

    <!-- Thematic layer selector -->
    <div v-if="loaded" ref="controlsEl" class="controls">
      <!-- What the map is made of comes first: the ground, what is drawn on it,
           and the grid over that. Then how the observations are drawn. Then the
           things you set once and leave — style, sharing, settings — which were
           in the middle of the bar and are the least reached for. -->

      <!-- Basemap and overlays are two different questions and were one
           control. The basemap is a single choice from five, made rarely and
           never revisited; the overlays are dozens, toggled constantly, and the
           thing you actually manage. On a phone that distinction is not worth a
           button, so both move inside the layer window. -->
      <PopoverMenu v-if="!compact" icon="◱" label="Basemap" title="The map underneath everything"
                   :badge="activeBaseName">
        <BasemapPicker :layers="baseLayers" :active="activeBase" @pick="setBase" />
      </PopoverMenu>

      <!-- A window rather than a dropdown; see components/LayerManager.vue. -->
      <button class="tool-btn" :class="{ on: showLayers || activeOverlays.size > 0 }"
              :aria-expanded="String(showLayers)"
              :title="tip(compact ? 'Basemap, layers and heatmap' : 'Manage the overlay layers', 'shift+L')"
              @click="showLayers = !showLayers">
        <span class="tool-icon" aria-hidden="true">≣</span>
        <span class="tool-label">Layers</span>
        <span v-if="activeOverlays.size" class="tool-badge">{{ activeOverlays.size }}</span>
      </button>

      <PopoverMenu v-if="!compact" ref="heatmapPop" icon="▦" label="Heatmap"
                   title="Grid summary drawn under the points"
                   :active="!!heatmapMode" :badge="heatmapMode ? heatmapMeta.label : ''">
        <HeatmapControls :tip-text="heatmapTip" />
      </PopoverMenu>

      <!-- How the points are drawn, behind one button. Two labelled selects
           side by side were the widest things on the bar and wrapped it to a
           second row on anything narrower than a laptop. The badge keeps the
           current answer visible, so the common case never needs opening. -->
      <PopoverMenu icon="🎨" label="Points" title="How the points are drawn"
                   :badge="coloring.title">
        <div class="pop-field">
          <label for="colorby-sel">Color by <HelpLink option="map-color-by" /></label>
          <select id="colorby-sel" v-model="colorBy">
            <optgroup label="Category">
              <option v-for="o in colorOptions.category" :key="o.key" :value="o.key">{{ o.label }}</option>
            </optgroup>
            <optgroup v-if="colorOptions.numeric.length" label="Numeric">
              <option v-for="o in colorOptions.numeric" :key="o.key" :value="o.key">{{ o.label }}</option>
            </optgroup>
          </select>
          <p v-if="colorCoverageNote" class="pop-note warn">{{ colorCoverageNote }}</p>
        </div>
        <div v-if="colorOptions.numeric.length" class="pop-field">
          <label for="sizeby-sel">Size by <HelpLink option="map-size-by" /></label>
          <select id="sizeby-sel" v-model="sizeBy">
            <option value="">Uniform</option>
            <option v-for="o in colorOptions.numeric" :key="o.key" :value="o.key">{{ o.label }}</option>
          </select>
        </div>
        <!-- Clustering is another way of colouring the same dots, so on a phone
             it belongs with them rather than beside them. -->
        <template v-if="compact">
          <div class="pop-sep"></div>
          <LiveClusterControls inline />
        </template>
      </PopoverMenu>

      <!-- Which observations, beside how they are drawn. The full set of
           filters stays on the Data tab; the three you reach for while looking
           at the map are here, writing to the same state. -->
      <MapFilters />

      <LiveClusterControls v-if="!compact" />

      <AppearanceControls icon-only :field="colorBy" :field-label="coloring.title"
                          :values="legendValues" />
      <ShareMenu icon-only :map-view="mapView" :color-by="colorBy" :size-by="sizeBy"
                 :title="shareTitle">
        <template #actions>
          <button :disabled="saving" :title="saveError || tip('Save the map, basemap and all, as a PNG', 'e')"
                  @click="saveMap">
            {{ saving ? 'Saving…' : 'Save this map as a PNG' }}
          </button>
        </template>
      </ShareMenu>
      <!-- Everything you do not reach for every minute — the points toggle, the
           excluded-rows option and the offline saves — lives behind one button.
           Spread across the bar they covered the map they were controlling. -->
      <MapSettings v-model="showPoints" :bounds="viewBounds" :sources="activeTileTemplates"
                   :dataset-label="datasetLabel" />
    </div>

    <!-- Both legends share one column, so they cannot overlap each other or the
         control bar, and neither needs to know how tall the other is. -->
    <div v-if="loaded" class="legends" :class="{ collapsed: keyCollapsed }">
      <!-- The key can be in the way as easily as it can be wanted — it is the
           tallest thing on the map when several layers are on. This folds it to
           its header and remembers the choice. -->
      <button class="key-toggle" :aria-expanded="String(!keyCollapsed)"
              :title="keyCollapsed ? 'Show the map key' : 'Hide the map key'"
              @click="setKeyCollapsed(!keyCollapsed)">
        <span class="caret" aria-hidden="true">{{ keyCollapsed ? '▸' : '▾' }}</span>
        Key
      </button>
    <!-- A reference layer that will not load looks the same as one reporting
         empty ground, so it says so instead. -->
    <div v-if="tileErrors.length" class="legend tile-warn">
      <div class="legend-title">Layer unavailable</div>
      <div class="legend-note">
        {{ tileErrors.join(', ') }} could not be reached. Treat it as no data, not as
        empty ground.
      </div>
    </div>
    <!-- An Earth Engine layer that failed to render says why, by name. Blank
         ground on a fire map reads as ground that never burned, so a silent
         failure here would be worse than no layer at all. -->
    <div v-for="e in eeErrors" :key="e.key" class="legend tile-warn">
      <div class="legend-title">{{ e.name }} could not be rendered</div>
      <div class="legend-note">{{ e.message }}</div>
    </div>
    <!-- One key for the whole reference stack, a section per layer that is
         switched on. A raster nobody can read is decoration, so every measured
         layer carries a key — but as separate cards, three of them squeezed
         each other and the heatmap key into strips too short to read. They are
         one stack of layers; they get one panel. -->
    <div v-if="activeTileNotes.length" class="legend tile-note">
      <div class="legend-kind">Map layers</div>
      <div v-for="n in activeTileNotes" :key="n.name" class="tk">
        <div class="tk-name">{{ n.name }}</div>
        <template v-if="n.legend?.type === 'ramp'">
          <div class="gradient" :style="{ background: gradientCss(n.legend.stops) }"></div>
          <!-- A cyclic ramp (aspect) labels evenly all the way round rather than
               just at its ends, so east and west are marked, not just north. -->
          <div v-if="n.legend.ticks" class="gradient-ticks">
            <span v-for="(t, ti) in n.legend.ticks" :key="ti">{{ t }}</span>
          </div>
          <div v-else class="gradient-scale">
            <span>{{ n.legend.min }}</span>
            <span class="unit">{{ n.legend.unit }}</span>
            <span>{{ n.legend.max }}</span>
          </div>
        </template>
        <div v-else-if="n.legend?.type === 'classes' && !n.legendInBrowser" class="class-key">
          <span v-for="c in n.legend.items" :key="c.label" class="ck">
            <span class="swatch" :style="{ background: c.color }"></span>{{ c.label }}
          </span>
        </div>
        <!-- A layer with more classes than a key can hold gets a browser for
             them instead: search, what each one means, and where to read more.
             Four hundred swatches is not a key, it is a lookup table. -->
        <SoilTaxonomyKey v-if="n.classes === 'great-groups'" :layer="n.ee"
                         :selectable="!!n.eeParams?.codes"
                         :codes="(eeParams[n.ee] || {}).codes ?? (n.eeParams?.codes?.default || '')"
                         @update:codes="setEeParam(n.ee, 'codes', $event)" />
        <!-- The date the layer is showing, movable for the ones that vary. Each
             product has its own latency, so "today" is usually blank tiles. -->
        <!-- The knobs an Earth Engine layer exposes. Which year, how far back
             to look: these change what is rendered, so they re-mint the tiles. -->
        <!-- Every parameter except the ones a dedicated control already owns.
             A list of class codes is not a number and not a dropdown; the class
             browser above is its editor, and a second one here would be a text
             field asking somebody to type four hundred numbers. -->
        <div v-for="(p, name) in (n.eeParams || {})" :key="name" class="layer-date"
             v-show="p.type !== 'codes'">
          <label :for="`ee-${n.slug}-${name}`">{{ p.label }}</label>
          <!-- A fixed set of choices (a season, say) is a dropdown; anything
               numeric is a stepper. Both re-mint the tiles on change. -->
          <select v-if="p.type === 'enum'" :id="`ee-${n.slug}-${name}`"
                  :value="(eeParams[n.ee] || {})[name] ?? p.default"
                  @change="setEeParam(n.ee, name, $event.target.value)">
            <option v-for="v in (p.values || [])" :key="v" :value="v">{{ v }}</option>
          </select>
          <input v-else-if="p.type === 'text'" :id="`ee-${n.slug}-${name}`" type="search"
                 :maxlength="p.maxLength || 60" :placeholder="p.default"
                 :value="(eeParams[n.ee] || {})[name] ?? p.default"
                 @change="setEeParam(n.ee, name, $event.target.value)" />
          <input v-else :id="`ee-${n.slug}-${name}`" type="number" :min="p.min" :max="p.max"
                 :value="(eeParams[n.ee] || {})[name] ?? p.default"
                 @change="setEeParam(n.ee, name, Number($event.target.value))" />
        </div>
        <div v-if="n.slow" class="legend-note">
          Computed as you look at it, so tiles arrive slowly the first time.
        </div>
        <div v-if="n.time" class="layer-date">
          <label :for="`ld-${n.slug}`">Date</label>
          <input :id="`ld-${n.slug}`" v-model="tileDate" type="date" :max="maxTileDate"
                 :title="`Which day of ${n.name} to draw. Satellite products lag by days, so recent dates can be blank.`" />
        </div>
        <!-- Past its native level the layer is being stretched, not resolved
             finer. Rainfall at 10 km does not become 30 m detail by zooming,
             and a blurry square that looks like data is worse than a caption. -->
        <div v-if="upscaleNote(n)" class="legend-note upscaled">{{ upscaleNote(n) }}</div>
        <div v-if="n.note" class="legend-note">{{ n.note }}</div>
      </div>
    </div>
    <!-- Heatmap key, with the caveat that belongs with each metric -->
    <!-- A heatmap that produced nothing has to say so. Drawing an empty map and
         leaving the viewer to work out whether the field is missing, the filter
         is too tight, or the feature is broken is what made this read as
         broken — the answer is usually that the column is simply not in the
         data yet. -->
    <!-- A model's suitability surface, handed over from the jobs page. Drawn as
         a tile overlay; this is its key, plus how old it is and whether its
         (expiring) tiles have started to fail. -->
    <div v-if="modelOverlay" class="legend overlay-legend">
      <div class="legend-title">{{ modelOverlay.label }} · suitability</div>
      <div class="gradient" :style="{ background: gradientCss(modelOverlay.legend.stops) }"></div>
      <div class="gradient-scale">
        <span>{{ modelOverlay.legend.min || '0' }} low</span>
        <span>high {{ modelOverlay.legend.max || '1' }}</span>
      </div>
      <div class="legend-note">
        Modelled habitat suitability, not observations.
        <template v-if="modelOverlay.age"> Fitted {{ modelOverlay.age }}.</template>
      </div>
      <div class="legend-note">
        <template v-if="modelOverlay.cv">
          <strong>AUC {{ modelOverlay.cv.auc.toFixed(2) }}</strong> ({{ modelOverlay.cv.grade }}) ·
          {{ modelOverlay.cv.folds }}-fold spatial CV, ±{{ modelOverlay.cv.sd.toFixed(2) }}
        </template>
        <template v-else>Not cross-validated — too few observations to score.</template>
      </div>
      <!-- Observer-effort bias is always present in presence-only models: the
           surface reflects where recorders went as much as where the species
           lives. Effort-weighted background (target-group background) reduces
           this by sampling the contrast against where recording happened, rather
           than against a uniform random background, but it does not remove it. -->
      <div class="legend-note">
        <template v-if="modelOverlay.effortWeighted">
          Background effort-weighted · observer-effort bias reduced but not removed — people record where people go.
        </template>
        <template v-else>
          Uniform background · observer-effort bias: where few records exist may look unsuitable regardless of habitat.
        </template>
      </div>
      <div v-if="modelOverlay.stale" class="legend-note warn">
        These tiles have stopped loading — the fitted surface expires.
        <button v-if="modelOverlay.jobId" class="linkish" :disabled="modelOverlay.refreshing"
                @click="refreshModelOverlay">
          {{ modelOverlay.refreshing ? 'Refreshing…' : 'Refresh from the saved model' }}
        </button>
      </div>
      <button class="linkish" @click="removeModelOverlay">Remove this surface</button>
    </div>

    <div v-if="!heatmapLegend && heatmapMode && loaded" class="legend overlay-legend">
      <div class="legend-title">{{ heatmapMeta.label }}</div>
      <div class="legend-note">{{ emptyHeatmapReason }}</div>
    </div>

    <div v-if="heatmapLegend" class="legend overlay-legend">
      <div class="legend-title">{{ heatmapMeta.label }}</div>
      <template v-if="heatmapLegend.type === 'sequential'">
        <div class="gradient" :style="{ background: gradientCss(heatmapLegend.ramp) }"></div>
        <div class="gradient-scale"><span>{{ heatmapLegend.min }}</span><span>{{ heatmapLegend.max }}</span></div>
        <div class="legend-note">{{ heatmapLegend.cells.toLocaleString() }} cells · {{ heatmapLegend.note }}</div>
      </template>
      <template v-else-if="heatmapLegend.type === 'vector'">
        <div class="gradient" :style="{ background: gradientCss(heatmapLegend.ramp) }"></div>
        <div class="gradient-scale"><span>{{ heatmapLegend.min }}</span><span>{{ heatmapLegend.max }}</span></div>
        <div class="legend-note">
          Source: <strong>{{ heatmapLegend.source }}</strong> · color = {{ heatmapLegend.colorBy }}<br />
          {{ heatmapLegend.cells.toLocaleString() }} arrows · {{ heatmapLegend.note }}
        </div>
      </template>
      <!-- Direction is circular, so its key is a compass rather than a bar:
           a low-to-high gradient would put 359° and 1° at opposite ends. -->
      <template v-else-if="heatmapLegend.type === 'compass'">
        <div class="compass-key">
          <span v-for="item in heatmapLegend.items" :key="item.label" class="ck">
            <span class="swatch" :style="{ background: item.color }"></span>{{ item.label }}
          </span>
        </div>
        <div class="legend-note">{{ heatmapLegend.cells.toLocaleString() }} cells · {{ heatmapLegend.note }}</div>
      </template>
      <template v-else>
        <div v-for="item in heatmapLegend.items" :key="item.label" class="legend-row hoverable"
             :class="{ dim: hoverValue && hoverValue !== item.label }"
             @pointerenter="hoverEnter(item.label, $event)" @pointerleave="hoverLeave($event)"
             @pointerup="pickValue(item.label, $event)">
          <span class="swatch" :style="{ background: item.color }"></span>
          <span><em>{{ item.label }}</em> <span class="legend-n">{{ item.n }}</span></span>
        </div>
        <div class="legend-note">
          {{ heatmapLegend.total }} values appear somewhere · {{ heatmapLegend.note }}
        </div>
      </template>
    </div>

    <!-- The observation key. Titled "Observations" and drawn with round
         swatches, because the layer key sits directly above it with square
         ones: two keys of identical shape, one over the other, left it to the
         viewer to work out which described the dots and which the ground. -->
    <div v-if="coloring" class="legend points-legend" @mouseleave="hoverValue = null">
      <div class="legend-kind">Observations</div>
      <div class="legend-title">{{ coloring.title }}</div>
      <template v-if="coloring.type === 'categorical'">
        <!-- Hovering a row picks out the marks it stands for. A legend of twenty
             species otherwise leaves you matching hues by eye. -->
        <div v-for="item in coloring.legend" :key="item.label" class="legend-row hoverable"
             :class="{ dim: hoverValue && hoverValue !== item.label }"
             @pointerenter="hoverEnter(item.label, $event)"
             @pointerup="pickValue(item.label, $event)">
          <span class="swatch dot" :style="{ background: item.color }"></span>
          <span>{{ item.label }}</span>
        </div>
      </template>
      <template v-else>
        <div class="gradient" :style="{ background: gradientCss(coloring.stops) }"></div>
        <div class="gradient-scale"><span>{{ fmtNum(coloring.min) }}</span><span>{{ fmtNum(coloring.max) }}</span></div>
      </template>
      <!-- Says whether these shades mean the same numbers as the layer's or
           only the same ranking. -->
      <div v-if="coloring.match" class="legend-note match">{{ coloring.match }}</div>
    </div>
    </div>

    <!-- The same drawer the charts and analysis pages use, so the two cannot
         drift apart on what an observation is worth showing. `inline` keeps it
         inside the map shell rather than pinned over the site header. -->
    <ObservationDrawer inline :selected="selected" :show-map-link="false"
                       @close="selected = null" />
  </div>
</template>

<script setup>
import 'leaflet/dist/leaflet.css'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { PALETTE, UNCLUSTERED, categoryColor, colorFor, hasValue, useObservations } from '~/composables/useObservations'
import { classColorFor, fraction, matchNote, paletteFor, rampColor } from '~/composables/fieldPalettes'
import { gradientCss, normaliseStops } from '~/composables/ramps'
import { drawnKeys, effectiveBlend, reorderStack } from '~/composables/blendModes'
import { normaliseCodes } from '~/netlify/lib/ee-tile-layers.mjs'
import { RAMP_PRESETS } from '~/composables/useMapHeatmaps'
import { ALL_CATEGORY, ALL_NUMERIC } from '~/composables/useChartFields'
import { coverageNote } from '~/composables/fieldCoverage'
import { fieldValue } from '~/composables/statistics'
import { useAppearance } from '~/composables/useAppearance'
import { useUnits } from '~/composables/useUnits'

const {
  data, filteredData, load, loadProgressive, chunks, partial,
  speciesFilter, focusObservation, setFocusObservation,
} = useObservations()
const { elevLabel, elevValue, tempValue, unit, tempUnit } = useUnits()
const live = useLiveClusters()
// On a phone the basemap, the heatmap and the clustering controls move inside
// the two windows that remain, rather than being three more buttons on a bar
// that already filled the width of the screen.
const { compact } = useCompactMap()
const appearance = useAppearance()
// How stacked layers combine when nobody has set a layer by hand. A preference,
// so it is read from Appearance rather than kept here.
const { stackBlend } = appearance

/**
 * The ramp the viewer has chosen for numeric point colouring, or null for
 * "whatever suits the field".
 *
 * Null is the default and means the field decides: one a layer also draws
 * borrows that layer's palette, anything else takes the app's own ramp.
 */
const chosenPointRamp = computed(() => {
  const key = appearance.pointRampKey.value
  if (key === 'auto') return null
  if (key === 'custom') return normaliseStops(appearance.pointRampCustom.value)
  return RAMP_PRESETS.find((p) => p.key === key)?.ramp || null
})
const share = useShareState()
const { pointRadius, pointOpacity, pointOutline, colorSeed, activeColors, colorOverrides } = appearance

const mapEl = ref(null)

// Leaflet parks its own controls in the map's corners, and on a phone the
// control bar is tall enough (three wrapped rows, more with the season sliders
// open) that the top-right corner lands inside it — the basemap button sat
// directly on the "Size by" dropdown. The bar's height is not a constant we can
// hard-code, so it is measured and published as a variable the stylesheet offsets
// against.
const controlsEl = ref(null)
let controlsResize = null
let mapResize = null

function trackControlsHeight() {
  if (!import.meta.client || !controlsEl.value) return
  const shell = controlsEl.value.parentElement
  const apply = () => {
    const h = Math.round(controlsEl.value?.getBoundingClientRect().height || 0)
    shell?.style.setProperty('--controls-h', `${h}px`)
  }
  apply()
  controlsResize = new ResizeObserver(apply)
  controlsResize.observe(controlsEl.value)
}
const loaded = ref(false)
const loadError = ref('')
// Remember the "Color by" dimension per viewer.
const COLORBY_KEY = 'map-color-by'
const colorBy = ref('cluster')
if (import.meta.client) {
  const saved = localStorage.getItem(COLORBY_KEY)
  if (saved) colorBy.value = saved
}
watch(colorBy, (v) => { if (import.meta.client) localStorage.setItem(COLORBY_KEY, v) })
// Remember the "Size by" dimension per viewer.
const SIZEBY_KEY = 'map-size-by'
const sizeBy = ref('')
if (import.meta.client) {
  const saved = localStorage.getItem(SIZEBY_KEY)
  if (saved !== null) sizeBy.value = saved
}
watch(sizeBy, (v) => { if (import.meta.client) localStorage.setItem(SIZEBY_KEY, v) })
const selected = ref(null)
const selectedLatLng = ref(null)
// The locate button lives in a Leaflet control rather than the Vue template, so
// its busy state is applied by hand. One class on one element is a smaller cost
// than teleporting a component into a control container.
let locateBtn = null
const locating = ref(false)
const locateError = ref('')
watch(locating, (v) => { if (locateBtn) locateBtn.classList.toggle('busy', v) })
watch(locateError, (msg) => { if (locateBtn && msg) locateBtn.title = msg })
let map, geoLayer, L, userLayer, selectedMarker

// Holds enriched observation info (photos, description, etc.) fetched from iNaturalist API

// ─── Heatmaps ─────────────────────────────────────────────────────────────────
// Grid summaries computed from the observations and drawn under the points:
// density, species richness, seasonal activity, an in-season hotspot score,
// the most common species and land cover, and a cell mean of any enriched
// field — rainfall, soil moisture, NDVI, slope, aspect, TWI, sun and wind
// exposure. See composables/useMapHeatmaps.js for what each one means.
const heatmaps = useMapHeatmaps()
const {
  mode: heatmapMode, cellSize: heatmapCell, cellShape, seasonDay, seasonWindow,
  activeMode: heatmapMeta, groupedModes, heatmapOpacity, tileOpacity, CELL_SIZES,
} = heatmaps
let heatmapLayer = null

const heatmapResult = computed(() =>
  heatmaps.computeHeatmap(filteredData.value?.features || [], heatmapMode.value))
const heatmapLegend = computed(() => heatmapResult.value.legend)

/**
 * Why a heatmap came out empty, distinguishing the three reasons.
 *
 * "Not in the data yet" is by far the most common and the least guessable: the
 * pipeline's NDVI and vegetation-moisture stages have not populated the shipped
 * dataset, so those modes have nothing to draw however far you zoom. Saying
 * that is the difference between a known gap and an apparently broken feature.
 */
const emptyHeatmapReason = computed(() => {
  const field = heatmapMeta.value?.field
  const feats = filteredData.value?.features || []
  if (!feats.length) return 'No observations match the current filters.'
  if (!field) return 'Nothing to show for the current filters.'

  const present = feats.some((f) => Number.isFinite(fieldValue(f.properties || {}, field)))
  if (!present) {
    // Label as written, not lowercased: NDVI and TWI are acronyms and "no ndvi
    // values" reads like a typo.
    return `No ${heatmapMeta.value.label} values in this dataset. `
      + 'The pipeline has not filled this column in yet, so it will stay blank until it is re-run.'
  }
  return 'No cells at this zoom. Zoom in, or widen the filters.'
})

/**
 * One arrow for a vector cell: a shaft plus two barbs, as canvas polylines.
 *
 * Directions are in compass space (dx east, dy north), so the shaft is drawn in
 * degrees with the longitude step divided by cos(lat) — otherwise every arrow
 * would skew east as you move away from the equator.
 */
function arrowFor(c) {
  const span = (c.lat1 - c.lat0) * 0.42          // keep arrows inside their cell
  const len = span * (0.35 + 0.65 * (c.t ?? 0.5))
  const kx = 1 / Math.max(0.2, Math.cos((c.lat * Math.PI) / 180))
  const tipLat = c.lat + c.dy * len
  const tipLon = c.lon + c.dx * len * kx
  const tailLat = c.lat - c.dy * len
  const tailLon = c.lon - c.dx * len * kx

  // Barbs at ±150° from the shaft direction, a third of its length.
  const barb = len * 0.38
  const head = (deg) => {
    const a = Math.atan2(c.dx, c.dy) + (deg * Math.PI) / 180
    return [tipLat - Math.cos(a) * barb, tipLon - Math.sin(a) * barb * kx]
  }
  const style = { color: c.color, weight: 1.6, opacity: 0.9, interactive: false }
  return [
    L.polyline([[tailLat, tailLon], [tipLat, tipLon]], style),
    L.polyline([head(-28), [tipLat, tipLon], head(28)], style),
  ]
}

function renderHeatmap() {
  if (!map || !L) return
  if (heatmapLayer) { heatmapLayer.remove(); heatmapLayer = null }
  const { cells } = heatmapResult.value
  if (!cells.length) return

  const shapes = heatmapResult.value.legend?.type === 'vector'
    ? cells.flatMap((c) => arrowFor(c))
    // Polygons go through the map's canvas renderer, so a few thousand cells
    // cost one canvas rather than a few thousand DOM nodes. The grid hands over
    // an outline whichever shape it is binning into, so this does not care.
    : cells.map((c) => L.polygon(c.polygon, {
      stroke: false, fillColor: c.color, fillOpacity: heatmapOpacity.value, interactive: false,
    }))

  heatmapLayer = L.layerGroup(shapes)
  heatmapLayer.addTo(map)
  // Keep the observation points on top of the shading.
  if (geoLayer) geoLayer.bringToFront()
}

watch(heatmapResult, () => renderHeatmap())
// Opacity is a redraw rather than a recompute: the cells are unchanged, only
// how hard they sit on the basemap.
watch(heatmapOpacity, () => renderHeatmap())
watch([heatmapMode, heatmapCell, cellShape, seasonDay, seasonWindow], () => heatmaps.persist())

// ─── Reference tile layers ────────────────────────────────────────────────────
// Public raster services stacked over the basemap — relief, rainfall, land
// cover, greenness, soil moisture, trails, ownership. The catalogue and its
// keys live in composables/mapLayers.js; this wires them to Leaflet.
//
// Distinct from the Heatmap picker in the control bar, which bins the
// observations themselves. A layer covers the whole map because somebody else
// measured it everywhere; a heatmap covers only where people have looked.

// How far in the map will go. Every tile layer is given this as its maxZoom
// and its own tile ceiling as maxNativeZoom, so the map's limit is a decision
// made here rather than an accident of whichever basemap happens to be on.
//
// 19 is roughly individual-tree scale, which is the scale a foray is planned
// at: "the north side of that draw" is a question about tens of metres. Past
// its native level a layer is upscaled, and upscaleNote below says so.
const MAP_MAX_ZOOM = 19

/**
 * Warn when a visible layer has run out of real tiles.
 *
 * Leaflet upscales past maxNativeZoom, which is what keeps these layers on
 * screen at all — but an upscaled tile looks like a measurement at that scale
 * and is not one. Rainfall sampled at 10 km does not resolve to 30 m because
 * the map was zoomed; it just gets blockier.
 */
function upscaleNote(n) {
  const zoom = mapView.value?.zoom
  if (!n?.native || !Number.isFinite(zoom) || zoom <= n.native) return ''
  return `Zoomed past this layer's detail — the tiles are stretched from zoom ${n.native}, not resolved finer.`
}

// Which day the time-varying layers draw. Defaults to the shortest lag in the
// catalogue, so switching one on lands on a date that exists rather than on
// today, which for an 8-day composite is always blank.
const DEFAULT_LAG = Math.min(...TILE_LAYERS.filter((l) => l.time).map((l) => l.lag ?? 1))
const tileDate = ref(layerDate(DEFAULT_LAG))
// Nothing is published for tomorrow, so the picker will not offer it.
const maxTileDate = layerDate(0)

// Reference layers that could not be reached. Shown rather than swallowed:
// an empty ownership layer reads as "no public land here".
// The season sliders collapse by default: their summary says what they are set
// to, so the bar stays one row until you actually want to move them.
// Whether the map key is folded away. Remembered, because it is a standing
// preference about screen space rather than a per-visit decision.
const KEY_COLLAPSED = 'map-key-collapsed'
const keyCollapsed = ref(false)
function setKeyCollapsed(v) {
  keyCollapsed.value = v
  try { localStorage.setItem(KEY_COLLAPSED, v ? '1' : '0') } catch { /* private mode */ }
}
onMounted(() => {
  // Read on the client only: the server has no localStorage, and rendering the
  // key expanded there and collapsed here is a hydration mismatch.
  try { keyCollapsed.value = localStorage.getItem(KEY_COLLAPSED) === '1' } catch { /* ignore */ }
})

// The heatmap popover, so the keyboard shortcut can still reach the season
// controls now that they live inside it.
const heatmapPop = ref(null)

// The legend value under the cursor. Everything not matching it is faded on the
// map, so a row in the key and the marks it stands for can be seen together.
const hoverValue = ref(null)

// Hover is a mouse idea, and highlighting a legend row has to work without one.
//
// A touch screen synthesises a mouseenter on tap but never a matching
// mouseleave, so a tapped row isolated a category and left the reader with a
// faded map and no way to clear it. Adding a click handler on top made it
// worse: the synthesised enter set the value and the click immediately toggled
// it back off, so tapping did nothing at all.
//
// Pointer events carry the device that raised them, so each gets what suits it
// — transient hover on a mouse, a sticky toggle on a finger.
function hoverEnter(label, e) { if (e.pointerType === 'mouse') hoverValue.value = label }
function hoverLeave(e) { if (!e || e.pointerType === 'mouse') hoverValue.value = null }
function pickValue(label, e) {
  if (e && e.pointerType === 'mouse') return
  hoverValue.value = hoverValue.value === label ? null : label
}
const tileErrors = ref([])
// The caveat belonging to whichever reference layers are switched on.
const activeTileNotes = ref([])
// The built tile layers, so the opacity slider can reach them after setup.
const tileLayers = []

// The fallback ramp, for a numeric field no layer draws — elevation, day of
// year, rainfall. Anything a layer DOES draw borrows that layer's palette
// instead; see composables/fieldPalettes.js.
const RAMP = ['#e8f1fb', '#0b3d91']

// Field labels + which keys are categorical, drawn from the shared chart
// registry so the map and the Explore builder stay in sync.
const CATEGORY_KEYS = new Set(ALL_CATEGORY.map((f) => f.key))
const FIELD_LABEL = Object.fromEntries([...ALL_CATEGORY, ...ALL_NUMERIC].map((f) => [f.key, f.label]))

// Offer only the dimensions that actually carry data in the current dataset,
// so an un-enriched layer (e.g. NDVI still empty) doesn't yield an all-gray map.
const colorOptions = computed(() => {
  const feats = filteredData.value?.features || []
  const present = (list) => list.filter((f) => (
    f.key === 'live_cluster' ? live.active.value : feats.some((ft) => hasValue(ft.properties[f.key]))
  ))
  return { category: present(ALL_CATEGORY), numeric: present(ALL_NUMERIC) }
})

// When the points are coloured by a taxonomic rank that most records lack, the
// map is showing a slice — say so, rather than let a key of five families read
// as the whole dataset.
const colorCoverageNote = computed(() => coverageNote(filteredData.value?.features || [], colorBy.value))

// If a dataset switch drops the active dimension's data, fall back to the first
// option still available (cluster, in practice).
watch(colorOptions, (opts) => {
  const keys = [...opts.category, ...opts.numeric].map((o) => o.key)
  if (keys.length && !keys.includes(colorBy.value)) colorBy.value = keys[0]
  // Drop a size field that the new dataset doesn't carry.
  if (sizeBy.value && !opts.numeric.some((o) => o.key === sizeBy.value)) sizeBy.value = ''
})

function fmtNum(v) { return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2) }


// Build the color function + legend for the current "color by" dimension.
const coloring = computed(() => {
  const feats = filteredData.value?.features || []
  const key = colorBy.value
  const title = FIELD_LABEL[key] || key

  // Live (in-browser) clusters: values come from the reactive assignment map,
  // not a property. Same stable palette as the pipeline clusters.
  if (key === 'live_cluster') {
    const seen = new Set()
    let hasNull = false
    for (const f of feats) {
      const lab = live.labelFor(f.properties)
      if (hasValue(lab)) seen.add(lab); else hasNull = true
    }
    const legend = [...seen].sort().map((lab) => ({ label: lab, color: categoryColor('live_cluster', lab) }))
    if (hasNull) legend.push({ label: 'Unclustered', color: UNCLUSTERED })
    return {
      type: 'categorical', title, legend,
      colorFn: (p) => categoryColor('live_cluster', live.labelFor(p)),
      // The legend label this mark would carry, for hover highlighting.
      labelOf: (p) => (hasValue(live.labelFor(p)) ? live.labelFor(p) : 'Unclustered'),
    }
  }

  // Cluster keeps its own stable palette + an explicit "Unclustered" bucket.
  if (key === 'cluster') {
    const seen = new Set()
    let hasNull = false
    for (const f of feats) {
      const c = f.properties.cluster
      if (hasValue(c)) seen.add(c); else hasNull = true
    }
    const legend = [...seen].sort((a, b) => a - b).map((c) => ({ label: `Cluster ${c}`, color: colorFor(c) }))
    if (hasNull) legend.push({ label: 'Unclustered', color: UNCLUSTERED })
    return {
      type: 'categorical', title, legend,
      colorFn: (p) => colorFor(p.cluster),
      labelOf: (p) => (hasValue(p.cluster) ? `Cluster ${p.cluster}` : 'Unclustered'),
    }
  }

  // Any other categorical dimension (land cover, species, …): assign palette
  // colors to the distinct values present, most frequent first. The legend is
  // capped (a dataset can have hundreds of species) with a "+N more" row.
  if (CATEGORY_KEYS.has(key)) {
    const counts = new Map()
    for (const f of feats) {
      const v = f.properties[key]
      if (hasValue(v)) counts.set(v, (counts.get(v) || 0) + 1)
    }
    const cats = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v)
    const LEGEND_CAP = 12
    // Where a layer draws the same classes, take its colours: a land cover dot
    // should be the colour of the land cover under it, not a hash of its name.
    // Anything the layer has no class for still falls back to the stable
    // palette, so an unexpected value is coloured rather than dropped.
    const palette = paletteFor(key)
    const classes = palette?.kind === 'classes' ? palette.items : null
    const colorOf = (v) => (classes && classColorFor(classes, v)) || categoryColor(key, v)

    const legend = cats.slice(0, LEGEND_CAP).map((v) => ({ label: String(v), color: colorOf(v) }))
    if (cats.length > LEGEND_CAP) legend.push({ label: `+${cats.length - LEGEND_CAP} more`, color: UNCLUSTERED })
    return {
      type: 'categorical', title, legend,
      match: palette ? matchNote(palette) : '',
      colorFn: (p) => (hasValue(p[key]) ? colorOf(p[key]) : UNCLUSTERED),
      labelOf: (p) => (hasValue(p[key]) ? String(p[key]) : null),
    }
  }

  // Numeric (sequential). Elevation and temperature follow the ft/m and °F/°C
  // settings, so the gradient scale matches the units shown elsewhere.
  const meta = ALL_NUMERIC.find((f) => f.key === key) || {}
  const conv = meta.unit === 'elev' ? elevValue : meta.unit === 'temp' ? tempValue : (v) => Number(v)
  const unitSuffix = meta.unit === 'elev' ? ` (${unit.value})` : meta.unit === 'temp' ? ` (°${tempUnit.value})` : ''
  const vals = feats.map((f) => f.properties[key]).filter(hasValue).map((v) => conv(Number(v)))
  const dataMin = vals.length ? Math.min(...vals) : 0
  const dataMax = vals.length ? Math.max(...vals) : 1

  // Where a layer draws the same quantity, borrow its ramp — and its stretch
  // too when the two sides measure in the same units, so a dot over a pixel is
  // the same colour for the same value. Where the pipeline normalises and the
  // layer does not, the palette still matches but the scale is the data's own;
  // `match` says which of the two the viewer is looking at.
  const palette = paletteFor(key)
  // A ramp the viewer chose outranks the layer match. Choosing a scale is a
  // decision about every field at once, and having it silently not apply to the
  // handful of fields a layer also draws would read as the control being broken.
  const chosen = chosenPointRamp.value
  const stops = chosen || (palette?.kind === 'ramp' ? palette.stops : RAMP)
  // The layer's stretch only applies while its palette does.
  const [min, max] = (!chosen && palette?.domain) || [dataMin, dataMax]

  return {
    type: 'sequential', title: title + unitSuffix, min, max, stops,
    match: !chosen && palette ? matchNote(palette) : '',
    colorFn: (p) => {
      const raw = p[key]
      if (!hasValue(raw)) return UNCLUSTERED
      return rampColor(stops, fraction(conv(Number(raw)), [min, max]))
    },
  }
})

// The RAW category values on screen, most common first — what the appearance
// panel keys its per-value overrides on. Deliberately not taken from the legend,
// whose labels are display text ("Cluster 3") rather than the value itself.
const legendValues = computed(() => {
  if (coloring.value?.type !== 'categorical') return []
  const key = colorBy.value
  const counts = new Map()
  for (const f of filteredData.value?.features || []) {
    const v = key === 'live_cluster' ? live.labelFor(f.properties) : f.properties[key]
    if (hasValue(v)) counts.set(v, (counts.get(v) || 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v)
})

// Size points by a numeric field (radius), or a uniform size when "none".
const sizeScale = computed(() => {
  if (!sizeBy.value) return null
  const feats = filteredData.value?.features || []
  const vals = feats.map((f) => f.properties[sizeBy.value]).filter(hasValue).map(Number)
  if (!vals.length) return null
  return { lo: Math.min(...vals), hi: Math.max(...vals) }
})
function radiusFor(props) {
  // The configured point size sets the baseline; a "Size by" field then scales
  // marks around it, so both controls compose instead of fighting.
  const base = pointRadius.value
  const s = sizeScale.value
  if (!s) return base * 1.5
  const v = props[sizeBy.value]
  if (!hasValue(v)) return base * 0.75
  return base + base * 2.25 * ((Number(v) - s.lo) / ((s.hi - s.lo) || 1))
}

/**
 * How one observation is drawn. Every place that styles a marker goes through
 * here, so creation and re-styling cannot disagree.
 *
 * The outline follows the opacity slider rather than staying at full strength:
 * fading the dots while their rings stayed solid turned a dense area into a gray
 * mesh — the opposite of what turning the dots down is for.
 */
function markerStyle(props) {
  // While a legend row is hovered, everything it does not stand for fades back
  // rather than disappearing — the surrounding marks are the context that makes
  // the highlighted ones mean something.
  const c = coloring.value
  const faded = hoverValue.value !== null && typeof c.labelOf === 'function'
    && c.labelOf(props) !== hoverValue.value
  const fill = faded ? pointOpacity.value * 0.12 : pointOpacity.value
  return {
    radius: radiusFor(props),
    fillColor: c.colorFn(props),
    fillOpacity: fill,
    stroke: pointOutline.value && !faded,
    weight: pointOutline.value && !faded ? 1 : 0,
    color: '#222',
    opacity: pointOutline.value && !faded ? fill : 0,
  }
}

// Coloring, sizing, palette, per-value overrides and point styling all restyle
// the existing layer in place — no need to rebuild it, which would refit the
// view.
watch([coloring, sizeScale, activeColors, colorOverrides, pointRadius, pointOpacity, pointOutline, colorSeed, hoverValue], () => {
  if (!geoLayer) return
  geoLayer.eachLayer((l) => {
    const style = markerStyle(l.feature.properties)
    l.setStyle(style)
    l.setRadius(style.radius)
  })
})

// When focusing an observation, the next re-render must not refit/clear it.
// Leaflet owns the centre and zoom, so they are mirrored into a ref for the
// share link rather than read out of shared state.
const mapView = ref(null)
// Flatten the live map — tiles, the point canvas, any overlay canvas — into a
// PNG. Everything is measured on screen rather than recomputed, so what is saved
// is exactly what is displayed.
const exporter = useImageExport()
const saving = ref(false)
const saveError = ref('')

// ─── Point visibility ───────────────────────────────────────────────────────
// The overlay is drawn UNDER the points, and 48k marks cover most of the
// shading they are meant to sit on. Hiding them is what makes an overlay
// readable, so it is a first-class toggle rather than an appearance setting.
const POINTS_KEY = 'map-show-points'
const showPoints = ref(true)
// Whether the layer manager window is up. Not a popover: it stays open while
// you work the map, because that is how you tell whether a layer was worth it.
const showLayers = ref(false)
if (import.meta.client) {
  showPoints.value = localStorage.getItem(POINTS_KEY) !== '0'
}
watch(showPoints, (v) => {
  if (import.meta.client) localStorage.setItem(POINTS_KEY, v ? '1' : '0')
  if (!map || !geoLayer) return
  if (v) { geoLayer.addTo(map); geoLayer.bringToFront() } else geoLayer.remove()
})

// ─── Tooltips ───────────────────────────────────────────────────────────────
const shortcuts = useShortcuts()
// Every control's tooltip says what it does and, where one exists, the key that
// does it — so shortcuts are discoverable without opening the help overlay.
const tip = (text, keys) => shortcuts.withKey(text, keys)

const heatmapTip = computed(() => {
  const note = heatmapMeta.value?.note
  return note
    ? tip(`${heatmapMeta.value.label}: ${note}`, 'o')
    : tip('Draw a grid summary under the points', 'o')
})

// The ? beside the heatmap picker documents the heatmap you actually have
// The per-mode reference entry, the season labels and the window's dates all
// live in useMapHeatmaps now: the heatmap's controls render both on the bar
// and inside the layer window, and two copies of these would be two copies to
// disagree.


async function saveMap() {
  if (!mapEl.value || saving.value) return
  saving.value = true
  saveError.value = ''
  try {
    const blob = await exporter.mapToPng(mapEl.value, { scale: 2 })
    exporter.download(blob, `map-${exporter.slugify(colorBy.value, 'view')}-${exporter.stamp()}.png`)
  } catch (err) {
    saveError.value = err.message || 'Could not save the map.'
  } finally {
    saving.value = false
  }
}

function syncMapView() {
  if (!map) return
  mapView.value = { center: map.getCenter(), zoom: map.getZoom() }
  const b = map.getBounds()
  viewBounds.value = {
    north: b.getNorth(), south: b.getSouth(), east: b.getEast(), west: b.getWest(),
    zoom: map.getZoom(),
  }
  // Left where another page can find it: the pipeline job form offers "use the
  // current map view" for its area, and asking someone to read four numbers off
  // one screen and type them into another is not a feature.
  try {
    localStorage.setItem('map-last-view', JSON.stringify({
      bounds: viewBounds.value,
      zoom: map.getZoom(),
      // The drawn layers ride along so the Options page can price an offline
      // save without a map: the cost is the area times the number of layers,
      // and one without the other is not an estimate of anything.
      sources: activeTileTemplates.value,
      at: Date.now(),
    }))
  } catch { /* private mode or quota */ }
}

/** The viewport, as the chunk loader wants it. */
function currentView() {
  if (!map) return {}
  const b = map.getBounds()
  const c = map.getCenter()
  return {
    bounds: { north: b.getNorth(), south: b.getSouth(), east: b.getEast(), west: b.getWest() },
    zoom: map.getZoom(),
    centre: [c.lat, c.lng],
  }
}

// Panning into new ground loads that ground. Debounced, because a drag fires
// moveend once but a pinch-zoom fires several, and each would otherwise start
// its own round of fetches for overlapping cells.
let panTimer = null
function loadVisible() {
  if (!chunks.available.value) return
  clearTimeout(panTimer)
  panTimer = setTimeout(() => { loadProgressive(currentView()) }, 220)
}

// What is on screen, for saving an area to read with no signal. Kept beside
// mapView rather than derived from it: a centre and a zoom do not give you the
// edges without knowing the container's size, which Leaflet already knows.
const viewBounds = ref(null)

// The layers actually drawn right now — the basemap in use plus whatever
// reference layers are switched on. Saving the whole catalogue would spend a
// viewer's data on layers they are not looking at.
//
// Each one carries a stable id as well as its template. For most layers the two
// are the same string, but an Earth Engine template holds a token that expires
// within hours: filed under its URL, a saved tile is unreachable by the time
// anyone is standing in the woods reading it. The id is what the cache keys on.
const activeTileTemplates = ref([])
function syncActiveTemplates() {
  if (!map) return
  const out = []
  map.eachLayer((l) => {
    // ArcGIS export layers build their URLs per tile rather than from a
    // template, so they cannot be enumerated ahead of time and are skipped.
    if (!l._url || typeof l._url !== 'string' || !l._url.includes('{z}')) return
    // The name rides along so the offline estimate can say which layer is
    // costing the download, rather than listing anonymous URLs at somebody
    // deciding what to turn off.
    out.push({
      template: l._url,
      id: l._spec?.ee ? l._spec.key : l._url,
      name: l._spec?.name || '',
      // Where this layer runs out of tiles, so a save does not request zooms it
      // does not publish. maxNativeZoom is the real ceiling; maxZoom on these
      // layers is the map's own limit, which every layer shares.
      maxZoom: Number.isFinite(l.options?.maxNativeZoom) ? l.options.maxNativeZoom : null,
    })
  })
  activeTileTemplates.value = out
}

const datasetLabel = computed(() => {
  const n = filteredData.value?.features?.length || 0
  return n ? `${n.toLocaleString()} observations` : ''
})

const shareTitle = computed(() => {
  const n = filteredData.value?.features?.length || 0
  const what = speciesFilter.value?.length === 1 ? speciesFilter.value[0] : 'mushroom observations'
  return `${n.toLocaleString()} ${what}: data-map`
})

let suppressFit = false
// Whether the map has ever been fitted to data. Until it has, a fit is the
// thing that puts the viewer somewhere sensible at all.
let fittedOnce = false

// Heatmap cells indexed by their grid key, so the cell under a point is found
// by arithmetic rather than by scanning thousands of polygons on every hover.
const heatmapCellIndex = computed(() => {
  const index = new Map()
  for (const c of heatmapResult.value.cells || []) index.set(c.key, c)
  return index
})

// ─── Earth Engine layers ─────────────────────────────────────────────────────
// These have no fixed URL: the server asks Earth Engine to render the layer and
// returns a tile template, which is then used like any other. So the layer is
// created empty, and the template is fetched the first time it is switched on —
// minting one for a layer nobody looks at would spend quota for nothing.
const eeTiles = useEeTiles()
const maxEnt = useMaxEnt()
// For authorising a point-sample of members' layers, the same token the tile
// path uses.
const { accessToken } = useAuth()
// Only for registering minted templates against their layer, so saved Earth
// Engine tiles survive a token rotation. The saving itself lives in the panel.
const offline = useOffline()
const eeParams = ref({})
const eeErrors = ref([])
// The layer picker's contents. Populated once the map and its layers exist, so
// the Vue side never has to know how Leaflet builds them.
const baseLayers = ref([])
const overlayLayers = ref([])
// The basemap the viewer last chose, remembered per browser so the map opens on
// the one they read best rather than resetting to the default every visit.
const BASE_KEY = 'map-basemap'
const activeBase = ref('grey')
// A Set of the overlay keys currently on. Replaced rather than mutated so the
// template re-renders.
const activeOverlays = ref(new Set())

/** Overlays grouped for display, in catalogue order. */
const overlayGroups = computed(() => {
  const groups = new Map()
  for (const o of overlayLayers.value) {
    if (!groups.has(o.group)) groups.set(o.group, [])
    groups.get(o.group).push(o)
  }
  return [...groups.entries()].map(([label, items]) => ({ label, items }))
})

// Sync MaxEnt model runs into the overlay layer list (HEAT-5).
// Each completed model appears as a toggleable entry in the Layer Manager.
// Entries are lightweight stubs — no actual Leaflet tile layer until the
// tile URL can be obtained from the suitability asset path via the GEE endpoint.
watch(maxEnt.maxentLayerSpecs, (specs) => {
  // Remove stale maxent entries and replace with the current model list.
  overlayLayers.value = [
    ...overlayLayers.value.filter((o) => !o.key.startsWith('maxent:')),
    ...specs.map((s) => ({
      key: s.key,
      name: s.name,
      group: s.group,
      note: s.note,
      layer: null, // rendered via the heatmap mode, not a Leaflet tile layer
    })),
  ]
}, { immediate: true })

function setBase(key) {
  const next = baseLayers.value.find((b) => b.key === key)
  if (!next || !map) return
  for (const b of baseLayers.value) if (b.layer !== next.layer) map.removeLayer(b.layer)
  if (!map.hasLayer(next.layer)) next.layer.addTo(map)
  // Basemaps sit under everything; without this a basemap switched on later
  // draws over the reference layers and the points.
  next.layer.bringToBack()
  activeBase.value = key
  try { localStorage.setItem(BASE_KEY, key) } catch { /* private mode; just don't remember */ }
  syncActiveTemplates()
}

/** Restore the remembered basemap, if it is one that still exists. */
function restoreBase() {
  let saved = null
  try { saved = localStorage.getItem(BASE_KEY) } catch { /* no storage; keep the default */ }
  if (saved && saved !== activeBase.value && baseLayers.value.some((b) => b.key === saved)) {
    setBase(saved)
  }
}

// A suitability surface handed over from the jobs page. Drawn as one tile
// overlay above the basemap and below the observation points, with its own
// legend card. `modelLayer` is the Leaflet layer; `modelOverlay` is what the
// legend reads.
const overlayHandoff = useModelOverlay()
const modelOverlay = ref(null)
let modelLayer = null

/** How old the surface is, from when its tiles were minted. */
function overlayAge(mintedAt) {
  if (!mintedAt) return ''
  const d = new Date(mintedAt)
  if (!Number.isFinite(d.getTime())) return ''
  const hours = Math.floor((Date.now() - d.getTime()) / 3600000)
  if (hours < 1) return 'just now'
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

/** Draw the pending suitability surface, if the jobs page left one. */
function applyModelOverlay() {
  const pending = overlayHandoff.pending.value
  if (!pending || !pending.template || !map || !L) return
  removeModelOverlay()

  const layer = L.tileLayer(pending.template, {
    opacity: 0.7,
    maxZoom: MAP_MAX_ZOOM,
    // The surface tracks zoom the way the reference overlays do (see the note on
    // updateWhenIdle where those are built).
    updateWhenIdle: false,
    updateWhenZooming: true,
    className: 'model-suitability',
  })

  // A minted map id expires; when its tiles start 404ing the surface is gone,
  // and a blank overlay reads as "nowhere is suitable" rather than "this
  // expired". Count failures and say so instead.
  let failed = 0
  layer.on('tileerror', () => {
    failed += 1
    if (failed >= 3 && modelOverlay.value && !modelOverlay.value.stale) {
      modelOverlay.value = { ...modelOverlay.value, stale: true }
    }
  })
  layer.addTo(map)
  modelLayer = layer

  modelOverlay.value = {
    jobId: pending.jobId || '',
    label: pending.label || 'Model',
    legend: pending.legend || { stops: ['#2c2f6b', '#c6301f'], min: '0', max: '1' },
    age: overlayAge(pending.mintedAt),
    cv: pending.cv || null,
    stale: false,
    refreshing: false,
  }

  // Frame the region the surface was projected over, so it is not off-screen.
  const r = pending.region
  if (r && Number.isFinite(r.north)) {
    try {
      map.fitBounds(L.latLngBounds([r.south, r.west], [r.north, r.east]).pad(0.05), { animate: false })
    } catch { /* a bad region is not worth failing the draw over */ }
  }

  // Consumed, so a later revisit of the map does not redraw a surface the member
  // removed. Reopening it from the jobs page sets it again.
  overlayHandoff.clear()
}

/**
 * Re-mint an expired surface from its stored model, in place.
 *
 * The template carries an Earth Engine map id that expires; rather than send the
 * viewer back to the jobs page to re-run, ask the server to re-serve the stored
 * model and swap the tile URL under the same layer.
 */
async function refreshModelOverlay() {
  const o = modelOverlay.value
  if (!o?.jobId || !modelLayer) return
  modelOverlay.value = { ...o, refreshing: true }
  try {
    const token = await accessToken()
    const res = await fetch(`/.netlify/functions/model-tiles?job=${encodeURIComponent(o.jobId)}`, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    })
    const body = await res.json()
    if (!res.ok || !body.ok || !body.template) throw new Error(body.error || 'Could not refresh the surface.')
    modelLayer.setUrl(body.template)
    modelOverlay.value = {
      ...modelOverlay.value, stale: false, refreshing: false, age: overlayAge(body.meta?.mintedAt),
    }
  } catch {
    modelOverlay.value = { ...modelOverlay.value, refreshing: false }
  }
}

/** Take the suitability surface off the map. */
function removeModelOverlay() {
  if (modelLayer && map) map.removeLayer(modelLayer)
  modelLayer = null
  modelOverlay.value = null
}

// The stacking order of the overlays that are on, topmost first, and how see-
// through each one is. Both are per-layer because both were global and that was
// wrong: overlays hide one another, so land ownership under a hillshade is a
// different map from the same two the other way up, and dimming the pile to
// read through it dimmed the one thing you were trying to read.
const overlayOrder = ref([])
const layerOpacity = ref({})

// Per-layer blend overrides, and the one layer being looked at on its own.
// Both are session state beside the order and the opacity, not preferences: a
// solo is a thing you do for ten seconds, and a blend set on one layer means
// nothing once that layer is off. The stack DEFAULT is a preference, and lives
// with the rest of them in Appearance.
const layerBlend = ref({})
const soloKey = ref('')

/** Push the current order down into Leaflet as z-indexes. */
function applyOverlayOrder() {
  if (!map) return
  const n = overlayOrder.value.length
  overlayOrder.value.forEach((key, i) => {
    const entry = overlayLayers.value.find((o) => o.key === key)
    // Topmost first in the list, so the first entry gets the highest index.
    entry?.layer?.setZIndex?.(200 + (n - i))
  })
}

/**
 * Push the blend modes down onto each drawn layer's own container.
 *
 * Leaflet gives every tile layer a div of its own inside the tile pane, so a
 * mix-blend-mode there composites that layer against the ones below it and
 * nothing else. The browser does the work per frame; no tile is re-fetched and
 * nothing is recomputed, which is why this is a dropdown rather than a job.
 *
 * Solo is applied here too, because the set of layers that are drawn is the set
 * whose blending matters — a hidden layer with multiply still on it would come
 * back blended when the solo ended, which is right, and blending a layer that
 * is not on screen is work for nothing.
 */
function applyBlendModes() {
  const drawn = drawnKeys([...activeOverlays.value], soloKey.value)
  for (const entry of overlayLayers.value) {
    const el = entry.layer?.getContainer?.()
    if (!el) continue
    el.style.mixBlendMode = drawn.includes(entry.key)
      ? effectiveBlend(entry.key, {
        overrides: layerBlend.value, fallback: stackBlend.value, drawn: drawn.length,
      })
      : 'normal'
  }
}

/**
 * Add or remove layers so that only the soloed one is drawn.
 *
 * The active set is not touched. Solo answers "what is this one contributing",
 * and answering it must not cost the viewer the stack they built — so the rest
 * stay ticked in the manager, dimmed, and come back untouched.
 */
function applySolo() {
  if (!map) return
  const drawn = new Set(drawnKeys([...activeOverlays.value], soloKey.value))
  for (const key of activeOverlays.value) {
    const entry = overlayLayers.value.find((o) => o.key === key)
    if (!entry?.layer) continue
    const on = map.hasLayer(entry.layer)
    if (drawn.has(key) && !on) entry.layer.addTo(map)
    else if (!drawn.has(key) && on) map.removeLayer(entry.layer)
  }
  applyOverlayOrder()
  applyBlendModes()
}

function setSolo(key) {
  soloKey.value = key === soloKey.value ? '' : key
  applySolo()
}

function setLayerBlend(key, mode) {
  // '' is "inherit the stack default", which is not the same as normal: change
  // the default later and this layer should move with it.
  const next = { ...layerBlend.value }
  if (mode) next[key] = mode
  else delete next[key]
  layerBlend.value = next
  applyBlendModes()
}

function toggleOverlay(entry) {
  if (!map) return
  const wasOn = activeOverlays.value.has(entry.key)
  const next = new Set(activeOverlays.value)
  if (wasOn) {
    next.delete(entry.key)
    overlayOrder.value = overlayOrder.value.filter((k) => k !== entry.key)
    // Take it off the map here. applySolo only walks the active set, so once the
    // key is gone from there it can no longer remove this layer — leaving an
    // unticked layer still drawn, which is the bug this fixes.
    if (entry.layer && map.hasLayer(entry.layer)) map.removeLayer(entry.layer)
    // Switching off the layer that was soloed ends the solo rather than
    // leaving an empty map with three layers still ticked.
    if (soloKey.value === entry.key) soloKey.value = ''
  } else {
    next.add(entry.key)
    // A layer just switched on goes on top, which is where someone who just
    // asked for it expects to see it — and ends any solo, since asking for a
    // second layer is asking to see two.
    overlayOrder.value = [entry.key, ...overlayOrder.value]
    soloKey.value = ''
  }
  activeOverlays.value = next
  // When a MaxEnt model layer is toggled on, switch the heatmap to MaxEnt
  // mode and select that model so HeatmapControls reflects the active entry.
  if (!wasOn && entry.key.startsWith('maxent:')) {
    heatmaps.mode.value = 'maxent'
    heatmaps.maxentModelId.value = entry.key.replace('maxent:', '')
  }
  // Read from the active set rather than from the map: with a solo running, a
  // layer can be switched on and yet not be on the map, so hasLayer answers a
  // different question from the one the checkbox asked.
  applySolo()
  syncActiveTemplates()
}

function toggleOverlayByKey(key) {
  const entry = overlayLayers.value.find((o) => o.key === key)
  if (entry) toggleOverlay(entry)
}

function moveOverlay(key, delta) {
  // `delta` is a number of places, or 'top' or 'bottom'.
  overlayOrder.value = reorderStack(overlayOrder.value, key, delta)
  applyOverlayOrder()
  // The stack default blends each layer against what is below it, so moving a
  // layer changes what it is blended with.
  applyBlendModes()
}

/** One layer's own opacity, multiplied into the global dimmer. */
function setLayerOpacity(key, value) {
  const entry = overlayLayers.value.find((o) => o.key === key)
  if (!entry) return
  layerOpacity.value = { ...layerOpacity.value, [key]: value }
  entry.layer.setOpacity(entry.layer._baseOpacity * value * tileOpacity.value)
  heatmaps.persist()
}

function clearOverlays() {
  soloKey.value = ''
  for (const key of [...activeOverlays.value]) toggleOverlayByKey(key)
}

// Changing the default moves every layer nobody has set by hand, which is what
// makes it a default rather than a one-time stamp.
watch(stackBlend, () => applyBlendModes())

const activeBaseName = computed(() =>
  baseLayers.value.find((b) => b.key === activeBase.value)?.name || '')
const eeLayers = new Map()

/** The parameters a layer is currently set to, defaulted from its schema. */
function paramsFor(spec) {
  const held = eeParams.value[spec.key] || {}
  const out = {}
  for (const [name, p] of Object.entries(spec.params || {})) {
    out[name] = held[name] ?? p.default
  }
  return out
}

async function refreshEeLayer(spec) {
  const layer = eeLayers.get(spec.key)
  if (!layer || !map.hasLayer(layer)) return
  eeErrors.value = eeErrors.value.filter((e) => e.key !== spec.key)
  try {
    const minted = await eeTiles.template(spec.key, paramsFor(spec))
    // setUrl rather than a rebuild, so the layer keeps its place in the stack
    // and its toggle stays on.
    layer.setUrl(minted.template)
    // Tell the offline worker which layer this token belongs to. Without it a
    // tile saved under an earlier token cannot be matched to this request, and
    // an area saved this morning draws blank this afternoon.
    offline.registerEeTemplate(spec.key, minted.template)
    syncActiveTemplates()
  } catch (err) {
    // Loud and by name. A layer that fails quietly is indistinguishable from
    // one showing that nothing is there, and on a fire map that is a lie.
    eeErrors.value = [...eeErrors.value, { key: spec.key, name: spec.name, message: err.message }]
  }
}

async function addEeLayers() {
  const layers = await eeTiles.loadCatalogue()
  if (!layers.length || !map || !L) return

  for (const spec of layers) {
    // An empty URL until it is switched on. Leaflet is content with that and
    // simply draws nothing, which is what an unrequested layer should do.
    const layer = L.tileLayer('', {
      attribution: spec.attribution,
      opacity: (spec.opacity ?? 1) * tileOpacity.value,
      maxZoom: MAP_MAX_ZOOM,
      // Earth Engine renders any zoom it is asked for, so there is no native
      // ceiling to upsample from.
      crossOrigin: 'anonymous',
      // Track the zoom continuously on touch too; see the reference layers.
      updateWhenIdle: false, updateWhenZooming: true,
    })
    layer._baseOpacity = spec.opacity ?? 1
    layer._spec = { ...spec, ee: true }
    tileLayers.push(layer)
    eeLayers.set(spec.key, layer)

    layer.on('add', () => {
      if (!activeTileNotes.value.some((n) => n.name === spec.name)) {
        activeTileNotes.value = [...activeTileNotes.value, {
          name: spec.name,
          note: spec.note,
          legend: spec.legend,
          ee: spec.key,
          eeParams: spec.params,
          slow: spec.slow,
          // A layer whose classes are too many to list in a key. The key shows
          // a browser for them instead; see SoilTaxonomyKey.
          classes: spec.classes,
          legendInBrowser: spec.legendInBrowser,
          slug: spec.name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
        }]
      }
      refreshEeLayer(spec)
    })
    layer.on('remove', () => {
      activeTileNotes.value = activeTileNotes.value.filter((n) => n.name !== spec.name)
      eeErrors.value = eeErrors.value.filter((e) => e.key !== spec.key)
    })

    // A layer the viewer's tier cannot render is still listed, marked, rather
    // than hidden: knowing FRMS computes it is part of what membership
    // is for. Ticking it explains itself through the error card.
    overlayLayers.value = [...overlayLayers.value, {
      key: spec.key, name: spec.name, group: spec.group, layer, tier: spec.tier, note: spec.note,
      source: layerSource(spec.attribution), type: layerDataType(spec.legend),
    }]
  }
}

/**
 * A parameter changed on an active Earth Engine layer: re-render it.
 *
 * Clamped to the schema the server sent. A number box can be typed into as well
 * as stepped, so "5" lands in a year field easily enough, and the server would
 * rightly refuse it — spending a round trip to be told what the schema already
 * says here. The server still checks; this only avoids asking a question whose
 * answer is known.
 */
function setEeParam(key, name, value) {
  const spec = eeTiles.catalogue.value.find((l) => l.key === key)
  if (!spec) return
  const p = spec.params?.[name]
  let next
  if (p?.type === 'enum') {
    // A choice from a fixed list: keep it as the string it is, falling back to
    // the default if somehow handed something off the list.
    next = (p.values || []).includes(String(value)) ? String(value) : (p.default ?? (p.values || [])[0])
  } else if (p?.type === 'codes') {
    // A set of class codes. Normalised here with the same function the server
    // normalises with, so the cache key the browser produces is the one the
    // server produces and a selection is minted once rather than twice.
    try {
      next = normaliseCodes(value, p.max)
    } catch {
      return
    }
  } else if (p?.type === 'text') {
    // A typed value, e.g. a taxon name. Kept as a trimmed string; an empty one
    // is ignored rather than sent, since the server rejects it and re-minting on
    // every emptied field would only surface an error mid-type.
    const text = String(value).trim()
    if (!text) return
    next = text
  } else {
    next = Math.floor(Number(value))
    if (!Number.isFinite(next)) next = p?.default ?? 0
    if (p && Number.isFinite(p.min)) next = Math.max(p.min, next)
    if (p && Number.isFinite(p.max)) next = Math.min(p.max, next)
  }

  eeParams.value = {
    ...eeParams.value,
    [key]: { ...(eeParams.value[key] || {}), [name]: next },
  }
  // Debounced: stepping a year field or dragging a "days back" spinner fires a
  // change per stop, and each re-mint is an Earth Engine call and a serverless
  // invocation. Coalescing the bursts into one request per key spends one call
  // for a settled value rather than one for every value passed through.
  debounceEeRefresh(spec)
}

// Per-layer timers, so changing one layer's parameters never delays another's.
const eeRefreshTimers = new Map()
function debounceEeRefresh(spec, wait = 400) {
  clearTimeout(eeRefreshTimers.get(spec.key))
  eeRefreshTimers.set(spec.key, setTimeout(() => {
    eeRefreshTimers.delete(spec.key)
    refreshEeLayer(spec)
  }, wait))
}

// ─── Dropped point ───────────────────────────────────────────────────────────
// Somewhere the viewer picked, as opposed to somewhere a record exists. Held as
// plain numbers rather than a Leaflet marker so the panel can be reactive and
// the marker stays a detail of the map.
const pin = ref(null)
const copied = ref(false)
let pinMarker = null

// A self-contained SVG marker for the dropped point. Leaflet's default marker
// pulls its image from a PNG whose URL the bundler rewrites out from under it,
// so it 404s and the pin shows up blank; an inline divIcon has no asset to lose.
// Blue, to read apart from the red pin that marks a selected observation.
function dropPinIcon() {
  return L.divIcon({
    className: 'drop-pin', iconSize: [28, 40], iconAnchor: [14, 38], tooltipAnchor: [0, -34],
    html: `<svg viewBox="0 0 24 34" width="28" height="40" aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.3 0 11.9 0 20.6 12 34 12 34s12-13.4 12-22.1C24 5.3 18.6 0 12 0z"
            fill="#2d7ff9" stroke="#fff" stroke-width="1.5"/>
      <circle cx="12" cy="12" r="4.5" fill="#fff"/></svg>`,
  })
}

function setPin(lat, lon) {
  pin.value = { lat, lon }
  copied.value = false
  if (!map || !L) return
  if (pinMarker) { pinMarker.setLatLng([lat, lon]); return }
  pinMarker = L.marker([lat, lon], {
    draggable: true,
    icon: dropPinIcon(),
    // Above the canvas the observations draw into, so the pin is never lost
    // under a dense patch of dots.
    zIndexOffset: 1000,
    title: 'Dropped point — drag to move',
  }).addTo(map)
  // Dragging is how you correct a click that landed a hundred metres off,
  // which on a phone is most of them.
  pinMarker.on('drag move', () => {
    const ll = pinMarker.getLatLng()
    pin.value = { lat: ll.lat, lon: ll.lng }
    copied.value = false
  })
}

function clearPin() {
  pin.value = null
  if (pinMarker) { pinMarker.remove(); pinMarker = null }
}

async function copyPin() {
  if (!pin.value) return
  const text = `${pin.value.lat.toFixed(5)}, ${pin.value.lon.toFixed(5)}`
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => { copied.value = false }, 1600)
  } catch {
    // Clipboard access is refused in plenty of contexts; selecting the text is
    // still possible, so this is not worth an error message.
  }
}

// A plus code for the point, at 11 digits (~3 m) since a dropped pin is a
// specific spot rather than a neighbourhood. encodePlusCode is auto-imported
// from composables/plusCode.js.
const pinPlusCode = computed(() => (pin.value ? encodePlusCode(pin.value.lat, pin.value.lon, 11) : ''))

// Ground elevation at the point. undefined while loading, null when it could
// not be fetched, a number in metres otherwise — three states so the panel can
// say "…" versus "—" rather than conflating them. From Open-Meteo's free,
// key-less elevation API (Copernicus DEM at 90 m), so it adds no cost and no
// Earth Engine quota; a dropped pin fetches once, debounced, and a drag replaces
// the in-flight request rather than stacking them.
const pinElevation = ref(undefined)
let elevTimer = null
let elevSeq = 0
watch(pin, (p) => {
  pinElevation.value = p ? undefined : null
  if (!p) return
  clearTimeout(elevTimer)
  const seq = (elevSeq += 1)
  elevTimer = setTimeout(async () => {
    try {
      const url = `https://api.open-meteo.com/v1/elevation?latitude=${p.lat.toFixed(5)}&longitude=${p.lon.toFixed(5)}`
      const res = await fetch(url)
      const data = await res.json()
      const v = Array.isArray(data?.elevation) ? Number(data.elevation[0]) : NaN
      if (seq === elevSeq) pinElevation.value = Number.isFinite(v) ? v : null
    } catch {
      if (seq === elevSeq) pinElevation.value = null
    }
  }, 350)
}, { deep: true })

/** Elevation formatted in both units, or the loading/unavailable marker. */
const pinElevationText = computed(() => {
  const v = pinElevation.value
  if (v === undefined) return '…'
  if (v === null) return '—'
  return `${Math.round(v)} m · ${Math.round(v * 3.28084).toLocaleString()} ft`
})

/** Copy any short string, reusing the pin's copied flag for the tick. */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => { copied.value = false }, 1600)
  } catch { /* clipboard refused; the text is still selectable */ }
}

// The active Earth Engine layers, with their current parameters, that the
// "Sample layers here" button will read at the pin. Reference layers (GIBS,
// ArcGIS) are external tiles with no server-side image to sample, so only the
// Earth Engine layers are offered.
const activeEeLayers = computed(() => {
  const cat = eeTiles.catalogue.value || []
  const out = []
  for (const key of activeOverlays.value) {
    const spec = cat.find((l) => l.key === key)
    if (spec) out.push({ key, params: paramsFor(spec) })
  }
  return out
})

const pinSamples = ref(null)
const pinSampling = ref(false)
const pinSampleError = ref('')

// A new point invalidates the old readings: sampling is per-coordinate, so the
// values from the last spot must not linger under a pin that has since moved.
watch(pin, () => { pinSamples.value = null; pinSampleError.value = '' }, { deep: true })

/**
 * Read the active Earth Engine layers at the pin, on demand.
 *
 * On demand, not automatically, because each layer sampled is one Earth Engine
 * read: a button press spends that deliberately, where sampling on every pin
 * drop would spend it on every misclick.
 */
async function samplePinLayers() {
  if (!pin.value || !activeEeLayers.value.length || pinSampling.value) return
  pinSampling.value = true
  pinSampleError.value = ''
  try {
    const token = await accessToken()
    const res = await fetch('/.netlify/functions/ee-sample', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ lat: pin.value.lat, lon: pin.value.lon, layers: activeEeLayers.value }),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(data.error || `Could not sample (${res.status}).`)
    pinSamples.value = data.results || []
  } catch (e) {
    pinSampleError.value = e.message
  } finally {
    pinSampling.value = false
  }
}

/** One sampled layer's value, formatted for the panel. */
function sampleText(s) {
  if (s.error) return 'unavailable'
  if (s.empty) return 'no data here'
  if (s.label) return s.label
  if (s.channels) return s.channels.map((c) => Math.round(c.value)).join(' / ')
  if (s.value === null || s.value === undefined) return '—'
  return `${typeof s.value === 'number' ? fmtNum(s.value) : s.value}${s.unit ? ` ${s.unit}` : ''}`
}

/** The heatmap cell under the pin, if a heatmap is on and it has one there. */
const pinCell = computed(() => (pin.value ? heatmapCellAt(pin.value.lat, pin.value.lon) : null))

const pinCellValue = computed(() => {
  const cell = pinCell.value
  if (!cell) return ''
  const v = cell.value
  if (v === null || v === undefined) return '—'
  return typeof v === 'number' ? fmtNum(v) : String(v)
})

/** The closest loaded observation, so the pin has something to be relative to. */
const pinNearest = computed(() => {
  if (!pin.value) return null
  const feats = filteredData.value?.features || []
  if (!feats.length) return null
  const { lat, lon } = pin.value
  // Equirectangular is plenty at these distances and avoids a trig call per
  // feature across tens of thousands of them.
  const scale = Math.cos((lat * Math.PI) / 180)
  let best = null
  let bestD = Infinity
  for (const f of feats) {
    const co = f.geometry?.coordinates
    if (!co) continue
    const dx = (Number(co[0]) - lon) * scale
    const dy = Number(co[1]) - lat
    const d = dx * dx + dy * dy
    if (d < bestD) { bestD = d; best = f }
  }
  if (!best) return null
  const km = Math.sqrt(bestD) * 111.32
  const p = best.properties || {}
  const name = p.species || p.genus || 'a record'
  const away = km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`
  return { label: `${name}, ${away} away`, feature: best }
})

function heatmapCellAt(lat, lon) {
  if (!heatmapCell.value || !heatmapMode.value) return null
  return heatmapCellIndex.value.get(heatmaps.keyAt(lat, lon)) || null
}

const esc = (v) => String(v)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** What the tooltip says about one observation, given how the map is set up. */
function pointTooltip(feature) {
  const p = feature?.properties || {}
  const co = feature?.geometry?.coordinates
  const rows = []

  const title = p.species || 'Observation'
  if (p.date) rows.push(['Observed', p.date])

  // The value behind this mark's color, named by the dimension chosen.
  const c = coloring.value
  if (c && typeof c.labelOf === 'function') {
    const v = c.labelOf(p)
    // "Cluster · Cluster 1" reads as a stutter, so a value that already carries
    // its dimension's name stands on its own.
    if (hasValue(v) && v !== title) {
      const dim = String(c.title || '')
      const val = String(v)
      if (dim && val.toLowerCase().startsWith(dim.toLowerCase())) rows.push(['', val])
      else rows.push([dim, val])
    }
  } else if (colorBy.value && hasValue(p[colorBy.value])) {
    rows.push([FIELD_LABEL[colorBy.value] || colorBy.value, fmtNum(p[colorBy.value])])
  }
  if (sizeBy.value && hasValue(p[sizeBy.value])) {
    rows.push([`${FIELD_LABEL[sizeBy.value] || sizeBy.value} (size)`, fmtNum(p[sizeBy.value])])
  }

  // And what the heatmap makes of the cell this point falls in.
  if (heatmapMode.value && co) {
    const cell = heatmapCellAt(co[1], co[0])
    if (cell) {
      const m = heatmapMode.value
      const meta = heatmapMeta.value
      const label = meta?.label || 'Heatmap'
      const value = meta?.kind === 'field'
        // The cell mean, with how many readings went into it — a mean of two is
        // a different claim from a mean of two hundred.
        ? `${meta.circular ? `${Math.round(cell.value)}°` : fmtNum(cell.value)} (${cell.samples} obs)`
        : m === 'common' || m === 'land_cover' ? (cell.label || ', ')
          : m === 'season' || m === 'hotspots'
            ? `${Math.round((cell.n ? cell.inWindow / cell.n : 0) * 100)}% of ${cell.n} finds`
            : m === 'richness' ? `${cell.species.size} species`
              : m === 'wind' ? `${Math.round(cell.aspectDeg ?? 0)}°`
                : `${cell.n} observations`
      rows.push([label, value])
    }
  }

  return `<strong>${esc(title)}</strong>`
    + rows.map(([k, v]) => `<span class="ot-row">${k ? `<span class="ot-k">${esc(k)}</span>` : ''}${esc(v)}</span>`).join('')
}

// The drawer shows the coordinates, but a GeoJSON feature keeps them in its
// geometry rather than its properties — so they are carried across here.
function selectFeature(feature) {
  if (!feature) return null
  const co = feature.geometry?.coordinates
  // Whether this point is the thinned copy from the overview or the full record
  // from its cell. The drawer needs to know: an unenriched-looking record that
  // is merely un-fetched must not be reported as one the pipeline never
  // sampled. The flag lives on the feature, not its properties, so it has to be
  // carried across explicitly.
  const thinned = chunks.available.value && !feature.__full
  const base = { ...feature.properties, __thinned: thinned }
  return co ? { ...base, lon: co[0], lat: co[1] } : base
}

// Rebuild the point layer whenever the dataset changes (e.g. species switch).
function renderPoints(geo) {
  if (!map || !L || !geo) return
  if (geoLayer) { geoLayer.remove(); geoLayer = null }
  if (!suppressFit) selected.value = null

  geoLayer = L.geoJSON(geo, {
    pointToLayer: (feature, latlng) => L.circleMarker(latlng, markerStyle(feature.properties)),
  }).addTo(map)

  // One tooltip and one click handler for the whole layer, resolved against
  // whichever marker the event came from. Binding them per feature created a
  // Tooltip object and a listener for every observation — ~48k of each — which
  // cost more than drawing the markers did.
  // Hovering a point says what it is AND what the map is currently saying about
  // it: the value behind its color and size, and what the overlay reports for
  // the cell it sits in. Without that, the encodings can only be read by eye
  // against a legend, and the overlay could not be read at a point at all.
  geoLayer.bindTooltip((lyr) => pointTooltip(lyr.feature),
                       { direction: 'top', sticky: true, className: 'obs-tip' })
  geoLayer.on('click', (e) => {
    const feature = e.layer?.feature
    if (!feature) return
    selected.value = selectFeature(feature)
    const co = feature.geometry?.coordinates
    selectedLatLng.value = co ? [co[1], co[0]] : null
  })

  if (!showPoints.value) geoLayer.remove()

  const bounds = geoLayer.getBounds()
  // Non-animated: an in-flight fit animation would block a subsequent zoom-in to
  // a focused observation (Leaflet ignores zoom changes mid-animation).
  if (bounds.isValid() && !suppressFit) {
    map.fitBounds(bounds.pad(0.1), { animate: false })
    fittedOnce = true
  }
  suppressFit = false // one-shot
}

// Fitting the view to the data is right when a filter narrows to one species,
// and wrong when a chunk lands. The viewer panned somewhere deliberately; the
// ground under them arriving is not a reason to throw them back to the extent
// of the whole dataset. Worse, the fit fires moveend, which asks for the cells
// of the view it just jumped to, which lands another chunk: the map would sit
// there flicking between where you were and the whole country.
//
// The first chunked render is the overview, which is everything, so that one
// still fits. After that only a real data change moves the map.
let seenChunkVersion = 0
watch(filteredData, (geo) => {
  if (chunks.version.value !== seenChunkVersion) {
    seenChunkVersion = chunks.version.value
    if (fittedOnce) suppressFit = true
  }
  renderPoints(geo)
})

// "Open on map" from a chart: select the matching observation and pan to it.
function applyFocus(target) {
  if (!target || !map) return
  const lon = Number(target.lon), lat = Number(target.lat)
  const feats = filteredData.value?.features || []
  const match = (target.uuid && feats.find((f) => f.properties?.uuid === target.uuid))
    || feats.find((f) => {
      const co = f.geometry?.coordinates
      return co && Math.abs(co[0] - lon) < 1e-6 && Math.abs(co[1] - lat) < 1e-6
    })
  if (match) selected.value = selectFeature(match)
  if (Number.isFinite(lat) && Number.isFinite(lon)) {
    selectedLatLng.value = [lat, lon]
    // Zoom in on the observation (not just pan). Stop any in-flight fit-to-data
    // animation first, or it would complete and override this zoom.
    suppressFit = true
    map.setView([lat, lon], 15)
  }
  setFocusObservation(null) // consume so a later revisit doesn't re-trigger
}
watch(focusObservation, (t) => t && applyFocus(t))

// A location pin marks the currently-selected observation (from a click or from
// "Open on map"), and clears when the detail drawer is closed.
function pinIcon() {
  return L.divIcon({
    className: 'obs-pin', iconSize: [28, 40], iconAnchor: [14, 38], tooltipAnchor: [0, -34],
    html: `<svg viewBox="0 0 24 34" width="28" height="40" aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.3 0 11.9 0 20.6 12 34 12 34s12-13.4 12-22.1C24 5.3 18.6 0 12 0z"
            fill="#e34948" stroke="#fff" stroke-width="1.5"/>
      <circle cx="12" cy="12" r="4.5" fill="#fff"/></svg>`,
  })
}
watch(selectedLatLng, (ll) => {
  if (!map || !L) return
  if (selectedMarker) { selectedMarker.remove(); selectedMarker = null }
  if (ll) selectedMarker = L.marker(ll, { icon: pinIcon(), interactive: false, zIndexOffset: 1000 }).addTo(map)
})
// Closing the drawer (selected → null) removes the pin.
watch(selected, (s) => { if (!s) selectedLatLng.value = null })

onMounted(async () => {
  try {
    await nextTick()
    if (!mapEl.value) throw new Error('map container not ready')
    L = (await import('leaflet')).default

    // crossOrigin: the image export composites these tiles onto a canvas, and a
    // tile fetched without it taints the canvas so toBlob() throws.
    //
    // maxNativeZoom, not maxZoom, on every layer that runs out of tiles before
    // MAP_MAX_ZOOM. The two mean different things and the difference is what
    // used to break this map: maxZoom tells Leaflet the layer does not exist
    // past that level, so it HIDES it, while maxNativeZoom says the tiles stop
    // there and Leaflet keeps showing the last real level, upscaled.
    //
    // Getting that wrong cost more than the base map. The default gray canvas
    // stops at 16, and Leaflet takes the map's own zoom ceiling from its
    // layers, so 16 was as far as the whole map would go.
    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: MAP_MAX_ZOOM, maxNativeZoom: 19, crossOrigin: 'anonymous',
    })
    const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenTopoMap (CC-BY-SA)',
      maxZoom: MAP_MAX_ZOOM, maxNativeZoom: 17, crossOrigin: 'anonymous',
    })
    // Muted basemaps, and the default. A street or topo map is drawn to be read
    // on its own; the moment 48k colored dots sit on top of it, its own color
    // is competing with the data for the same hues. A gray canvas gives the dots
    // the only saturation on screen — which is why it is the conventional base
    // for a point map, and why it is what this one opens with. Terrain is still
    // one click away, and the hillshade overlay puts relief back without color.
    // Esri's gray canvas rather than CARTO's, which now demands an API key and
    // answers without one by serving a tile that says so — a 200 response, so
    // nothing downstream can tell it apart from a map. These come from the same
    // host as the satellite and hillshade layers the app already uses.
    const grey = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri, HERE, Garmin, © OpenStreetMap contributors',
      maxZoom: MAP_MAX_ZOOM, maxNativeZoom: 16, crossOrigin: 'anonymous',
    })
    const greyDark = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri, HERE, Garmin, © OpenStreetMap contributors',
      maxZoom: MAP_MAX_ZOOM, maxNativeZoom: 16, crossOrigin: 'anonymous',
    })
    const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Imagery © Esri',
      maxZoom: MAP_MAX_ZOOM, maxNativeZoom: 19, crossOrigin: 'anonymous',
    })

    // Zoom control on the bottom-left so it never overlaps the top-left
    // "Color by" control (previously it clipped the label).
    // preferCanvas draws the markers into a single <canvas> instead of giving
    // each one its own SVG <path>. At ~48k observations the SVG renderer put
    // 48k interactive nodes in the DOM, which is what made panning and zooming
    // crawl; the canvas renderer keeps that flat as the dataset grows.
    // A dot is drawn at 3px. On a mouse that is a fine target; on a finger it
    // is roughly a tenth of the contact patch, and tapping one was a matter of
    // luck. The canvas renderer's `tolerance` widens the hit test without
    // widening the mark, so points stay small and become tappable — and the
    // padding it needs is the fingertip's, not the mouse's.
    const coarse = import.meta.client
      && window.matchMedia?.('(pointer: coarse)').matches
    map = L.map(mapEl.value, {
      scrollWheelZoom: true, zoomControl: false, layers: [grey], preferCanvas: true,
      renderer: L.canvas({ tolerance: coarse ? 12 : 2 }),
      // Two fingers to pan the page-length map on touch would be right if the
      // map were incidental; here it IS the page, so one finger pans it.
      tap: true, tapTolerance: 20,
      maxZoom: MAP_MAX_ZOOM,
    }).setView([39.5, -105.7], 7)
    // Locate first, then zoom: Leaflet stacks a corner's controls in the order
    // they are added, so this puts the crosshair directly above the +/- pair
    // rather than in the control bar at the top, which is where it was competing
    // with controls about the data rather than about the view.
    const LocateControl = L.Control.extend({
      onAdd() {
        const wrap = L.DomUtil.create('div', 'leaflet-bar locate-ctl')
        const btn = L.DomUtil.create('a', '', wrap)
        btn.href = '#'
        btn.role = 'button'
        btn.title = tip('Centre the map on where you are', 'l')
        btn.setAttribute('aria-label', 'My location')
        btn.innerHTML = '<span class="dot-icon"></span>'
        locateBtn = btn
        // stop() as well as preventDefault: without it the click reaches the map
        // underneath and, in pin mode, drops a point behind the button.
        L.DomEvent.on(btn, 'click', (e) => { L.DomEvent.stop(e); locateMe() })
        L.DomEvent.disableClickPropagation(wrap)
        return wrap
      },
    })
    new LocateControl({ position: 'bottomleft' }).addTo(map)
    L.control.zoom({ position: 'bottomleft' }).addTo(map)
    // ArcGIS MapServer services render from a bbox rather than serving a cut
    // tile pyramid, so their tiles are asked for by extent. Everything else is
    // a plain XYZ template.
    const ArcGISLayer = L.TileLayer.extend({
      getTileUrl(coords) {
        return arcgisExportUrl(this.options.service, coords.x, coords.y, coords.z,
          { size: 256, layers: this.options.serviceLayers })
      },
    })

    // Reference tile services as toggleable layers alongside the basemaps.
    const tileOverlayList = []
    for (const o of TILE_LAYERS) {
      const opts = {
        // The catalogue's maxZoom is where each service's tiles stop, which is
        // maxNativeZoom here. Passed as maxZoom it made every coarse layer —
        // all of Weather, Ground and Vegetation — vanish the moment the map was
        // zoomed past it, so ticking them appeared to do nothing at all.
        attribution: o.attribution, maxZoom: MAP_MAX_ZOOM, maxNativeZoom: o.maxZoom,
        opacity: (o.opacity ?? 1) * tileOpacity.value,
        crossOrigin: 'anonymous',
        // Leaflet defaults updateWhenIdle to true on touch devices, which leaves
        // an overlay's tiles pinned in place through a pinch-zoom and only
        // repositioned once the gesture ends — the layer reads as "stuck" while
        // the basemap moves under it. Update continuously instead so the overlay
        // tracks the zoom the way the basemap does.
        updateWhenIdle: false, updateWhenZooming: true,
      }
      const layer = o.arcgis
        ? new ArcGISLayer('', { ...opts, service: o.arcgis, serviceLayers: o.layers || '' })
        : L.tileLayer(o.url.replace('{date}', tileDate.value), opts)
      // Its own opacity is kept beside it: the global dimmer multiplies into
      // this rather than replacing it, so a hillshade meant to sit at 60%
      // stays proportionally lighter than a layer meant to sit at full.
      layer._baseOpacity = o.opacity ?? 1
      layer._spec = o
      tileLayers.push(layer)
      // A reference layer that fails to load looks exactly like one saying there
      // is nothing there — no trails, no public land — which is the most
      // misleading thing this map could do. Track whether a layer has ever
      // succeeded, and say so when it has not.
      let loaded = 0
      let failed = 0
      layer.on('tileload', () => {
        loaded += 1
        if (loaded === 1) tileErrors.value = tileErrors.value.filter((n) => n !== o.name)
      })
      layer.on('tileerror', () => {
        failed += 1
        // One failure is a hiccup; several with nothing loaded is the service.
        if (loaded === 0 && failed >= 3 && !tileErrors.value.includes(o.name)) {
          tileErrors.value = [...tileErrors.value, o.name]
        }
      })
      // Its key and its caveat travel with it: shown while it is on, gone when
      // it is off. A layer with neither still registers nothing, which is right
      // — imagery and place labels are pictures, not measurements.
      if (o.note || o.legend || o.time) {
        layer.on('add', () => {
          if (!activeTileNotes.value.some((n) => n.name === o.name)) {
            activeTileNotes.value = [...activeTileNotes.value, {
              name: o.name, note: o.note, legend: o.legend, time: !!o.time,
              native: o.maxZoom,
              slug: o.name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
            }]
          }
        })
      }
      // Clear the warning when the layer is switched off, so it does not linger.
      layer.on('remove', () => {
        tileErrors.value = tileErrors.value.filter((n) => n !== o.name)
        activeTileNotes.value = activeTileNotes.value.filter((n) => n.name !== o.name)
        loaded = 0
        failed = 0
      })
      tileOverlayList.push({
        key: o.name, name: o.name, group: o.group, layer, note: o.note,
        source: layerSource(o.attribution), type: layerDataType(o.legend),
      })
    }
    // The global dimmer still dims everything at once — but it now multiplies
    // into whatever each layer has been set to individually, rather than
    // replacing it. A hillshade meant to sit at 60% stays proportionally
    // lighter than a layer meant to sit at full, and a layer someone has faded
    // by hand in the manager stays faded.
    watch(tileOpacity, (v) => {
      for (const l of tileLayers) {
        const own = layerOpacity.value[l._spec?.ee ? l._spec.key : l._spec?.name] ?? 1
        l.setOpacity(l._baseOpacity * own * v)
      }
      heatmaps.persist()
    })

    // Moving the date re-points the time-varying layers at another day's tiles.
    // setUrl redraws in place, so a layer keeps its position in the stack and
    // its toggle stays on rather than the layer being rebuilt under the viewer.
    watch(tileDate, (d) => {
      if (!d) return
      for (const l of tileLayers) {
        if (l._spec?.time && l._spec.url) l.setUrl(l._spec.url.replace('{date}', d))
      }
    })
    // Our own layer picker rather than L.control.layers, for two reasons. It
    // sits in the control bar with everything else instead of floating over the
    // map in its own white box, and Leaflet's takes one flat list, so grouping
    // had to be smuggled into the labels as markup.
    baseLayers.value = [
      { key: 'grey', name: 'Light gray', layer: grey },
      { key: 'greyDark', name: 'Dark gray', layer: greyDark },
      { key: 'osm', name: 'Street (OSM)', layer: osm },
      { key: 'topo', name: 'Terrain (OpenTopoMap)', layer: topo },
      { key: 'sat', name: 'Satellite (Esri)', layer: sat },
    ]
    // The map was created with the default basemap; swap in the remembered one
    // now that the choices exist.
    restoreBase()
    overlayLayers.value = tileOverlayList

    // Earth Engine layers arrive after their catalogue does, so they join the
    // list rather than being in it from the start.
    addEeLayers()

    // Right-click, or a long press on a touch screen. A plain click already
    // opens an observation, and a mode button for something used this rarely was
    // a permanent icon paying for an occasional action.
    map.on('contextmenu', (e) => setPin(e.latlng.lat, e.latlng.lng))

    // Long press, done by hand. Leaflet maps a native contextmenu to its own
    // event, but which browsers synthesise one from a long press is uneven —
    // iOS Safari in particular does not — so the gesture is timed here.
    const container = map.getContainer()
    let pressTimer = null
    let pressAt = null
    const cancelPress = () => { clearTimeout(pressTimer); pressTimer = null; pressAt = null }

    container.addEventListener('touchstart', (ev) => {
      // One finger only: a two-finger touch is a pinch-zoom starting.
      if (ev.touches.length !== 1) return cancelPress()
      const t = ev.touches[0]
      pressAt = { x: t.clientX, y: t.clientY }
      pressTimer = setTimeout(() => {
        pressTimer = null
        const pt = map.mouseEventToLatLng({ clientX: pressAt.x, clientY: pressAt.y })
        setPin(pt.lat, pt.lng)
        // A pin that appears with no other feedback feels like a glitch; a tick
        // of haptics is what every map app uses to say "that registered".
        navigator.vibrate?.(15)
      }, 550)
    }, { passive: true })

    // A finger that has moved is a pan, not a press.
    container.addEventListener('touchmove', (ev) => {
      if (!pressAt || !ev.touches.length) return
      const t = ev.touches[0]
      if (Math.hypot(t.clientX - pressAt.x, t.clientY - pressAt.y) > 10) cancelPress()
    }, { passive: true })
    container.addEventListener('touchend', cancelPress, { passive: true })
    container.addEventListener('touchcancel', cancelPress, { passive: true })

    map.on('moveend zoomend', syncMapView)
    map.on('moveend zoomend', loadVisible)
    // Only the layers-control events. `layeradd` and `layerremove` fire once
    // per layer, and every observation marker is a layer: binding a handler
    // that walks map.eachLayer() to them made adding n markers cost n squared
    // layer visits. At 9,647 markers that was 3.2 seconds of eachLayer on the
    // main thread; at the full 48,233 it is 25 times worse, which is most of
    // why the map used to sit there and never appear.
    map.on('baselayerchange overlayadd overlayremove', syncActiveTemplates)
    syncMapView()
    syncActiveTemplates()

    // Leaflet calculates tile positions and canvas bounds from the container size
    // it measures at creation. Any resize after that — drawer sliding in, screen
    // rotation, mobile keyboard, browser window resize — leaves the internal pixel
    // origin stale so every layer appears offset from the basemap. Watching the
    // container and calling invalidateSize() keeps the two in sync.
    if (typeof ResizeObserver !== 'undefined') {
      mapResize = new ResizeObserver(() => { if (map) map.invalidateSize({ animate: false }) })
      mapResize.observe(mapEl.value)
    }

    heatmaps.loadFromStorage()
    appearance.loadFromStorage()

    // A suitability surface waiting from the jobs page, if the member pressed
    // "View suitability on map". Drawn after the base layers exist so it sits
    // above them.
    applyModelOverlay()

    // A shared link wins over stored preferences: the point of opening one is to
    // see what the sender saw, not what you last had configured.
    const shared = share.apply(useRoute().query)
    if (shared.colorBy) colorBy.value = shared.colorBy
    if (shared.sizeBy !== null) sizeBy.value = shared.sizeBy

    // Chunks first: the overview paints at once and the cells under the view
    // follow. Only when they were never built does this fall back to fetching
    // all 49.7 MB before drawing anything, which is what it used to do always.
    const chunked = await loadProgressive(currentView())
    if (!chunked) await load()
    if (!data.value) throw new Error('no data')
    // A link carrying a view sets it explicitly; skip the fit-to-data that would
    // otherwise throw that view away.
    if (shared.view) suppressFit = true
    renderPoints(filteredData.value)
    if (shared.view) {
      map.setView(shared.view.center, shared.view.zoom, { animate: false })
      // The map is now deliberately placed, so the chunks that arrive for this
      // view must not refit it away.
      fittedOnce = true
    }
    syncMapView()
    renderHeatmap()
    loaded.value = true
    // If arriving via "Open on map" from a chart, focus that observation now.
    if (focusObservation.value) applyFocus(focusObservation.value)
  } catch (err) {
    loadError.value = `Could not load map (${err.message}).`
  }
})

// Show a dot at the viewer's location (browser geolocation, opt-in per click).
function locateMe() {
  if (!map || !L) return
  if (!('geolocation' in navigator)) {
    locateError.value = 'Geolocation not supported by this browser.'
    return
  }
  locating.value = true
  locateError.value = ''
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      locating.value = false
      const { latitude: lat, longitude: lon, accuracy } = pos.coords
      if (userLayer) { userLayer.remove(); userLayer = null }
      userLayer = L.layerGroup([
        // Accuracy halo + a solid "you are here" dot.
        L.circle([lat, lon], { radius: accuracy || 0, color: '#2a78d6', weight: 1, fillOpacity: 0.12 }),
        L.circleMarker([lat, lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#2a78d6', fillOpacity: 1 })
          .bindTooltip('You are here', { direction: 'top' }),
      ]).addTo(map)
      map.setView([lat, lon], Math.max(map.getZoom() || 0, 11))
    },
    (err) => {
      locating.value = false
      locateError.value = err.code === err.PERMISSION_DENIED
        ? 'Location permission denied.'
        : 'Could not get your location.'
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
  )
}

// Only while the map is on screen: pressing "o" on the Charts page should do
// nothing rather than reach for a control that is not there.
const HEATMAP_KEYS = heatmaps.HEATMAP_MODES.map((m) => m.key)
function cycleHeatmap(step) {
  const i = HEATMAP_KEYS.indexOf(heatmapMode.value)
  heatmapMode.value = HEATMAP_KEYS[(i + step + HEATMAP_KEYS.length) % HEATMAP_KEYS.length]
}
function nudgeDay(days) {
  seasonDay.value = ((seasonDay.value - 1 + days + 365) % 365) + 1
}

shortcuts.register([
  { scope: 'Map', keys: 'p', label: 'Show / hide observation points', run: () => { showPoints.value = !showPoints.value } },
  { scope: 'Map', keys: 'o', label: 'Next heatmap', run: () => cycleHeatmap(1) },
  { scope: 'Map', keys: 'shift+O', label: 'Previous heatmap', run: () => cycleHeatmap(-1) },
  { scope: 'Map', keys: 'l', label: 'My location', run: () => locateMe() },
  { scope: 'Map', keys: 'shift+L', label: 'Manage layers', run: () => { showLayers.value = !showLayers.value } },
  { scope: 'Map', keys: 'e', label: 'Save the map as an image', run: () => saveMap() },
  { scope: 'Map', keys: '[', label: 'Heatmap date back a week', run: () => nudgeDay(-7) },
  { scope: 'Map', keys: ']', label: 'Heatmap date forward a week', run: () => nudgeDay(7) },
  { scope: 'Map', keys: 's', label: 'Heatmap and season window', run: () => heatmapPop.value?.toggle() },
  { scope: 'Map', keys: 'shift+T', label: 'Replay the feature tour', run: () => window.dispatchEvent(new CustomEvent('map-tour-open')) },
  { scope: 'Map', keys: 'escape', label: 'Close the observation drawer', run: () => { selected.value = null } },
])

// The bar renders behind v-if="loaded", so start measuring when it appears.
watch(loaded, (ok) => { if (ok) nextTick(trackControlsHeight) }, { immediate: true })

// Escape leaves pin mode, and leaves it again to clear the pin. A mode with no
// keyboard way out is a trap on a laptop, where the toggle can be off-screen.
function onKeydown(e) {
  if (e.key !== 'Escape') return
  if (!pin.value) return
  clearPin()
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown)
  controlsResize?.disconnect()
  mapResize?.disconnect()
  if (map) map.remove()
})
</script>

<style scoped>
/* Must match .drawer's width in ObservationDrawer.vue: the key column steps
   aside by exactly this when the drawer opens. It was 320 here and 340 there,
   which nothing noticed only because nothing read it. */
.map-shell { --drawer-w: 340px; }

.map-shell { position: relative; width: 100%; height: 100%; }
.map { width: 100%; height: 100%; }

.overlay {
  position: absolute; inset: 0; display: grid; place-items: center;
  background: rgba(255, 255, 255, 0.7); font: 500 15px/1.4 system-ui, sans-serif;
  color: #333; z-index: 500; pointer-events: none;
}
.overlay.error { color: #b00020; }

.controls {
  position: absolute; top: 12px; left: 12px; z-index: 500; display: flex; gap: 10px; align-items: center;
  flex-wrap: wrap;
}
/* One look for every button in the bar, wherever its component happens to
   define it. Five components contribute controls here and each had its own
   height, radius, border and background — a light one next to a dark one next
   to a colored one — so side by side they read as several toolbars that had
   collided rather than one. The rules live here because this bar is the only
   place they sit together; each component keeps its own styling everywhere
   else it is used. */
/* Leaflet's div-icon ships a white box with a grey border; our SVG pins supply
   their own shape, so strip the box or it frames the teardrop. */
.map :deep(.leaflet-div-icon.drop-pin),
.map :deep(.leaflet-div-icon.obs-pin) {
  background: none;
  border: 0;
}

.controls :deep(.pop-btn),
.controls .tool-btn,
.controls :deep(.sh-btn),
.controls :deep(.ap-btn),
.controls :deep(.set-btn) {
  box-sizing: border-box;
  min-height: 34px; height: 34px; min-width: 34px; padding: 0 9px;
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  background: var(--surface, rgba(255, 255, 255, 0.95));
  color: var(--text, #333);
  border: 1px solid var(--border, #ddd); border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.12);
  font: inherit; font-size: 0.82rem; font-weight: 500;
  white-space: nowrap; cursor: pointer;
}
/* A finger is not a cursor. The app-wide rule already asks for 40px on a coarse
   pointer, but it does that with min-height on `button`, and min-height beats
   height whatever the specificity — so three of the six grew and three did not,
   which is how the bar ended up with two heights on a phone and one on a
   desktop. Setting both here makes them agree, at the larger size, which is the
   one a thumb wants. */
@media (pointer: coarse) {
  .controls :deep(.pop-btn),
  .controls .tool-btn,
    .controls :deep(.sh-btn),
  .controls :deep(.ap-btn),
  .controls :deep(.set-btn) {
    min-height: 40px; height: 40px; min-width: 40px;
  }
}

.controls :deep(.pop-btn):hover,
.controls .tool-btn:hover,
.controls :deep(.sh-btn):hover,
.controls :deep(.ap-btn):hover,
.controls :deep(.set-btn):hover { border-color: var(--muted, #999); }

/* One "this is doing something" state, rather than three. */
.controls :deep(.pop-btn.on),
.controls .tool-btn.on,
.controls :deep(.sh-btn.on),
.controls :deep(.ap-btn.on),
.controls :deep(.set-btn.on) {
  border-color: var(--accent, #2b7a3d);
  box-shadow: 0 0 0 2px rgba(43, 122, 61, 0.18);
}

.colorby {
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 7px 10px; font: 13px system-ui, sans-serif; display: flex; gap: 8px; align-items: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.colorby label { color: var(--muted); font-weight: 600; }
.colorby select { border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-size: 13px; }
.locate {
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 7px 10px; font: 600 13px system-ui, sans-serif; color: #333; cursor: pointer;
  display: inline-flex; gap: 7px; align-items: center; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.locate:hover { background: #fff; }
.locate.busy { opacity: 0.7; cursor: progress; }
.locate .dot-icon {
  width: 11px; height: 11px; border-radius: 50%; background: #2a78d6; border: 2px solid #fff;
  box-shadow: 0 0 0 1px #2a78d6; flex: 0 0 auto;
}

/* One column down the right-hand side holds both legends. They used to place
   themselves independently — the overlay legend pinned to the top, the coloring
   legend to the bottom — which works only while the control bar is a single row.
   On a phone the bar is three rows tall and the overlay legend landed on top of
   it; worse, the mobile rule added `bottom` without clearing the `top` it
   inherited, so the box was stretched between the two and rendered as a mostly
   empty panel half the height of the map.
   Placement now belongs to the container, and the legends only stack inside it,
   so neither has to know how tall the other is. */
.legends {
  position: absolute; bottom: 18px; right: 12px; z-index: 500;
  /* Slides aside when the drawer opens rather than being buried under it: the
     key is how you read the colors on the map, and opening a record is exactly
     when you want to check what a color meant. */
  transition: right 0.22s ease;
  /* Below the control bar, whose height depends on how many rows it wraps into —
     picking an overlay adds a "Cell size" dropdown and a second row, which is
     exactly when the overlay legend appears to collide with it. */
  top: calc(var(--controls-h, 0px) + 20px);
  display: flex; flex-direction: column; align-items: flex-end; gap: 10px;
  justify-content: flex-end;
  /* The column spans the map so the two ends are reachable; only the panels
     themselves should catch a click. */
  pointer-events: none; max-width: 46vw; min-height: 0;
}
.legends > * { pointer-events: auto; }

/* Beside the drawer rather than under it. */
.map-shell.drawer-open .legends { right: calc(var(--drawer-w) + 12px); }

/* Collapsed: the toggle stays, everything it controls goes. */
.legends.collapsed .legend { display: none; }

/* Where the drawer takes most of the screen there is no "beside" to move to,
   so the key folds to its header and the viewer opens it when they want it.
   Sliding it off the left edge instead would just lose it. */
@media (max-width: 760px) {
  .map-shell.drawer-open .legends { right: 12px; }
  .map-shell.drawer-open .legends .legend { display: none; }
}

.key-toggle {
  align-self: flex-end; pointer-events: auto;
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 4px 9px; font: 600 11px/1 system-ui, sans-serif; color: #444;
  cursor: pointer; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.12);
}
.key-toggle:hover { border-color: #bbb; }
.key-toggle .caret { font-size: 9px; color: #888; }

@media (prefers-reduced-motion: reduce) { .legends { transition: none; } }

.legend {
  position: static; z-index: 500;
  /* Theme tokens, not a hardcoded white card: the child keys (the soil taxonomy
     browser especially) colour their text with --text, so on a fixed white
     panel their dark-mode text came out white on white. */
  background: var(--surface, rgba(255, 255, 255, 0.95));
  border: 1px solid var(--border, #ddd); border-radius: 8px;
  padding: 10px 12px; font: 13px/1.4 system-ui, sans-serif;
  color: var(--text, #222); min-width: 120px;
  max-width: 100%; max-height: 44vh; overflow-y: auto; overscroll-behavior: contain;
  /* min-height: 0 — a flex item will not shrink below its content without it,
     so the panel grew past its max-height instead of scrolling.
     flex-shrink 1 is what lets a card give way when the column is crowded; the
     cap above already keeps any one card from taking the whole column, and
     without the shrink a tall stack pushes the last card off the map. */
  flex: 0 1 auto; min-height: 0;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.legend-title { font-weight: 600; margin-bottom: 6px; position: sticky; top: 0; }
.legend-row { display: flex; align-items: center; gap: 8px; }

/* Which of the two keys this is. The layer key and the observation key sit in
   one column, one above the other, and previously each was headed only by what
   it described — "Land cover" over "Land cover" — leaving the viewer to work
   out which explained the ground and which the dots. */
.legend-kind {
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 4px;
}
/* Reinforced by shape, for the same reason a legend has text at all: the marks
   on the map are round and the layers are areas of flat colour, so the key
   repeats that distinction rather than relying on the heading alone. */
.points-legend .swatch.dot { border-radius: 50%; }
.tile-note .swatch { border-radius: 2px; }
.legend-note.match { font-style: italic; }

/* One section per active layer inside the layers key: a ramp with its units in
   the middle of the scale, or a list of classes. Rules between them, because
   without one a note and the next layer's name run together. */
.tk + .tk { border-top: 1px solid rgba(0, 0, 0, 0.1); margin-top: 8px; padding-top: 8px; }
.tk-name { font-weight: 600; font-size: 0.8rem; margin-bottom: 4px; }
.tk .legend-note { margin-top: 4px; }

/* Wraps, because eleven land-cover classes will not fit on one line of a
   260px card. */
.class-key { display: flex; flex-wrap: wrap; gap: 3px 9px; margin: 3px 0 5px; }
.class-key .ck { display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; }
.gradient-scale .unit { color: var(--muted); }
.layer-date { display: flex; align-items: center; gap: 6px; margin: 6px 0 2px; font-size: 0.74rem; }
.layer-date label { color: var(--muted); font-weight: 600; }
.layer-date input {
  flex: 1 1 auto; min-width: 0; background: var(--input-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: 2px 5px; font-size: 0.74rem;
}

/* The chunk indicator: bottom-left, above Leaflet's zoom control, and quiet
   enough to ignore while still being legible over imagery. */
/* ── Touch ────────────────────────────────────────────────────────────────
   A fingertip is about 9mm across. The control bar's icon buttons ship at
   34px and Leaflet's zoom at 30px, both under half of that, so on a coarse
   pointer they get real hit areas — 40px of reach without 40px of bulk where
   the bulk would crowd the map. */
@media (pointer: coarse) {
  .map-shell :deep(.ap-btn.icon-only),
  .map-shell :deep(.sh-btn.icon-only),
  .map-shell :deep(.set-btn) { width: 40px; height: 40px; }

  /* The locate button belongs in this list. It sits directly on top of the
     zoom pair, so a different width reads as a misalignment rather than as two
     controls. */
  .map-shell :deep(.leaflet-control-zoom a),
  .map-shell :deep(.locate-ctl a),
  .map-shell :deep(.leaflet-control-layers-toggle) { width: 40px; height: 40px; line-height: 40px; }

  /* A legend row is a tap target now, not just a hover target, so it needs
     the height of one — and it must not select its own text when held. */
  .legend-row.hoverable {
    min-height: 32px; -webkit-user-select: none; user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  .legend-row.hoverable:active { background: rgba(0, 0, 0, 0.06); border-radius: 4px; }

  /* Scrolling a key must scroll the key, not pan the map underneath it. */
  .legends > * { touch-action: pan-y; }

  /* A range input's own box is only as tall as its thumb, so the band you can
     start a drag in is a few pixels. Height here is hit area, not appearance:
     the thumb stays centred and the browser keeps its native look, which
     restyling it (appearance: none) would throw away along with the track.
     touch-action stops a horizontal drag on the slider from panning the map. */
  .map-shell input[type="range"] { height: 32px; touch-action: none; }
}

/* Direction has no low and no high, so its key is four labelled swatches
   round the compass rather than a bar with two ends. */
.compass-key { display: flex; flex-wrap: wrap; gap: 4px 10px; margin: 2px 0 4px; }
.compass-key .ck { display: inline-flex; align-items: center; gap: 4px; font-size: 0.74rem; }

/* The overlay legend sits above the point legend, in the same column. */
/* Pushed to the top of the column, leaving the coloring legend at the bottom —
   the arrangement this had before, now expressed as a relationship between the
   two rather than as two absolute positions that can collide. */
.overlay-legend { max-width: 280px; }
/* The layers key sits at the top of the column and the coloring key at the
   bottom, with the heatmap key between them — a relationship between the three
   rather than three absolute positions that can collide. */
.tile-note { margin-bottom: auto; }
.legends > .overlay-legend:first-child { margin-bottom: auto; }
.legend-note {
  margin-top: 6px; font-size: 11px; line-height: 1.35; color: #555;
  border-top: 1px solid #e6e6e6; padding-top: 5px;
}
.legend-n { color: #777; font-size: 11px; }

/* The locate control, styled to match Leaflet's own zoom buttons it sits on. */
/* Deliberately no width or height. Leaflet sizes .leaflet-bar a itself, and the
   locate button sits directly on top of the zoom pair — so inheriting that
   sizing is the only way the two cannot drift apart. Setting 30px here is what
   made it 30 against the zoom's 40: the coarse-pointer rule above enlarges both,
   but this rule comes later in the file and won on equal specificity. */
.map-shell :deep(.locate-ctl a) {
  display: flex; align-items: center; justify-content: center;
  background: #fff; cursor: pointer;
}
.map-shell :deep(.locate-ctl a.busy) { opacity: 0.6; cursor: progress; }
.map-shell :deep(.locate-ctl .dot-icon) {
  width: 11px; height: 11px; border-radius: 50%; background: #2a78d6;
  border: 2px solid #fff; box-shadow: 0 0 0 1px #2a78d6;
}

/* Dropped point */
.map-shell :deep(.leaflet-container.picking) { cursor: crosshair; }
.pin-panel {
  position: absolute; left: 12px; bottom: 96px; z-index: 620;
  width: 232px; max-width: calc(100vw - 24px);
  background: rgba(255, 255, 255, 0.96); border: 1px solid #d8d8d8; border-radius: 8px;
  padding: 9px 10px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.13);
  font: 12px/1.4 system-ui, sans-serif; color: #333;
}
.pin-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.pin-x {
  background: none; border: none; font-size: 17px; line-height: 1; color: #888;
  cursor: pointer; padding: 0 2px;
}
.pin-x:hover { color: #333; }
.pin-coords {
  display: flex; align-items: baseline; gap: 6px; width: 100%; margin-top: 6px;
  background: #f4f4f4; border: 1px solid #e2e2e2; border-radius: 5px;
  padding: 4px 6px; font: 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #333; cursor: pointer; text-align: left;
}
.pin-coords:hover { border-color: #bbb; }
.pin-copy { margin-left: auto; font-family: system-ui, sans-serif; color: #777; font-size: 10px; }
.pin-facts { margin: 7px 0 0; display: flex; flex-direction: column; gap: 3px; }
.pin-facts > div { display: flex; justify-content: space-between; gap: 10px; }
.pin-facts dt { color: #777; }
.pin-facts dd { margin: 0; font-weight: 600; text-align: right; }
.pin-mini {
  border: 0; background: transparent; padding: 0; cursor: pointer;
  font: inherit; font-weight: 600; font-variant-numeric: tabular-nums;
  color: inherit; text-decoration: underline dotted; text-underline-offset: 2px;
}
.pin-mini:hover { color: var(--accent, #2b7a3d); }
.pin-sample { margin-top: 8px; border-top: 1px solid var(--border-soft, #eee); padding-top: 8px; }
.pin-sample-btn {
  width: 100%; border: 1px solid var(--accent, #2b7a3d); background: transparent;
  color: var(--accent, #2b7a3d); border-radius: 6px; padding: 5px 8px;
  font: inherit; font-size: 0.78rem; font-weight: 600; cursor: pointer;
}
.pin-sample-btn:hover:not(:disabled) { background: var(--accent, #2b7a3d); color: #fff; }
.pin-sample-btn:disabled { opacity: 0.6; cursor: default; }
.pin-facts.sampled { margin-top: 7px; }
.pin-sw {
  display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 5px; vertical-align: -1px; border: 1px solid rgba(0, 0, 0, 0.2);
}
.pin-note { margin: 6px 0 0; font-size: 11px; line-height: 1.35; color: #777; }

@media (prefers-color-scheme: dark) {
  .pin-panel { background: rgba(32, 32, 34, 0.96); border-color: #444; color: #ddd; }
  .pin-coords { background: #2a2a2c; border-color: #444; color: #ddd; }
  .pin-facts dt, .pin-note, .pin-copy { color: #999; }
}
/* A caveat about the data being stretched should read as a caveat. */
.legend-note.upscaled { color: #8a5a1f; }
.legend-note.warn { color: #b3492f; }
/* The one link inside a legend card (remove a model surface). */
.overlay-legend .linkish {
  margin-top: 6px; background: none; border: 0; padding: 0;
  font: inherit; font-size: 12px; color: var(--accent); text-decoration: underline; cursor: pointer;
}

/* :deep — Leaflet builds the tooltip outside this component's tree. */
.map-shell :deep(.obs-tip) {
  max-width: 260px; padding: 7px 9px; font: 12px/1.45 system-ui, sans-serif;
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  box-shadow: 0 2px 10px var(--shadow);
}
.map-shell :deep(.obs-tip strong) { display: block; margin-bottom: 3px; font-style: italic; }
.map-shell :deep(.obs-tip .ot-row) { display: block; white-space: nowrap; }
.map-shell :deep(.obs-tip .ot-k) { color: var(--muted); margin-right: 6px; }
.map-shell :deep(.obs-tip::before) { border-top-color: var(--border); }
.tile-note { max-width: 260px; }
.tile-warn { max-width: 260px; border-color: #e0b4b4; background: rgba(255, 244, 244, 0.97); }
.tile-warn .legend-title { color: #b00020; }

/* Day-of-year window controls for the seasonal overlays. */
/* Collapsed, it is one chip the width of its own summary. Expanded, it floats
   over a solid panel rather than pushing the bar taller — which also keeps the
   bar's measured height, and so Leaflet's offset, stable. */

.season { position: relative; }
/* Fields inside a popover: label above control, full width. On the bar these
   were label-beside-control, which is what made them wide. */
.pop-field { display: flex; flex-direction: column; gap: 4px; }
.pop-field label {
  display: flex; align-items: center; gap: 5px;
  font-size: 0.78rem; color: var(--muted); white-space: nowrap;
}
.pop-field select { width: 100%; }
.pop-note { margin: 5px 0 0; font-size: 0.72rem; line-height: 1.4; color: var(--muted); }
.pop-note.warn { color: #8a5a1f; }

/* Layer rows. A whole row is the hit target, not just the box. */
/* The layer manager's opener. Sized and stated by the shared block above with
   the rest of the bar; these are only the parts inside it, which mirror
   PopoverMenu's so the two read as the same kind of control. */
.tool-icon { font-size: 0.95rem; line-height: 1; }
.tool-badge {
  background: var(--surface-2, #eee); border-radius: 999px;
  padding: 1px 6px; font-size: 0.7rem; color: var(--muted, #666);
}
@media (max-width: 720px) {
  /* As with the popovers: on a phone the icon carries it and the outline says
     something is on. */
  .tool-label, .tool-badge { display: none; }
  .controls .tool-btn { padding: 0 8px; gap: 0; }
}

/* The basemap radios. The overlay rows that also used this, and the tier badge
   that went with them, live in LayerManager.vue now. */
.lay-row {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 2px; font-size: 0.82rem; cursor: pointer; line-height: 1.3;
}
.lay-row:hover { color: var(--text); }
.lay-row input { flex: 0 0 auto; margin: 0; }
.lay-row span { flex: 1 1 auto; }
.pop-field input[type="range"] { width: 100%; margin: 0; accent-color: var(--accent); }


/* A control that is currently off reads as off, not just unstyled. */
.legend-row span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.legend-row.hoverable { cursor: default; border-radius: 4px; padding: 1px 3px; margin: 0 -3px; }
.legend-row.hoverable:hover { background: var(--surface-2, rgba(0, 0, 0, 0.06)); }
.legend-row.dim { opacity: 0.35; }
.swatch { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #222; flex: 0 0 auto; }
.gradient { height: 12px; border-radius: 3px; border: 1px solid #ccc; }
.gradient-scale { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; }
.gradient-ticks { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; font-variant-numeric: tabular-nums; }
.gradient-ticks span { flex: 1 1 0; text-align: center; }
.gradient-ticks span:first-child { text-align: left; }
.gradient-ticks span:last-child { text-align: right; }

/* Mobile: tighten the on-map controls and legend so they don't swallow the map. */
@media (max-width: 640px) {
  .controls { top: 8px; left: 8px; right: 8px; gap: 6px; }
  /* Two dropdowns to a row instead of one. Each pairing is natural — what the
     dots mean beside how big they are, the overlay beside its cell size — and
     it halves the number of rows the bar spends covering the map. */
  .colorby {
    padding: 5px 8px; font-size: 12px;
    flex: 1 1 calc(50% - 3px); min-width: 0; box-sizing: border-box;
  }
  .colorby label { flex: 0 0 auto; }
  .colorby select { flex: 1 1 auto; min-width: 0; }
  /* The popover buttons share the remaining row rather than each taking one. */
  .season { flex: 1 1 100%; }
  /* Both legends drop to the bottom, clear of the control bar, and share the
     space rather than the overlay one claiming the top. */
  .legends {
    top: auto; bottom: 8px; right: 8px; max-width: 70vw;
    gap: 6px; max-height: 46vh;
  }
  .overlay-legend { margin-bottom: 0; }
  .legend {
    max-height: 22vh; min-height: 0; padding: 8px 10px; font-size: 12px;
  }
}

/* Above Leaflet's own controls, which sit at z-index 1000. At 600 the layers
   control — a 48px square parked in the top-right corner — landed exactly on
   the drawer's close button and swallowed the click, so an observation could be
   opened and never dismissed. */
/* Raising the drawer settles the click, but it then covers the basemap picker
   it was fighting with. Where there is room, the picker steps aside instead of
   being buried, so basemaps can still be switched with an observation open. It
   moves in step with the drawer's own slide. On a narrow screen there is
   nowhere to step to — the control bar already owns the left — so the drawer
   simply covers it until it is closed. */
/* Below the control bar, whose height varies with how many rows it wraps into
   and whether the season sliders are open. The bar occupies the top of the map
   at every width, and the layers control now shares its corner, so this is no
   longer only a phone problem. */
/* The layers control carries nine reference layers. The group each belongs to
   is rendered ahead of its name so the list can be scanned by subject —
   Terrain, Weather, Ground, Vegetation, Context — rather than read end to end. */
.map-shell :deep(.leaflet-control-layers-overlays .lg) {
  display: inline-block; min-width: 68px; color: #777;
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em;
}
.map-shell :deep(.leaflet-control-layers-overlays label) { white-space: nowrap; }
/* Quiet: it marks a layer, it does not advertise at someone mid-task. */
.map-shell :deep(.leaflet-control-layers-overlays .lg-tier) {
  margin-left: 5px; padding: 1px 5px; border-radius: 999px;
  background: #eee; color: #777; font-size: 0.62rem; text-transform: uppercase;
  letter-spacing: 0.04em; vertical-align: 1px;
}

.map-shell :deep(.leaflet-top.leaflet-left) {
  transform: translateY(calc(var(--controls-h, 0px) + 4px));
}



.photos { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }


/* No fixed square: let the photo keep its own aspect ratio up to a height cap,
   so landscape shots aren't letterboxed into a small square. */
</style>
