<template>
  <div class="map-shell" :class="{ 'drawer-open': selected }">
    <div ref="mapEl" class="map"></div>

    <div v-if="loadError" class="overlay error">{{ loadError }}</div>
    <div v-else-if="!loaded" class="overlay">Loading observations…</div>

    <!-- Bottom-left, out of the way of the controls and the legends. It says
         what is happening rather than only that something is: "loading this
         area" is a different state from "showing a thinned sample", and a
         viewer zoomed out is looking at the second one. -->
    <div v-if="loaded && chunkStatus" class="chunk-status" :class="{ busy: chunks.busy.value }">
      <span v-if="chunks.busy.value" class="spinner" aria-hidden="true"></span>
      <span>{{ chunkStatus }}</span>
    </div>

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
    </div>

    <!-- Thematic layer selector -->
    <div v-if="loaded" ref="controlsEl" class="controls">
      <div class="colorby">
        <label for="colorby-sel">Color by <HelpLink option="map-color-by" /></label>
        <select id="colorby-sel" v-model="colorBy">
          <optgroup label="Category">
            <option v-for="o in colorOptions.category" :key="o.key" :value="o.key">{{ o.label }}</option>
          </optgroup>
          <optgroup v-if="colorOptions.numeric.length" label="Numeric">
            <option v-for="o in colorOptions.numeric" :key="o.key" :value="o.key">{{ o.label }}</option>
          </optgroup>
        </select>
      </div>
      <div v-if="colorOptions.numeric.length" class="colorby">
        <label for="sizeby-sel">Size by <HelpLink option="map-size-by" /></label>
        <select id="sizeby-sel" v-model="sizeBy">
          <option value="">Uniform</option>
          <option v-for="o in colorOptions.numeric" :key="o.key" :value="o.key">{{ o.label }}</option>
        </select>
      </div>
      <!-- A mode rather than a plain click handler: the map already uses a
           click to open an observation, and making an empty click drop a pin
           would put one down every time someone missed a dot. -->
      <button class="icon-btn" :class="{ on: pinMode }" :aria-pressed="String(pinMode)"
              :title="pinMode ? 'Click the map to place a point (Esc to stop)' : 'Drop a point on the map'"
              @click="togglePinMode">📍</button>
      <LiveClusterControls />
      <AppearanceControls icon-only :field="colorBy" :field-label="coloring.title"
                          :values="legendValues" />
      <ShareMenu icon-only :map-view="mapView" :color-by="colorBy" :size-by="sizeBy"
                 :title="shareTitle" />
      <!-- Everything you do not reach for every minute — the points toggle, the
           excluded-rows option and the two actions — lives behind one button.
           Spread across the bar they covered the map they were controlling. -->
      <!-- Actions are one tap each rather than two: they were folded into
           Settings to save bar space, but a button you press to DO something
           does not belong behind a menu of things you set. As icons they cost
           almost nothing. -->
      <button class="icon-btn" :class="{ busy: locating }"
              :title="locateError || tip('Centre the map on where you are', 'l')"
              aria-label="My location" @click="locateMe">
        <span class="dot-icon"></span>
      </button>
      <button class="icon-btn" :disabled="saving"
              :title="saveError || tip('Save the map, basemap and all, as a PNG', 'e')"
              aria-label="Save image" @click="saveMap">
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <path fill="currentColor" d="M12 16l-5-5h3V4h4v7h3l-5 5zm-7 2h14v2H5z" />
        </svg>
      </button>
      <MapSettings v-model="showPoints" :bounds="viewBounds" :templates="activeTileTemplates"
                   :dataset-label="datasetLabel" />

      <!-- Heatmap: grid summaries computed from the observations and drawn
           under the points. Named apart from the reference tile layers in the
           layers control, which are somebody else's imagery, not our numbers. -->
      <div class="colorby">
        <label for="overlay-sel">Heatmap <HelpLink :option="heatmapDocId" /></label>
        <select id="overlay-sel" v-model="heatmapMode" :title="heatmapTip">
          <option value="">None</option>
          <optgroup v-for="g in groupedModes" :key="g.label" :label="g.label">
            <option v-for="o in g.modes" :key="o.key" :value="o.key">{{ o.label }}</option>
          </optgroup>
        </select>
      </div>
      <div v-if="heatmapMode" class="colorby">
        <label for="overlay-cell">Cell size <HelpLink option="map-cell-size" /></label>
        <select id="overlay-cell" v-model.number="heatmapCell"
                title="Ground size of each grid cell. Smaller is more precise and noisier.">
          <option v-for="c in CELL_SIZES" :key="c.value" :value="c.value">{{ c.label }}</option>
        </select>
      </div>
      <!-- The date and window sliders were bare text and a bare track sitting
           directly on the map tiles, which made both unreadable and cost a full
           row of the bar. They now collapse to a summary of their own values,
           and expand over a solid panel. -->
      <div v-if="heatmapMode === 'season' || heatmapMode === 'hotspots'" ref="seasonEl" class="season">
        <button class="season-toggle" :class="{ on: seasonOpen }" :aria-expanded="String(seasonOpen)"
                :title="tip('Set the date and window the seasonal heatmaps use', 's')"
                @click="seasonOpen = !seasonOpen">
          <span class="s-label">Season</span>
          <strong>{{ seasonLabel }} · ±{{ seasonWindow }}d</strong>
          <span class="caret" aria-hidden="true">{{ seasonOpen ? '▴' : '▾' }}</span>
        </button>

        <div v-if="seasonOpen" class="season-panel">
          <div class="slider">
            <label for="season-day">
              Date <strong>{{ seasonLabel }}</strong> <HelpLink option="map-season-day" keys="[" />
              <!-- The slider opens on today, but the setting is remembered, so a
                   return visit lands wherever it was left. This is the way back. -->
              <button class="today-btn" :disabled="seasonDay === todayDay"
                      title="Centre the window on today"
                      @click="seasonDay = todayDay">Today</button>
            </label>
            <input id="season-day" v-model.number="seasonDay" type="range" min="1" max="365" step="1"
                   :title="tip(`Centre of the date window: currently ${seasonLabel}`, '[')" />
          </div>
          <div class="slider">
            <label for="season-window">
              Window <strong>±{{ seasonWindow }} days</strong> <HelpLink option="map-season-window" />
            </label>
            <input id="season-window" v-model.number="seasonWindow" type="range" min="3" max="60" step="1"
                   title="How wide a window counts as 'in season'. Wider is smoother and less specific." />
          </div>
          <p class="slider-note">{{ windowSpan }}</p>
        </div>
      </div>
    </div>

    <!-- Both legends share one column, so they cannot overlap each other or the
         control bar, and neither needs to know how tall the other is. -->
    <div v-if="loaded" class="legends">
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
      <div class="legend-title">Map layers</div>
      <div v-for="n in activeTileNotes" :key="n.name" class="tk">
        <div class="tk-name">{{ n.name }}</div>
        <template v-if="n.legend?.type === 'ramp'">
          <div class="gradient" :style="{ background: `linear-gradient(90deg, ${n.legend.stops.join(', ')})` }"></div>
          <div class="gradient-scale">
            <span>{{ n.legend.min }}</span>
            <span class="unit">{{ n.legend.unit }}</span>
            <span>{{ n.legend.max }}</span>
          </div>
        </template>
        <div v-else-if="n.legend?.type === 'classes'" class="class-key">
          <span v-for="c in n.legend.items" :key="c.label" class="ck">
            <span class="swatch" :style="{ background: c.color }"></span>{{ c.label }}
          </span>
        </div>
        <!-- The date the layer is showing, movable for the ones that vary. Each
             product has its own latency, so "today" is usually blank tiles. -->
        <!-- The knobs an Earth Engine layer exposes. Which year, how far back
             to look: these change what is rendered, so they re-mint the tiles. -->
        <div v-for="(p, name) in (n.eeParams || {})" :key="name" class="layer-date">
          <label :for="`ee-${n.slug}-${name}`">{{ p.label }}</label>
          <input :id="`ee-${n.slug}-${name}`" type="number" :min="p.min" :max="p.max"
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
    <div v-if="!heatmapLegend && heatmapMode && loaded" class="legend overlay-legend">
      <div class="legend-title">{{ heatmapMeta.label }}</div>
      <div class="legend-note">{{ emptyHeatmapReason }}</div>
    </div>

    <div v-if="heatmapLegend" class="legend overlay-legend">
      <div class="legend-title">{{ heatmapMeta.label }}</div>
      <template v-if="heatmapLegend.type === 'sequential'">
        <div class="gradient" :style="{ background: `linear-gradient(90deg, ${heatmapLegend.ramp[0]}, ${heatmapLegend.ramp[1]})` }"></div>
        <div class="gradient-scale"><span>{{ heatmapLegend.min }}</span><span>{{ heatmapLegend.max }}</span></div>
        <div class="legend-note">{{ heatmapLegend.cells.toLocaleString() }} cells · {{ heatmapLegend.note }}</div>
      </template>
      <template v-else-if="heatmapLegend.type === 'vector'">
        <div class="gradient" :style="{ background: `linear-gradient(90deg, ${heatmapLegend.ramp[0]}, ${heatmapLegend.ramp[1]})` }"></div>
        <div class="gradient-scale"><span>{{ heatmapLegend.min }}</span><span>{{ heatmapLegend.max }}</span></div>
        <div class="legend-note">
          Source: <strong>{{ heatmapLegend.source }}</strong> · colour = {{ heatmapLegend.colorBy }}<br />
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

    <!-- Legend (categorical swatches or a sequential gradient) -->
    <div v-if="coloring" class="legend" @mouseleave="hoverValue = null">
      <div class="legend-title">{{ coloring.title }}</div>
      <template v-if="coloring.type === 'categorical'">
        <!-- Hovering a row picks out the marks it stands for. A legend of twenty
             species otherwise leaves you matching hues by eye. -->
        <div v-for="item in coloring.legend" :key="item.label" class="legend-row hoverable"
             :class="{ dim: hoverValue && hoverValue !== item.label }"
             @pointerenter="hoverEnter(item.label, $event)"
             @pointerup="pickValue(item.label, $event)">
          <span class="swatch" :style="{ background: item.color }"></span>
          <span>{{ item.label }}</span>
        </div>
      </template>
      <template v-else>
        <div class="gradient" :style="{ background: `linear-gradient(90deg, ${RAMP[0]}, ${RAMP[1]})` }"></div>
        <div class="gradient-scale"><span>{{ fmtNum(coloring.min) }}</span><span>{{ fmtNum(coloring.max) }}</span></div>
      </template>
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
import { ALL_CATEGORY, ALL_NUMERIC } from '~/composables/useChartFields'
import { fieldValue } from '~/composables/statistics'
import { useAppearance } from '~/composables/useAppearance'
import { useUnits } from '~/composables/useUnits'

const {
  data, filteredData, load, loadProgressive, chunks, partial,
  speciesFilter, focusObservation, setFocusObservation,
} = useObservations()
const { elevLabel, elevValue, tempValue, unit, tempUnit } = useUnits()
const live = useLiveClusters()
const appearance = useAppearance()
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
const locating = ref(false)
const locateError = ref('')
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

const seasonLabel = computed(() => {
  const d = new Date(Date.UTC(2001, 0, 1))
  d.setUTCDate(seasonDay.value)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })
})

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
// What the corner indicator says. Silent once everything in view is loaded,
// because a permanent badge is furniture rather than information.
const chunkStatus = computed(() => {
  if (!chunks.available.value) return ''
  const s = chunks.stats.value
  if (chunks.busy.value) return 'Loading this area…'
  if (mapView.value && mapView.value.zoom < DETAIL_ZOOM && s.thinned > 1) {
    return `Showing 1 in ${s.thinned}. Zoom in for every record.`
  }
  if (partial.value) return `${s.loaded.toLocaleString()} of ${s.total.toLocaleString()} loaded`
  return ''
})

const seasonEl = ref(null)
const seasonOpen = ref(false)
const todayDay = heatmaps.todayOfYear()

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

const RAMP = ['#e8f1fb', '#0b3d91'] // sequential light → dark blue

// Field labels + which keys are categorical, drawn from the shared chart
// registry so the map and the Explore builder stay in sync.
const CATEGORY_KEYS = new Set(ALL_CATEGORY.map((f) => f.key))
const FIELD_LABEL = Object.fromEntries([...ALL_CATEGORY, ...ALL_NUMERIC].map((f) => [f.key, f.label]))

// Offer only the dimensions that actually carry data in the current dataset,
// so an un-enriched layer (e.g. NDVI still empty) doesn't yield an all-grey map.
const colorOptions = computed(() => {
  const feats = filteredData.value?.features || []
  const present = (list) => list.filter((f) => (
    f.key === 'live_cluster' ? live.active.value : feats.some((ft) => hasValue(ft.properties[f.key]))
  ))
  return { category: present(ALL_CATEGORY), numeric: present(ALL_NUMERIC) }
})

// If a dataset switch drops the active dimension's data, fall back to the first
// option still available (cluster, in practice).
watch(colorOptions, (opts) => {
  const keys = [...opts.category, ...opts.numeric].map((o) => o.key)
  if (keys.length && !keys.includes(colorBy.value)) colorBy.value = keys[0]
  // Drop a size field that the new dataset doesn't carry.
  if (sizeBy.value && !opts.numeric.some((o) => o.key === sizeBy.value)) sizeBy.value = ''
})

function fmtNum(v) { return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2) }

function hexLerp(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t))
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`
}

// Build the colour function + legend for the current "color by" dimension.
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
  // colours to the distinct values present, most frequent first. The legend is
  // capped (a dataset can have hundreds of species) with a "+N more" row.
  if (CATEGORY_KEYS.has(key)) {
    const counts = new Map()
    for (const f of feats) {
      const v = f.properties[key]
      if (hasValue(v)) counts.set(v, (counts.get(v) || 0) + 1)
    }
    const cats = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v)
    const LEGEND_CAP = 12
    // Stable per-value colours, so a species/year/class matches its colour in
    // the charts. Legend shows the most frequent values first.
    const legend = cats.slice(0, LEGEND_CAP).map((v) => ({ label: String(v), color: categoryColor(key, v) }))
    if (cats.length > LEGEND_CAP) legend.push({ label: `+${cats.length - LEGEND_CAP} more`, color: UNCLUSTERED })
    return {
      type: 'categorical', title, legend,
      colorFn: (p) => (hasValue(p[key]) ? categoryColor(key, p[key]) : UNCLUSTERED),
      labelOf: (p) => (hasValue(p[key]) ? String(p[key]) : null),
    }
  }

  // Numeric (sequential). Elevation and temperature follow the ft/m and °F/°C
  // settings, so the gradient scale matches the units shown elsewhere.
  const meta = ALL_NUMERIC.find((f) => f.key === key) || {}
  const conv = meta.unit === 'elev' ? elevValue : meta.unit === 'temp' ? tempValue : (v) => Number(v)
  const unitSuffix = meta.unit === 'elev' ? ` (${unit.value})` : meta.unit === 'temp' ? ` (°${tempUnit.value})` : ''
  const vals = feats.map((f) => f.properties[key]).filter(hasValue).map((v) => conv(Number(v)))
  const min = vals.length ? Math.min(...vals) : 0
  const max = vals.length ? Math.max(...vals) : 1
  return {
    type: 'sequential', title: title + unitSuffix, min, max,
    colorFn: (p) => {
      const raw = p[key]
      if (!hasValue(raw)) return UNCLUSTERED
      return hexLerp(RAMP[0], RAMP[1], (conv(Number(raw)) - min) / ((max - min) || 1))
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
 * fading the dots while their rings stayed solid turned a dense area into a grey
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

// Colouring, sizing, palette, per-value overrides and point styling all restyle
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
// selected, not the concept in general — each mode is misleading in its own way,
// and that caveat is the part worth one click. The field heatmaps share one
// entry: what they all get wrong is the same thing (a cell has a value only
// where somebody looked), and it is said once.
const HEATMAP_DOCS = {
  density: 'map-heatmap-density',
  richness: 'map-heatmap-richness',
  season: 'map-heatmap-season',
  hotspots: 'map-heatmap-hotspots',
  common: 'map-heatmap-common',
  land_cover: 'map-heatmap-land-cover',
  wind: 'map-heatmap-wind',
}
const heatmapDocId = computed(() => (
  heatmapMeta.value?.kind === 'field'
    ? 'map-heatmap-field'
    : HEATMAP_DOCS[heatmapMode.value] || 'map-heatmap'))

// Spelling out the window's actual dates removes the arithmetic from reading it.
const windowSpan = computed(() => {
  const fmt = (day) => {
    const d = new Date(Date.UTC(2001, 0, 1))
    d.setUTCDate(((day - 1 + 365) % 365) + 1)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })
  }
  return `Counting finds from ${fmt(seasonDay.value - seasonWindow.value)} to ${fmt(seasonDay.value + seasonWindow.value)}`
})

async function saveMap() {
  if (!mapEl.value || saving.value) return
  saving.value = true
  saveError.value = ''
  try {
    const blob = await exporter.mapToPng(mapEl.value, { scale: 2 })
    exporter.download(blob, `map-${exporter.slugify(colorBy.value, 'view')}-${exporter.stamp()}.png`)
  } catch (err) {
    saveError.value = err.message || 'Could not save the map.'
    console.error('Map export failed:', err)
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
      bounds: viewBounds.value, zoom: map.getZoom(), at: Date.now(),
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

// The URL templates of the layers actually drawn right now — the basemap in use
// plus whatever reference layers are switched on. Saving the whole catalogue
// would spend a viewer's data on layers they are not looking at.
const activeTileTemplates = ref([])
function syncActiveTemplates() {
  if (!map) return
  const out = []
  map.eachLayer((l) => {
    // ArcGIS export layers build their URLs per tile rather than from a
    // template, so they cannot be enumerated ahead of time and are skipped.
    if (l._url && typeof l._url === 'string' && l._url.includes('{z}')) out.push(l._url)
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
const eeParams = ref({})
const eeErrors = ref([])
let layersControl = null
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
    // than hidden: knowing the society computes it is part of what membership
    // is for. Ticking it explains itself through the error card.
    const gated = spec.tier && spec.tier !== 'free'
    const badge = gated ? ' <span class="lg-tier">members</span>' : ''
    layersControl?.addOverlay(layer, `<span class="lg">${spec.group}</span> ${spec.name}${badge}`)
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
  let next = Math.floor(Number(value))
  if (!Number.isFinite(next)) next = p?.default ?? 0
  if (p && Number.isFinite(p.min)) next = Math.max(p.min, next)
  if (p && Number.isFinite(p.max)) next = Math.min(p.max, next)

  eeParams.value = {
    ...eeParams.value,
    [key]: { ...(eeParams.value[key] || {}), [name]: next },
  }
  refreshEeLayer(spec)
}

// ─── Dropped point ───────────────────────────────────────────────────────────
// Somewhere the viewer picked, as opposed to somewhere a record exists. Held as
// plain numbers rather than a Leaflet marker so the panel can be reactive and
// the marker stays a detail of the map.
const pinMode = ref(false)
const pin = ref(null)
const copied = ref(false)
let pinMarker = null

function togglePinMode() {
  pinMode.value = !pinMode.value
  if (map) L.DomUtil[pinMode.value ? 'addClass' : 'removeClass'](map.getContainer(), 'picking')
}

function setPin(lat, lon) {
  pin.value = { lat, lon }
  copied.value = false
  if (!map || !L) return
  if (pinMarker) { pinMarker.setLatLng([lat, lon]); return }
  pinMarker = L.marker([lat, lon], {
    draggable: true,
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

  // The value behind this mark's colour, named by the dimension chosen.
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
  return co ? { ...feature.properties, lon: co[0], lat: co[1] } : feature.properties
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
  // it: the value behind its colour and size, and what the overlay reports for
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
    // Getting that wrong cost more than the base map. The default grey canvas
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
    // on its own; the moment 48k coloured dots sit on top of it, its own colour
    // is competing with the data for the same hues. A grey canvas gives the dots
    // the only saturation on screen — which is why it is the conventional base
    // for a point map, and why it is what this one opens with. Terrain is still
    // one click away, and the hillshade overlay puts relief back without colour.
    // Esri's grey canvas rather than CARTO's, which now demands an API key and
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
    const tileOverlays = {}
    for (const o of TILE_LAYERS) {
      const opts = {
        // The catalogue's maxZoom is where each service's tiles stop, which is
        // maxNativeZoom here. Passed as maxZoom it made every coarse layer —
        // all of Weather, Ground and Vegetation — vanish the moment the map was
        // zoomed past it, so ticking them appeared to do nothing at all.
        attribution: o.attribution, maxZoom: MAP_MAX_ZOOM, maxNativeZoom: o.maxZoom,
        opacity: (o.opacity ?? 1) * tileOpacity.value,
        crossOrigin: 'anonymous',
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
      tileOverlays[`<span class="lg">${o.group}</span> ${o.name}`] = layer
    }
    // One slider dims every reference layer at once, which is what you actually
    // want: they stack, and dimming them one at a time to see the data through
    // the pile is several controls doing one job.
    watch(tileOpacity, (v) => {
      for (const l of tileLayers) l.setOpacity(l._baseOpacity * v)
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
    layersControl = L.control.layers(
      {
        'Light grey': grey,
        'Dark grey': greyDark,
        'Street (OSM)': osm,
        'Terrain (OpenTopoMap)': topo,
        'Satellite (Esri)': sat,
      },
      // Leaflet's layers control takes one flat list, so the group rides in the
      // label — nine layers unlabelled is a wall, and grouping them by what
      // they are about is the difference between scanning and hunting.
      tileOverlays, { position: 'topleft', collapsed: true },
    ).addTo(map)

    // Earth Engine layers arrive after their catalogue does, so they join the
    // control rather than being in it from the start.
    addEeLayers()

    // Only while the mode is on, so an ordinary click that misses a dot still
    // does nothing rather than littering the map with pins.
    map.on('click', (e) => {
      if (!pinMode.value) return
      setPin(e.latlng.lat, e.latlng.lng)
    })

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

    heatmaps.loadFromStorage()
    appearance.loadFromStorage()

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
  { scope: 'Map', keys: 'e', label: 'Save the map as an image', run: () => saveMap() },
  { scope: 'Map', keys: '[', label: 'Heatmap date back a week', run: () => nudgeDay(-7) },
  { scope: 'Map', keys: ']', label: 'Heatmap date forward a week', run: () => nudgeDay(7) },
  { scope: 'Map', keys: 's', label: 'Season date and window', run: () => { seasonOpen.value = !seasonOpen.value } },
  { scope: 'Map', keys: 'escape', label: 'Close the observation drawer', run: () => { selected.value = null } },
])

// The bar renders behind v-if="loaded", so start measuring when it appears.
watch(loaded, (ok) => { if (ok) nextTick(trackControlsHeight) }, { immediate: true })

function onDocClick(e) {
  if (seasonOpen.value && seasonEl.value && !seasonEl.value.contains(e.target)) {
    seasonOpen.value = false
  }
}
// Escape leaves pin mode, and leaves it again to clear the pin. A mode with no
// keyboard way out is a trap on a laptop, where the toggle can be off-screen.
function onKeydown(e) {
  if (e.key !== 'Escape') return
  if (pinMode.value) pinMode.value = false
  else if (pin.value) clearPin()
  else return
  if (map) L.DomUtil.removeClass(map.getContainer(), 'picking')
}

onMounted(() => {
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('keydown', onKeydown)
  controlsResize?.disconnect()
  if (map) map.remove()
})
</script>

<style scoped>
.map-shell { --drawer-w: 320px; }

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
   themselves independently — the overlay legend pinned to the top, the colouring
   legend to the bottom — which works only while the control bar is a single row.
   On a phone the bar is three rows tall and the overlay legend landed on top of
   it; worse, the mobile rule added `bottom` without clearing the `top` it
   inherited, so the box was stretched between the two and rendered as a mostly
   empty panel half the height of the map.
   Placement now belongs to the container, and the legends only stack inside it,
   so neither has to know how tall the other is. */
.legends {
  position: absolute; bottom: 18px; right: 12px; z-index: 500;
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

.legend {
  position: static; z-index: 500;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 10px 12px; font: 13px/1.4 system-ui, sans-serif; color: #222; min-width: 120px;
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
.chunk-status {
  position: absolute; left: 12px; bottom: 92px; z-index: 500;
  display: flex; align-items: center; gap: 7px;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 5px 10px; font: 12px/1.35 system-ui, sans-serif; color: #333;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); max-width: 240px;
}
.spinner {
  width: 11px; height: 11px; flex: 0 0 auto; border-radius: 50%;
  border: 2px solid rgba(0, 0, 0, 0.15); border-top-color: #2b7a3d;
  animation: chunk-spin 0.8s linear infinite;
}
@keyframes chunk-spin { to { transform: rotate(360deg); } }
/* A viewer who has asked for no motion gets a pulse instead of a spin. */
@media (prefers-reduced-motion: reduce) {
  .spinner { animation: chunk-pulse 1.4s ease-in-out infinite; }
  @keyframes chunk-pulse { 50% { opacity: 0.35; } }
}

/* ── Touch ────────────────────────────────────────────────────────────────
   A fingertip is about 9mm across. The control bar's icon buttons ship at
   34px and Leaflet's zoom at 30px, both under half of that, so on a coarse
   pointer they get real hit areas — 40px of reach without 40px of bulk where
   the bulk would crowd the map. */
@media (pointer: coarse) {
  .map-shell .icon-btn,
  .map-shell :deep(.ap-btn.icon-only),
  .map-shell :deep(.sh-btn.icon-only),
  .map-shell :deep(.set-btn) { width: 40px; height: 40px; }

  .map-shell :deep(.leaflet-control-zoom a),
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
/* Pushed to the top of the column, leaving the colouring legend at the bottom —
   the arrangement this had before, now expressed as a relationship between the
   two rather than as two absolute positions that can collide. */
.overlay-legend { max-width: 280px; }
/* The layers key sits at the top of the column and the colouring key at the
   bottom, with the heatmap key between them — a relationship between the three
   rather than three absolute positions that can collide. */
.tile-note { margin-bottom: auto; }
.legends > .overlay-legend:first-child { margin-bottom: auto; }
.legend-note {
  margin-top: 6px; font-size: 11px; line-height: 1.35; color: #555;
  border-top: 1px solid #e6e6e6; padding-top: 5px;
}
.legend-n { color: #777; font-size: 11px; }

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
.pin-note { margin: 6px 0 0; font-size: 11px; line-height: 1.35; color: #777; }

@media (prefers-color-scheme: dark) {
  .pin-panel { background: rgba(32, 32, 34, 0.96); border-color: #444; color: #ddd; }
  .pin-coords { background: #2a2a2c; border-color: #444; color: #ddd; }
  .pin-facts dt, .pin-note, .pin-copy { color: #999; }
}
/* A caveat about the data being stretched should read as a caveat. */
.legend-note.upscaled { color: #8a5a1f; }

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
/* Square icon buttons, matching the other on-map controls' chrome. */
.icon-btn {
  display: inline-flex; align-items: center; justify-content: center;
  width: 34px; height: 34px; flex: 0 0 auto;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  color: #333; cursor: pointer; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); padding: 0;
}
.icon-btn:hover:not(:disabled) { background: #fff; }
/* A mode that is on has to look on, or the crosshair cursor is the only clue. */
.icon-btn.on { background: #2a78d6; border-color: #2a78d6; box-shadow: 0 0 0 2px rgba(42, 120, 214, 0.25); }
.icon-btn:disabled { opacity: 0.6; cursor: progress; }
.icon-btn.busy { opacity: 0.7; cursor: progress; }
.icon-btn .dot-icon {
  width: 11px; height: 11px; border-radius: 50%; background: #2a78d6;
  border: 2px solid #fff; box-shadow: 0 0 0 1px #2a78d6;
}

.season { position: relative; }
.season-toggle {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  padding: 7px 10px; font: 13px system-ui, sans-serif; color: #333; cursor: pointer;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); white-space: nowrap;
}
.season-toggle:hover, .season-toggle.on { background: #fff; }
.season-toggle .s-label { color: var(--muted); font-weight: 600; }
.season-toggle .caret { color: var(--muted); font-size: 10px; }

.season-panel {
  position: absolute; top: calc(100% + 6px); left: 0; z-index: 900; width: 260px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 12px;
  display: flex; flex-direction: column; gap: 10px; font-size: 0.8rem; color: var(--text);
}
.slider { display: flex; flex-direction: column; gap: 3px; }
.slider label { color: var(--muted); display: flex; align-items: center; gap: 6px; }
.slider label strong { color: var(--text); }
.season-panel input[type="range"] { width: 100%; margin: 0; accent-color: var(--accent); }
.slider-note { margin: 0; color: var(--muted); font-size: 0.74rem; line-height: 1.35; }
.today-btn {
  margin-left: auto; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 5px; padding: 2px 8px; font-size: 0.72rem;
  font-weight: 600; cursor: pointer;
}
.today-btn:hover:not(:disabled) { background: var(--surface-3); }
.today-btn:disabled { opacity: 0.4; cursor: default; }

@media (max-width: 640px) {
  .season-panel { width: min(260px, calc(100vw - 40px)); }
}

/* A control that is currently off reads as off, not just unstyled. */
.legend-row span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.legend-row.hoverable { cursor: default; border-radius: 4px; padding: 1px 3px; margin: 0 -3px; }
.legend-row.hoverable:hover { background: var(--surface-2, rgba(0, 0, 0, 0.06)); }
.legend-row.dim { opacity: 0.35; }
.swatch { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #222; flex: 0 0 auto; }
.gradient { height: 12px; border-radius: 3px; border: 1px solid #ccc; }
.gradient-scale { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; }

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
.obs-photo { max-width: 100%; height: auto; border-radius: 4px; }


/* Carousel styles */
/* No fixed square: let the photo keep its own aspect ratio up to a height cap,
   so landscape shots aren't letterboxed into a small square. */
.indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.5);
  cursor: pointer;
  transition: all 0.2s;
}
.indicator:hover {
  background: rgba(255, 255, 255, 0.8);
}
.indicator.active {
  background: #fff;
  transform: scale(1.2);
}
</style>
