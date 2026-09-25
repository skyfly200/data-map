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
      :order="overlayOrder" :opacity="layerOpacity" :ee-loading="eeLoading"
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

    <!-- Single key card, bottom-right, behind the control bar. All sections
         (layers, heatmap, observations) share one surface so they do not crowd
         each other and the card stays behind the toolbar at z-index 400. -->
    <div v-if="loaded" class="legends" :class="{ collapsed: keyCollapsed }">
      <button class="key-toggle" :aria-expanded="String(!keyCollapsed)"
              :title="keyCollapsed ? 'Show the map key' : 'Hide the map key'"
              @click="setKeyCollapsed(!keyCollapsed)">
        <span class="caret" aria-hidden="true">{{ keyCollapsed ? '▸' : '▾' }}</span>
        Key
      </button>

      <div class="key-card" @mouseleave="hoverValue = null">

        <!-- Status: unavailable layers and EE loading/errors -->
        <div v-if="tileErrors.length" class="key-section">
          <div class="key-sec-head warn-head">Layer unavailable</div>
          <div class="legend-note no-border">
            {{ tileErrors.join(', ') }} could not be reached.
          </div>
        </div>
        <div v-for="name in eeLoading.values()" :key="name" class="key-section ee-loading">
          <span class="ee-spinner" aria-hidden="true"></span>
          <span class="ee-loading-label">Rendering {{ name }}…</span>
        </div>
        <div v-for="e in eeErrors" :key="e.key" class="key-section">
          <div class="key-sec-head warn-head">{{ e.name }} failed</div>
          <div class="legend-note no-border">{{ e.message }}</div>
        </div>

        <!-- Reference tile layers -->
        <template v-if="activeTileNotes.length">
          <div class="key-section-label">Map layers</div>
          <div v-for="n in activeTileNotes" :key="n.name" class="key-section tk">
            <button class="tk-name-toggle" @click="toggleLayerExpanded(n.name)">
              <span class="tk-name">{{ n.name }}</span>
              <span class="tk-caret" aria-hidden="true">{{ expandedLayers[n.name] ? '▾' : '▸' }}</span>
            </button>
            <template v-if="expandedLayers[n.name]">
              <template v-if="n.legend?.type === 'ramp'">
                <div class="gradient" :style="{ background: gradientCss(n.legend.stops) }"></div>
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
              <SoilTaxonomyKey v-if="n.classes === 'great-groups' && (eeParams[n.ee]?.mode ?? 'orders') === 'classes'"
                               :layer="n.ee"
                               :selectable="!!n.eeParams?.codes"
                               :codes="(eeParams[n.ee] || {}).codes ?? (n.eeParams?.codes?.default || '')"
                               @update:codes="setEeParam(n.ee, 'codes', $event)" />
              <template v-for="(p, name) in (n.eeParams || {})" :key="name">
                <div v-if="p.type === 'zones'" class="layer-zones">
                  <span class="layer-zones-label">{{ p.label }}</span>
                  <label v-for="(v, i) in (p.values || [])" :key="v" class="zone-check">
                    <input type="checkbox"
                           :checked="((eeParams[n.ee] || {})[name] ?? p.default).toString().split(',').includes(v)"
                           @change="setEeParam(n.ee, name, toggleZone((eeParams[n.ee] || {})[name] ?? p.default, v, $event.target.checked))" />
                    {{ (p.labels || p.values)[i] }}
                  </label>
                </div>
                <div v-else-if="p.type !== 'codes'" class="layer-date">
                  <label :for="`ee-${n.slug}-${name}`">{{ p.label }}</label>
                  <select v-if="p.type === 'enum'" :id="`ee-${n.slug}-${name}`"
                          :value="(eeParams[n.ee] || {})[name] ?? p.default"
                          @change="setEeParam(n.ee, name, $event.target.value)">
                    <option v-for="v in (p.values || [])" :key="v" :value="v">{{ v }}</option>
                  </select>
                  <select v-else-if="p.type === 'yearSelect'" :id="`ee-${n.slug}-${name}`"
                          :value="(eeParams[n.ee] || {})[name] ?? p.default"
                          @change="setEeParam(n.ee, name, Number($event.target.value))">
                    <option v-for="v in (p.values || [])" :key="v" :value="v">{{ v }}</option>
                  </select>
                  <input v-else-if="p.type === 'text'" :id="`ee-${n.slug}-${name}`" type="search"
                         :maxlength="p.maxLength || 60" :placeholder="p.default"
                         :value="(eeParams[n.ee] || {})[name] ?? p.default"
                         @change="setEeParam(n.ee, name, $event.target.value)" />
                  <input v-else-if="p.type === 'date'" :id="`ee-${n.slug}-${name}`" type="date"
                         :min="typeof p.min === 'function' ? p.min() : p.min"
                         :max="typeof p.max === 'function' ? p.max() : p.max"
                         :value="(eeParams[n.ee] || {})[name] ?? (typeof p.default === 'function' ? p.default() : p.default)"
                         @change="setEeParam(n.ee, name, $event.target.value)" />
                  <input v-else :id="`ee-${n.slug}-${name}`" type="number" :min="p.min" :max="p.max"
                         :value="(eeParams[n.ee] || {})[name] ?? p.default"
                         @change="setEeParam(n.ee, name, Number($event.target.value))" />
                </div>
              </template>
              <div v-if="n.minZoom && mapView?.zoom < n.minZoom" class="legend-note zoom-in no-border">
                Zoom in to level {{ n.minZoom }} to see this layer.
              </div>
              <div v-if="n.time" class="layer-date">
                <label :for="`ld-${n.slug}`">Date</label>
                <input :id="`ld-${n.slug}`" v-model="tileDate" type="date" :max="maxTileDate"
                       :title="`Which day of ${n.name} to draw. Satellite products lag by days, so recent dates can be blank.`" />
              </div>
              <div v-if="upscaleNote(n)" class="legend-note">{{ upscaleNote(n) }}</div>
              <div v-if="n.slow" class="legend-note">Tiles arrive slowly first time.</div>
              <div v-if="n.note" class="legend-note">{{ n.note }}</div>
            </template>
          </div>
        </template>

        <!-- Heatmap section -->
        <template v-if="heatmapMode">
          <div class="key-section-label">Heatmap</div>
          <div v-if="!heatmapLegend" class="key-section">
            <div class="tk-name">{{ heatmapMeta.label }}</div>
            <div class="legend-note no-border">{{ emptyHeatmapReason }}</div>
          </div>
          <div v-else class="key-section">
            <div class="tk-name">{{ heatmapMeta.label }}</div>
            <template v-if="heatmapLegend.type === 'sequential'">
              <div class="gradient" :style="{ background: gradientCss(heatmapLegend.ramp) }"></div>
              <div class="gradient-scale"><span>{{ heatmapLegend.min }}</span><span>{{ heatmapLegend.max }}</span></div>
              <div class="legend-note tk-hover-note">{{ heatmapLegend.cells.toLocaleString() }} cells · {{ heatmapLegend.note }}</div>
            </template>
            <template v-else-if="heatmapLegend.type === 'vector'">
              <div class="gradient" :style="{ background: gradientCss(heatmapLegend.ramp) }"></div>
              <div class="gradient-scale"><span>{{ heatmapLegend.min }}</span><span>{{ heatmapLegend.max }}</span></div>
              <div class="legend-note tk-hover-note">
                {{ heatmapLegend.cells.toLocaleString() }} arrows · {{ heatmapLegend.note }}
              </div>
            </template>
            <template v-else-if="heatmapLegend.type === 'compass'">
              <div class="compass-key">
                <span v-for="item in heatmapLegend.items" :key="item.label" class="ck">
                  <span class="swatch" :style="{ background: item.color }"></span>{{ item.label }}
                </span>
              </div>
              <div class="legend-note tk-hover-note">{{ heatmapLegend.cells.toLocaleString() }} cells · {{ heatmapLegend.note }}</div>
            </template>
            <template v-else>
              <div v-for="item in heatmapLegend.items" :key="item.label" class="legend-row hoverable"
                   :class="{ dim: hoverValue && hoverValue !== item.label }"
                   @pointerenter="hoverEnter(item.label, $event)" @pointerleave="hoverLeave($event)"
                   @pointerup="pickValue(item.label, $event)">
                <span class="swatch" :style="{ background: item.color }"></span>
                <span><em>{{ item.label }}</em> <span class="legend-n">{{ item.n }}</span></span>
              </div>
              <div class="legend-note tk-hover-note">
                {{ heatmapLegend.total }} values · {{ heatmapLegend.note }}
              </div>
            </template>
          </div>
        </template>

        <!-- Observations section -->
        <template v-if="coloring">
          <div class="key-section-label">Observations</div>
          <div class="key-section">
            <div class="tk-name">{{ coloring.title }}</div>
            <template v-if="coloring.type === 'categorical'">
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
            <div v-if="coloring.match" class="legend-note tk-hover-note match">{{ coloring.match }}</div>
          </div>
        </template>

      </div><!-- .key-card -->
    </div>

    <!-- The same drawer the charts and analysis pages use, so the two cannot
         drift apart on what an observation is worth showing. `inline` keeps it
         inside the map shell rather than pinned over the site header. -->
    <ObservationDrawer inline :selected="selected" :show-map-link="false"
                       @close="selected = null" />
  </div>
</template>

<script setup>
// Leaflet CSS is loaded dynamically on mount so it does not bloat non-map routes.
import { nextTick, onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { hasValue, useObservations } from '~/composables/useObservations'
import { gradientCss } from '~/composables/ramps'
import { useAppearance } from '~/composables/useAppearance'
import { useGlossary } from '~/composables/useGlossary'
import GlossaryTooltip from '~/components/GlossaryTooltip.vue'
import { useMapPointStyle, FIELD_LABEL, fmtNum } from '~/composables/useMapPointStyle'
import { useMapPin } from '~/composables/useMapPin'
import { useMapLayerManager } from '~/composables/useMapLayerManager'
import { useMapTileDate, MAP_MAX_ZOOM } from '~/composables/useMapTileDate'
import { useModelOverlay } from '~/composables/useModelOverlay'
import { useMapHeatmapRenderer } from '~/composables/useMapHeatmapRenderer'
import { useMapSelection } from '~/composables/useMapSelection'
import { useMapLocate } from '~/composables/useMapLocate'
import { setupReferenceTileLayers } from '~/composables/useMapRefTileLayers'
import { setupElevBandLayer } from '~/composables/useMapElevBandLayer'

// Slow EE layers (Sentinel-2 composites) time out below this zoom — a single
// tile covers ~600 km² at zoom 8. Leaflet skips tile requests; the legend notes
// the threshold so both stay in sync from one place.
const EE_SLOW_MIN_ZOOM = 8

const { define: g } = useGlossary()

const {
  data, filteredData, load, loadProgressive, chunks, partial,
  speciesFilter, focusObservation, setFocusObservation, error: obsError,
} = useObservations()
watch(obsError, (msg) => { if (msg) useAppAlerts().error('Could not load observations — ' + msg) })
const { elevLabel, elevValue, tempValue, unit, tempUnit } = useUnits()
const { filters } = useFilters()
const live = useLiveClusters()
// On a phone the basemap, the heatmap and the clustering controls move inside
// the two windows that remain, rather than being three more buttons on a bar
// that already filled the width of the screen.
const { compact } = useCompactMap()
const appearance = useAppearance()
const { stackBlend } = appearance
const share = useShareState()

const {
  colorBy, sizeBy, hoverValue,
  colorOptions, colorCoverageNote, coloring, legendValues, sizeScale,
  radiusFor, markerStyle,
  hoverEnter, hoverLeave, pickValue,
  pointRadius, pointOpacity, pointOutline, colorSeed, activeColors, colorOverrides,
} = useMapPointStyle({ filteredData, live })

const mapEl = ref(null)
const mapRef = shallowRef(null)
const LRef = shallowRef(null)
// The active GeoJSON point layer; shared between heatmap renderer and point-style watch.
const geoLayerRef = shallowRef(null)

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
// The locate button lives in a Leaflet control rather than the Vue template, so
// its busy state is applied by hand. One class on one element is a smaller cost
// than teleporting a component into a control container.
let map, L
const { locating, locateError, setLocateBtn, locateMe } = useMapLocate({ mapRef, LRef })
// mapRef and LRef are set in onMounted so composables can reactively access them

// Holds enriched observation info (photos, description, etc.) fetched from iNaturalist API

// ─── Heatmaps ─────────────────────────────────────────────────────────────────
const {
  heatmaps,
  heatmapMode, heatmapCell, cellShape, seasonDay, seasonWindow,
  heatmapMeta, groupedModes, heatmapOpacity, tileOpacity, CELL_SIZES,
  heatmapResult, heatmapLegend, emptyHeatmapReason,
  renderHeatmap, heatmapCellIndex, heatmapCellAt,
} = useMapHeatmapRenderer({ mapRef, LRef, geoLayerRef, filteredData })

// ─── Reference tile layers ────────────────────────────────────────────────────
// Public raster services stacked over the basemap — relief, rainfall, land
// cover, greenness, soil moisture, trails, ownership. The catalogue and its
// keys live in composables/mapLayers.js; this wires them to Leaflet.
//
// Distinct from the Heatmap picker in the control bar, which bins the
// observations themselves. A layer covers the whole map because somebody else
// measured it everywhere; a heatmap covers only where people have looked.

// When focusing an observation, the next re-render must not refit/clear it.
// Leaflet owns the centre and zoom, so they are mirrored into a ref for the
// share link rather than read out of shared state.
const mapView = ref(null)

// ─── Tile date + UI state ─────────────────────────────────────────────────────
const {
  tileDate, maxTileDate, tileErrors, activeTileNotes, tileLayers,
  activeTileTemplates, keyCollapsed, setKeyCollapsed, upscaleNote, syncActiveTemplates,
} = useMapTileDate({ mapRef, mapView })

// The heatmap popover, so the keyboard shortcut can still reach the season
// controls now that they live inside it.
const heatmapPop = ref(null)

// Coloring, sizing, palette, per-value overrides and point styling all restyle
// the existing layer in place — no need to rebuild it, which would refit the
// view.
// Debounced: rapid-fire changes (filter slider drag, palette picker) collapse into
// one restyle pass instead of one per reactive tick across up to 48k markers.
let coloringDebounceTimer = null
watch([coloring, sizeScale, activeColors, colorOverrides, pointRadius, pointOpacity, pointOutline, colorSeed, hoverValue], () => {
  if (!geoLayerRef.value) return
  clearTimeout(coloringDebounceTimer)
  coloringDebounceTimer = setTimeout(() => {
    geoLayerRef.value.eachLayer((l) => {
      const style = markerStyle(l.feature.properties)
      l.setStyle(style)
      l.setRadius(style.radius)
    })
  }, 40)
})
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
  if (!map || !geoLayerRef.value) return
  if (v) { geoLayerRef.value.addTo(map); geoLayerRef.value.bringToFront() } else geoLayerRef.value.remove()
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
    const msg = err.message || 'Could not save the map.'
    saveError.value = msg
    useAppAlerts().error(msg)
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
const datasetLabel = computed(() => {
  const n = filteredData.value?.features?.length || 0
  return n ? `${n.toLocaleString()} observations` : ''
})

const shareTitle = computed(() => {
  const n = filteredData.value?.features?.length || 0
  const what = speciesFilter.value?.length === 1 ? speciesFilter.value[0] : 'mushroom observations'
  return `${n.toLocaleString()} ${what}: data-map`
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

const {
  overlayOrder, layerOpacity, layerBlend, soloKey,
  activeOverlays, overlayLayers, baseLayers, activeBase, activeBaseName,
  overlayGroups, eeParams, eeErrors, eeLoading, eeLayers, activeEeLayers,
  applyOverlayOrder, applyBlendModes, applySolo,
  setSolo, setLayerBlend, moveOverlay, setLayerOpacity, clearOverlays,
  paramsFor, debounceEeRefresh, setEeParam,
  setBase: _setBase, restoreBase: _restoreBase, restoreOverlays, restoreEeLayer,
  toggleOverlay: _toggleOverlay, toggleOverlayByKey: _toggleOverlayByKey,
  refreshEeLayer: _refreshEeLayer,
} = useMapLayerManager({ mapRef, tileOpacity, heatmaps, offline, eeTiles, maxEnt })

// Wrap the three functions that must also call syncActiveTemplates after running.
function setBase(key) { _setBase(key); syncActiveTemplates() }
function restoreBase() { _restoreBase(); syncActiveTemplates() }
function toggleOverlay(entry) { _toggleOverlay(entry); syncActiveTemplates() }
function toggleOverlayByKey(key) { _toggleOverlayByKey(key); syncActiveTemplates() }
async function refreshEeLayer(spec) { await _refreshEeLayer(spec); syncActiveTemplates() }

const expandedLayers = reactive({})
function toggleLayerExpanded(name) {
  expandedLayers[name] = !expandedLayers[name]
}

function toggleZone(current, zone, checked) {
  const set = new Set(String(current || '').split(',').filter(Boolean))
  checked ? set.add(zone) : set.delete(zone)
  return set.size ? [...set].sort().join(',') : zone
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
      minZoom: spec.slow ? EE_SLOW_MIN_ZOOM : 0,
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
          minZoom: spec.slow ? EE_SLOW_MIN_ZOOM : undefined,
          // A layer whose classes are too many to list in a key. The key shows
          // a browser for them instead; see SoilTaxonomyKey.
          classes: spec.classes,
          legendInBrowser: spec.legendInBrowser,
          slug: spec.name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
        }]
      }
      refreshEeLayer(spec)
    })
    // A 400 means the tile token has expired. Force a re-mint by clearing the
    // in-memory cache entry and requesting a fresh template. Debounced so a
    // screenful of simultaneously-failing tiles collapses into one round trip.
    layer.on('tileerror', () => {
      if (!eeLoading.value.has(spec.key)) {
        eeTiles.evict(spec.key)
        debounceEeRefresh(spec, 300)
      }
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

    // If this EE layer was active in the previous session, restore it now that
    // its catalogue entry exists. Static layers are restored in bulk after init;
    // EE layers arrive one at a time from the server and need per-layer recovery.
    restoreEeLayer(spec.key, layer)
  }
}

// ─── MaxEnt suitability layers ────────────────────────────────────────────────
// Each saved model config becomes a real Layer Manager entry. The tile template
// is minted lazily (model-tiles?model=configId) when the layer is first toggled
// on, so inactive models cost nothing.

function addMaxEntLayers(specs) {
  const map = mapRef.value
  const L = LRef.value
  if (!map || !L || !specs?.length) return

  for (const spec of specs) {
    if (eeLayers.has(spec.key)) continue  // already wired

    const layer = L.tileLayer('', {
      opacity: 0.7,
      maxZoom: MAP_MAX_ZOOM,
      updateWhenIdle: false,
      updateWhenZooming: true,
      className: 'model-suitability',
    })
    layer._baseOpacity = 0.7
    layer._spec = { key: spec.key, name: spec.name, maxent: true }
    eeLayers.set(spec.key, layer)
    tileLayers.push(layer)

    const configId = spec.key.replace('maxent:', '')
    layer.on('add', () => {
      if (!activeTileNotes.value.some((n) => n.name === spec.name)) {
        activeTileNotes.value = [...activeTileNotes.value, {
          name: spec.name,
          note: spec.note || 'MaxEnt habitat suitability surface.',
          legend: { type: 'ramp', min: '0', max: '1', stops: ['#2c2f6b', '#4a7db5', '#a1d99b', '#c6301f'] },
          slug: spec.name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
        }]
      }
      loadMaxEntTileUrl(spec.key, configId, layer)
    })
    layer.on('remove', () => {
      activeTileNotes.value = activeTileNotes.value.filter((n) => n.name !== spec.name)
      eeErrors.value = eeErrors.value.filter((e) => e.key !== spec.key)
    })

    // Update the Layer Manager entry to hold the real Leaflet layer.
    overlayLayers.value = overlayLayers.value.map((o) =>
      o.key === spec.key ? { ...o, layer } : o
    )
    restoreEeLayer(spec.key, layer)
  }
}

async function loadMaxEntTileUrl(key, configId, layer) {
  const map = mapRef.value
  if (!layer || !map?.hasLayer(layer)) return
  eeErrors.value = eeErrors.value.filter((e) => e.key !== key)
  const loadingNext = new Map(eeLoading.value)
  loadingNext.set(key, key.replace('maxent:', ''))
  eeLoading.value = loadingNext
  try {
    const token = await accessToken()
    const res = await fetch(`/.netlify/functions/model-tiles?model=${encodeURIComponent(configId)}`, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    })
    const body = await res.json()
    if (!res.ok || !body.ok || !body.template) throw new Error(body.error || 'Could not load model tiles.')
    layer.setUrl(body.template)
  } catch (err) {
    eeErrors.value = [...eeErrors.value, { key, name: key.replace('maxent:', ''), message: err.message }]
  } finally {
    const loadingDone = new Map(eeLoading.value)
    loadingDone.delete(key)
    eeLoading.value = loadingDone
  }
}

// When new model configs arrive (after a training job succeeds), wire their layers.
watch(() => maxEnt.maxentLayerSpecs.value, (specs) => {
  if (mapRef.value && LRef.value) addMaxEntLayers(specs)
})

// Open a model requested from /jobs or /modeling/maxent by enabling its Layer
// Manager entry. The layer mints its tile template lazily on first toggle.
const openModel = useModelOverlay()

function tryOpenPendingModel() {
  const p = openModel.pending.value
  if (!p || !mapRef.value) return
  const key = `maxent:${p.configId}`
  if (!activeOverlays.value.has(key) && overlayLayers.value.some((o) => o.key === key && o.layer)) {
    toggleOverlayByKey(key)
    openModel.clear()
  }
}

watch(() => openModel.pending.value, (p) => {
  if (p && mapRef.value) tryOpenPendingModel()
})

// ─── Dropped point ───────────────────────────────────────────────────────────
const {
  pin, copied,
  pinElevation, pinElevationText,
  pinSamples, pinSampling, pinSampleError,
  pinPlusCode, pinCell, pinCellValue, pinNearest,
  setPin, clearPin, copyPin, copyText,
  samplePinLayers, sampleText, heatmapCellAt: _pinHeatmapCellAt,
} = useMapPin({ mapRef, LRef, heatmapCellIndex, heatmapMode, heatmapCell, filteredData, activeEeLayers, accessToken, heatmaps })

// ─── Observation selection + point rendering ──────────────────────────────────
const { selected, selectedLatLng, renderPoints, applyFocus, setSuppressFit, setFittedOnce } = useMapSelection({
  mapRef, LRef, geoLayerRef, filteredData, chunks,
  focusObservation, setFocusObservation,
  coloring, colorBy, sizeBy,
  heatmapMode, heatmapMeta, heatmapCellAt,
  markerStyle, showPoints,
})

onMounted(async () => {
  try {
    await nextTick()
    if (!mapEl.value) throw new Error('map container not ready')
    await import('leaflet/dist/leaflet.css')
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
    }).setView([39.5, -105.7], 7)  // Colorado default; overridden below if a saved view exists
    mapRef.value = map
    LRef.value = L
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
        setLocateBtn(btn)
        // stop() as well as preventDefault: without it the click reaches the map
        // underneath and, in pin mode, drops a point behind the button.
        L.DomEvent.on(btn, 'click', (e) => { L.DomEvent.stop(e); locateMe() })
        L.DomEvent.disableClickPropagation(wrap)
        return wrap
      },
    })
    new LocateControl({ position: 'bottomleft' }).addTo(map)
    L.control.zoom({ position: 'bottomleft' }).addTo(map)

    // Reference tile services (public rasters: hillshade, land cover, trails…).
    const tileOverlayList = setupReferenceTileLayers({
      L, map, tileOpacity, tileDate, tileErrors, activeTileNotes,
      tileLayers, layerOpacity, heatmaps,
    })
    baseLayers.value = [
      { key: 'grey', name: 'Light gray', layer: grey },
      { key: 'greyDark', name: 'Dark gray', layer: greyDark },
      { key: 'osm', name: 'Street (OSM)', layer: osm },
      { key: 'topo', name: 'Terrain (OpenTopoMap)', layer: topo },
      { key: 'sat', name: 'Satellite (Esri)', layer: sat },
    ]
    restoreBase()
    overlayLayers.value = tileOverlayList
    restoreOverlays()

    // Elevation band canvas layer — decodes Terrarium DEM tiles, highlights
    // the elevation filter band or shows a hypsometric tint when none is set.
    tileOverlayList.push(setupElevBandLayer({ L, map, filters, activeTileNotes, tileLayers }))

    // Earth Engine layers arrive after their catalogue does, so they join the
    // list rather than being in it from the start.
    addEeLayers()

    // MaxEnt suitability layers — wire any models already in shared state, then
    // fetch so newly trained models are present and a pending overlay can open.
    addMaxEntLayers(maxEnt.maxentLayerSpecs.value)
    maxEnt.fetchModels().then(() => {
      addMaxEntLayers(maxEnt.maxentLayerSpecs.value)
      tryOpenPendingModel()
    })

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

    // Restore the last saved view so returning users land where they left off.
    // A shared link or fit-to-data will override this below.
    try {
      const saved = JSON.parse(localStorage.getItem('map-last-view') || 'null')
      if (saved?.bounds && saved?.zoom) {
        map.setView(
          [(saved.bounds.north + saved.bounds.south) / 2, (saved.bounds.east + saved.bounds.west) / 2],
          saved.zoom,
          { animate: false },
        )
      }
    } catch { /* ignore */ }

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
    if (shared.view) setSuppressFit(true)
    renderPoints(filteredData.value)
    if (shared.view) {
      map.setView(shared.view.center, shared.view.zoom, { animate: false })
      // The map is now deliberately placed, so the chunks that arrive for this
      // view must not refit it away.
      setFittedOnce(true)
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
  const m = map
  map = null
  mapRef.value = null
  if (m) m.remove()
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
  position: absolute; top: 12px; right: 12px; z-index: 500; display: flex; gap: 10px; align-items: center;
  flex-wrap: wrap; justify-content: flex-end;
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
/* Key column: bottom-right, behind the control bar (z-index 400 < controls 500). */
.legends {
  position: absolute; bottom: 18px; right: 12px; z-index: 400;
  transition: right 0.22s ease;
  display: flex; flex-direction: column; align-items: flex-end; gap: 6px;
  pointer-events: none; max-width: 220px; min-height: 0;
}
.legends > * { pointer-events: auto; }

.map-shell.drawer-open .legends { right: calc(var(--drawer-w) + 12px); }
.legends.collapsed .key-card { display: none; }

@media (max-width: 760px) {
  .map-shell.drawer-open .legends { right: 12px; }
  .map-shell.drawer-open .legends .key-card { display: none; }
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

/* Single consolidated card holding all key sections. */
.key-card {
  background: var(--surface, rgba(255, 255, 255, 0.95));
  border: 1px solid var(--border, #ddd); border-radius: 8px;
  padding: 8px 10px; font: 12px/1.4 system-ui, sans-serif;
  color: var(--text, #222); width: 200px; max-width: 200px;
  max-height: 44vh; overflow-y: auto; overscroll-behavior: contain;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}

/* Group label above each section (Map layers / Heatmap / Observations). */
.key-section-label {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--muted); margin: 6px 0 3px;
}
.key-section-label:first-child { margin-top: 0; }

/* One section = one layer or one key block. Rule between sections. */
.key-section + .key-section { border-top: 1px solid rgba(0, 0, 0, 0.08); margin-top: 6px; padding-top: 6px; }
.key-section-label + .key-section { margin-top: 0; }

.legend-row { display: flex; align-items: center; gap: 6px; }

.warn-head { font-weight: 600; font-size: 0.75rem; color: #b00020; margin-bottom: 2px; }

.tk-name { font-weight: 600; font-size: 0.78rem; margin-bottom: 3px; }

/* Layer explanation notes: hidden until the section is hovered. */
.tk-hover-note {
  display: none;
}
.key-section:hover .tk-hover-note { display: block; }

/* Swatch shapes: round for observations, square for layers. */
.key-section .swatch.dot { border-radius: 50%; }
.key-section .swatch { border-radius: 2px; }
.legend-note.match { font-style: italic; }

.class-key { display: flex; flex-wrap: wrap; gap: 2px 7px; margin: 2px 0 4px; }
.class-key .ck { display: inline-flex; align-items: center; gap: 3px; font-size: 0.7rem; }
.gradient-scale .unit { color: var(--muted); }
.layer-date { display: flex; align-items: center; gap: 6px; margin: 5px 0 2px; font-size: 0.72rem; }
.layer-date label { color: var(--muted); font-weight: 600; }
.layer-date input {
  flex: 1 1 auto; min-width: 0; background: var(--input-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: 2px 5px; font-size: 0.72rem;
}
.tk-name-toggle {
  display: flex; align-items: center; justify-content: space-between; width: 100%;
  background: none; border: none; padding: 0; cursor: pointer;
  text-align: left; color: inherit;
}
.tk-caret { font-size: 0.65rem; color: var(--muted); flex-shrink: 0; margin-left: 4px; }
.layer-zones { margin: 5px 0 2px; font-size: 0.72rem; }
.layer-zones-label { color: var(--muted); font-weight: 600; display: block; margin-bottom: 3px; }
.zone-check { display: flex; align-items: center; gap: 4px; margin: 2px 0; cursor: pointer; }
.zone-check input { cursor: pointer; margin: 0; }

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

.legend-note {
  margin-top: 4px; font-size: 11px; line-height: 1.35; color: #555;
}
.legend-note:not(.no-border) {
  border-top: 1px solid #e6e6e6; padding-top: 4px;
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
.legend-note.zoom-in { color: #3d6b8b; }
/* Link inside a key section (e.g. remove a model surface). */
.key-section .linkish {
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
.ee-loading { display: flex; align-items: center; gap: 7px; }
.ee-loading-label { font-size: 0.78em; color: var(--fg-muted, #666); }
@keyframes ee-spin { to { transform: rotate(360deg); } }
.ee-spinner {
  display: inline-block; width: 14px; height: 14px; flex-shrink: 0;
  border: 2px solid var(--border, #ccc);
  border-top-color: var(--accent, #3b82f6);
  border-radius: 50%;
  animation: ee-spin 0.7s linear infinite;
}

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
  .controls { top: 8px; left: 8px; right: 8px; gap: 6px; justify-content: flex-start; }
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
  .key-card {
    max-height: 38vh; min-height: 0; padding: 7px 9px; font-size: 11px; width: 180px; max-width: 180px;
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
