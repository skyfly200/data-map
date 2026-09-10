<template>
  <transition name="slide">
    <aside v-if="selected" class="drawer" :class="{ inline }">
      <button class="close" aria-label="Close" @click="$emit('close')">×</button>
      <h3><em>{{ selected.species || 'Observation' }}</em></h3>

      <!-- Image Carousel -->
      <div v-if="images.length > 0" class="carousel">
        <div class="carousel-viewport">
          <img :src="images[currentImageIndex]" :alt="`Observation image ${currentImageIndex + 1}`" class="carousel-image" />
        </div>
        <button v-if="images.length > 1" class="carousel-nav prev" aria-label="Previous image" @click="prevImage">‹</button>
        <button v-if="images.length > 1" class="carousel-nav next" aria-label="Next image" @click="nextImage">›</button>
        <div v-if="images.length > 1" class="carousel-indicators">
          <span v-for="(img, idx) in images" :key="idx" class="indicator" :class="{ active: idx === currentImageIndex }" @click="currentImageIndex = idx"></span>
        </div>
        <div class="image-counter">{{ currentImageIndex + 1 }} / {{ images.length }}</div>
      </div>

      <!-- Sections fold. A fully enriched record runs to seventeen rows plus
           two charts, which is a lot of scrolling to reach the thing you opened
           it for. Record stays open because it is what identifies the record;
           the rest remember whether you had them open. -->
      <section v-for="sec in sections" :key="sec.title" class="group">
        <h4>
          <button type="button" class="sec-toggle" :aria-expanded="String(isOpen(sec.title))"
                  @click="toggleSection(sec.title)">
            <span class="caret" aria-hidden="true">{{ isOpen(sec.title) ? '▾' : '▸' }}</span>
            {{ sec.title }}
            <span v-if="!isOpen(sec.title)" class="sec-count">{{ sec.rows.length }}</span>
          </button>
        </h4>
        <dl v-show="isOpen(sec.title)" class="meta">
          <div v-for="r in sec.rows" :key="r.label" :class="{ warn: r.warn }">
            <!-- The label explains itself. On a pointer the title is enough; on
                 a touch screen there is no hover at all, so it is also a button
                 that expands the same text inline. Several of these values are
                 modelled rather than measured, and a bare "0.62" gives a reader
                 no way to know that. -->
            <dt>
              <button v-if="r.tip" type="button" class="tip-label" :title="r.tip"
                      :aria-expanded="String(openTips.has(r.label))"
                      @click="toggleTip(r.label)">{{ r.label }}</button>
              <template v-else>{{ r.label }}</template>
            </dt>
            <dd>
              {{ r.value }}
              <!-- A 0-1 index means little as a bare number; the bar puts it on
                   its own scale at a glance. -->
              <span v-if="r.bar !== undefined && r.bar !== null" class="bar" aria-hidden="true">
                <span :style="{ width: `${Math.round(r.bar * 100)}%` }"></span>
              </span>
              <small v-if="r.hint" class="hint">{{ r.hint }}</small>
              <small v-if="openTips.has(r.label)" class="hint tip-body">{{ r.tip }}</small>
            </dd>
          </div>
        </dl>
      </section>

      <dl v-if="hasValue(selected.cluster) || genus" class="meta">
        <!-- Built here rather than in detailSections because neither depends on
             the unit formatters, but they take their explanations from the same
             table so the drawer speaks with one voice. -->
        <div v-if="genus">
          <dt><button type="button" class="tip-label" :title="STAT_TIPS.Genus"
                      :aria-expanded="String(openTips.has('Genus'))"
                      @click="toggleTip('Genus')">Genus</button></dt>
          <dd>
            <em>{{ genus }}</em>
            <small v-if="openTips.has('Genus')" class="hint tip-body">{{ STAT_TIPS.Genus }}</small>
          </dd>
        </div>
        <div v-if="hasValue(selected.cluster)">
          <dt><button type="button" class="tip-label" :title="STAT_TIPS.Cluster"
                      :aria-expanded="String(openTips.has('Cluster'))"
                      @click="toggleTip('Cluster')">Cluster</button></dt>
          <dd>
            <span class="chip" :style="{ background: colorFor(selected.cluster) }">{{ selected.cluster }}</span>
            <small v-if="openTips.has('Cluster')" class="hint tip-body">{{ STAT_TIPS.Cluster }}</small>
          </dd>
        </div>
      </dl>

      <!-- Two different absences, and conflating them would be a lie. A thinned
           point carries only the handful of fields the map draws with; its
           terrain and weather exist and simply have not been fetched. Saying
           "no terrain data on this record" there would blame the pipeline for
           the loader's laziness. -->
      <p v-if="selected.__thinned" class="missing">
        Showing the overview copy of this record. Zoom in to load its full
        terrain and weather.
      </p>
      <p v-else-if="missing.length" class="missing">
        No {{ missing.join(', ') }} data on this record yet.
      </p>

      <div v-if="details && details.description" class="description">{{ details.description }}</div>

      <LeadUpCharts v-if="showCharts" :p="selected" />
      <div class="actions">
        <button v-if="showMapLink" class="act map" @click="openOnMap">Open on map ↗</button>
        <a v-if="inatUrl(selected)" :href="inatUrl(selected)" target="_blank" rel="noopener" class="act inat">View on iNaturalist ↗</a>
      </div>
    </aside>
  </transition>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import {
  LOCATION_PRECISION_LABELS, colorFor, hasValue, inatUrl, inatPhotoUrl,
  useObservations, fetchObservationDetails,
} from '~/composables/useObservations'
import { STAT_TIPS, detailSections, missingEnrichment } from '~/composables/observationDetail'
import { useUnits } from '~/composables/useUnits'

// Which sections are folded open. Remembered across records and across visits,
// because it is a preference about how much detail you want, not a decision
// about one observation — the opposite of the tips below, which are.
//
// Record is open by default: it is what tells you which record this is. The
// enrichment sections are the ones worth folding away.
const SECTIONS_KEY = 'drawer-sections'
const DEFAULT_OPEN = ['Record']
const openSections = ref(new Set(DEFAULT_OPEN))
const isOpen = (title) => openSections.value.has(title)

function toggleSection(title) {
  const next = new Set(openSections.value)
  if (next.has(title)) next.delete(title)
  else next.add(title)
  openSections.value = next
  try { localStorage.setItem(SECTIONS_KEY, JSON.stringify([...next])) } catch { /* private mode */ }
}

onMounted(() => {
  // Client only: the server has no localStorage, and rendering a section open
  // there and closed here is a hydration mismatch.
  try {
    const saved = JSON.parse(localStorage.getItem(SECTIONS_KEY) || 'null')
    if (Array.isArray(saved)) openSections.value = new Set(saved)
  } catch { /* ignore */ }
})

// Which explanations are expanded. Cleared when the record changes: they are
// about a field, but the reader opened them about THIS record, and leaving them
// open across a selection makes the next drawer look permanently annotated.
const openTips = ref(new Set())
function toggleTip(label) {
  const next = new Set(openTips.value)
  if (next.has(label)) next.delete(label)
  else next.add(label)
  openTips.value = next
}

const props = defineProps({
  selected: { type: Object, default: null },
  // Show the "Open on map" action. False on the map itself (already there).
  showMapLink: { type: Boolean, default: true },
  // Show the weather lead-up mini-charts (needs the enriched history columns).
  showCharts: { type: Boolean, default: true },
  // The map hosts this inside its own shell, below the header; everywhere else
  // it is pinned to the viewport.
  inline: { type: Boolean, default: false },
})
const emit = defineEmits(['close'])

const { elevLabel, tempLabel } = useUnits()
const { setFocusObservation } = useObservations()

// Elevation and temperature follow the ft/m and °F/°C toggles, so the formatters
// are handed to the section builder rather than looked up inside it.
const sections = computed(() => detailSections(props.selected, {
  elevLabel,
  tempLabel,
  precisionLabel: (k) => LOCATION_PRECISION_LABELS[k] || k,
}))
const missing = computed(() => missingEnrichment(props.selected))
const genus = computed(() => {
  const s = props.selected
  if (!s) return ''
  if (hasValue(s.genus)) return s.genus
  return hasValue(s.species) ? String(s.species).trim().split(/\s+/)[0] : ''
})

// Photo carousel: fetch the observation's photos from iNaturalist when a point
// is selected, at a large size (the API's bare `url` is a 75px square thumb).
const images = ref([])
const currentImageIndex = ref(0)
// The same fetch already ran for the photos; keeping the record lets the
// observer's own notes show without a second request.
const details = ref(null)

watch(() => props.selected, async (s) => {
  currentImageIndex.value = 0
  images.value = []
  details.value = null
  openTips.value = new Set()
  if (!s) return
  const id = s.inat_id ?? s.uuid
  const fetched = await fetchObservationDetails(id)
  // Guard against a stale response landing after the user picked another point.
  if (props.selected !== s) return
  details.value = fetched
  images.value = (fetched?.photos || []).map((p) => inatPhotoUrl(p, 'large')).filter(Boolean)
}, { immediate: true })

function prevImage() {
  if (!images.value.length) return
  currentImageIndex.value = (currentImageIndex.value - 1 + images.value.length) % images.value.length
}
function nextImage() {
  if (!images.value.length) return
  currentImageIndex.value = (currentImageIndex.value + 1) % images.value.length
}

function openOnMap() {
  const s = props.selected
  if (!s) return
  // Hand the map a target it can select + pan to, then navigate there.
  setFocusObservation({
    uuid: s.uuid, inat_id: s.inat_id,
    lon: s.lon ?? s.longitude, lat: s.lat ?? s.latitude,
    species: s.species,
  })
  emit('close')
  navigateTo('/map')
}
</script>

<style scoped>
.drawer {
  position: fixed; top: 0; right: 0; z-index: 1200; width: 340px; max-width: 92vw;
  height: 100%; background: var(--surface); box-shadow: -2px 0 16px var(--shadow);
  padding: 16px 18px; overflow-y: auto; font: 14px/1.45 system-ui, sans-serif; color: var(--text);
}
.drawer h3 { margin: 0 26px 10px 0; font-size: 1.05rem; }

/* Photo carousel. The viewport has no fixed aspect ratio so a photo keeps its
   own shape up to a height cap, instead of being boxed into a small square. */
.carousel { position: relative; margin: 0 0 14px; border-radius: 8px; overflow: hidden; background: #000; }
.carousel-viewport { width: 100%; min-height: 150px; display: flex; align-items: center; justify-content: center; }
.carousel-image { width: 100%; height: auto; max-height: 320px; object-fit: contain; display: block; }
.carousel-nav {
  position: absolute; top: 50%; transform: translateY(-50%); border: 0;
  background: rgba(0, 0, 0, 0.5); color: #fff; cursor: pointer;
  width: 28px; height: 44px; font-size: 1.4rem; line-height: 1; padding: 0;
}
.carousel-nav:hover { background: rgba(0, 0, 0, 0.72); }
.carousel-nav.prev { left: 0; border-radius: 0 6px 6px 0; }
.carousel-nav.next { right: 0; border-radius: 6px 0 0 6px; }
.carousel-indicators {
  position: absolute; bottom: 8px; left: 0; right: 0; display: flex;
  justify-content: center; gap: 6px;
}
.carousel-indicators .indicator {
  width: 7px; height: 7px; border-radius: 50%; background: rgba(255, 255, 255, 0.45);
  cursor: pointer; transition: background 0.15s;
}
.carousel-indicators .indicator.active { background: #fff; }
.image-counter {
  position: absolute; top: 8px; right: 8px; background: rgba(0, 0, 0, 0.55);
  color: #fff; font-size: 0.72rem; padding: 2px 7px; border-radius: 10px;
}
.close {
  position: absolute; top: 8px; right: 10px; border: 0; background: transparent;
  font-size: 1.5rem; line-height: 1; color: var(--muted); cursor: pointer;
}
.meta { margin: 0 0 14px; display: grid; gap: 5px; }
.meta div { display: grid; grid-template-columns: 92px 1fr; gap: 8px; align-items: start; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; }
.chip { display: inline-block; min-width: 20px; padding: 0 7px; border-radius: 10px; color: #fff; font-weight: 600; text-align: center; }

.actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.act {
  display: inline-block; border-radius: 6px; padding: 7px 12px; font-size: 0.85rem; font-weight: 600;
  text-decoration: none; cursor: pointer; border: 1px solid var(--border);
}
.act.map { background: var(--surface-2); color: var(--text); }
.act.map:hover { background: var(--surface-3); }
.act.inat { background: #2b7a3d; color: #fff; border-color: #2b7a3d; }
.act.inat:hover { background: #246833; }


/* Hosted inside the map's own shell rather than pinned to the viewport, so it
   sits below the site header instead of over it. Above Leaflet's controls,
   which live at z-index 1000 and would otherwise take the close button's tap. */
.drawer.inline { position: absolute; height: 100%; z-index: 1100; }

.group { margin: 0 0 12px; }
.group h4 {
  margin: 0 0 6px; font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted); font-weight: 700;
}
/* The whole heading is the hit target, not just the caret: a 0.7rem triangle
   is not something to ask a thumb to find. */
.sec-toggle {
  display: flex; align-items: center; gap: 6px; width: 100%;
  background: none; border: 0; padding: 3px 0; font: inherit; color: inherit;
  text-transform: inherit; letter-spacing: inherit; cursor: pointer; text-align: left;
}
.sec-toggle:hover { color: var(--text); }
.sec-toggle .caret { font-size: 0.6rem; opacity: 0.8; }
.sec-count {
  margin-left: auto; font-weight: 600; letter-spacing: 0;
  background: var(--surface-2, #eee); border-radius: 999px; padding: 0 6px;
}

.meta div.warn dd { color: var(--danger, #b00020); }
.hint { display: block; color: var(--muted); font-size: 0.72rem; line-height: 1.35; margin-top: 2px; }
/* The label is the affordance. A dotted underline says "there is more here"
   without putting a row of question marks down a 340px panel. */
.tip-label {
  background: none; border: 0; padding: 0; font: inherit; color: inherit;
  text-align: left; cursor: help;
  text-decoration: underline dotted; text-underline-offset: 2px;
  text-decoration-color: var(--border);
}
.tip-label:hover, .tip-label:focus-visible { color: var(--text); text-decoration-color: currentColor; }
.tip-body { border-left: 2px solid var(--border); padding-left: 6px; margin-top: 4px; }
.bar {
  display: block; height: 4px; border-radius: 2px; background: var(--surface-3, #e6e6e6);
  margin-top: 4px; overflow: hidden;
}
.bar > span { display: block; height: 100%; background: var(--accent, #2b7a3d); }

.missing {
  margin: 0 0 12px; padding: 7px 10px; border-radius: 6px; font-size: 0.76rem;
  color: var(--muted); background: var(--surface-2); border: 1px solid var(--border);
}
.description {
  margin: 0 0 14px; font-size: 0.84rem; line-height: 1.5; color: var(--text);
  white-space: pre-wrap; max-height: 160px; overflow-y: auto;
}

.slide-enter-active, .slide-leave-active { transition: transform 0.2s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }
</style>
