<template>
  <transition name="slide">
    <aside v-if="selected" class="drawer">
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

      <dl class="meta">
        <div v-if="selected.date"><dt>Observed</dt><dd>{{ selected.date }}</dd></div>
        <div v-if="selected.location"><dt>Location</dt><dd>{{ selected.location }}</dd></div>
        <div v-if="hasValue(selected.elevation)"><dt>Elevation</dt><dd>{{ elevLabel(selected.elevation) }}</dd></div>
        <div v-if="hasValue(selected.land_cover_label)"><dt>Land cover</dt><dd>{{ selected.land_cover_label }}</dd></div>
        <div v-if="hasValue(selected.cluster)"><dt>Cluster</dt><dd><span class="chip" :style="{ background: colorFor(selected.cluster) }">{{ selected.cluster }}</span></dd></div>
      </dl>
      <LeadUpCharts v-if="showCharts" :p="selected" />
      <div class="actions">
        <button v-if="showMapLink" class="act map" @click="openOnMap">Open on map ↗</button>
        <a v-if="inatUrl(selected)" :href="inatUrl(selected)" target="_blank" rel="noopener" class="act inat">View on iNaturalist ↗</a>
      </div>
    </aside>
  </transition>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { colorFor, hasValue, inatUrl, inatPhotoUrl, useObservations, fetchObservationDetails } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const props = defineProps({
  selected: { type: Object, default: null },
  // Show the "Open on map" action. False on the map itself (already there).
  showMapLink: { type: Boolean, default: true },
  // Show the weather lead-up mini-charts (needs the enriched history columns).
  showCharts: { type: Boolean, default: true },
})
const emit = defineEmits(['close'])

const { elevLabel } = useUnits()
const { setFocusObservation } = useObservations()

// Photo carousel: fetch the observation's photos from iNaturalist when a point
// is selected, at a large size (the API's bare `url` is a 75px square thumb).
const images = ref([])
const currentImageIndex = ref(0)

watch(() => props.selected, async (s) => {
  currentImageIndex.value = 0
  images.value = []
  if (!s) return
  const id = s.inat_id ?? s.uuid
  const details = await fetchObservationDetails(id)
  // Guard against a stale response landing after the user picked another point.
  if (props.selected !== s) return
  images.value = (details?.photos || []).map((p) => inatPhotoUrl(p, 'large')).filter(Boolean)
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
.meta div { display: grid; grid-template-columns: 84px 1fr; gap: 8px; }
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

.slide-enter-active, .slide-leave-active { transition: transform 0.2s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }
</style>
