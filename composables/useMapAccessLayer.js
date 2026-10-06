// The map's Access overlay (WANT-17): PAD-US / owner-set areas as polygons,
// loaded for the visible viewport only and styled by one attribute. Pure
// decisions live in accessLayer.ts; this is the Leaflet + state glue.

import { computed, onBeforeUnmount, reactive, ref, shallowRef, watch } from 'vue'
import { fetchAccess } from '~/composables/useAccess'
import {
  ACCESS_ATTRS, DEFAULT_SOURCES, accessNotice, filterAreas, planView, popupHtml, sourceCounts, styleForArea,
} from '~/composables/accessLayer'

const LOAD_DEBOUNCE_MS = 350

export function useMapAccessLayer({ mapRef, LRef, accessToken, user }) {
  const enabled = ref(false)
  const attr = ref('public')
  const switches = reactive({ free: false, public: false, collecting: false, includeLikely: false, includeUnknown: false })
  const sources = reactive({ ...DEFAULT_SOURCES })
  const status = ref('idle')
  const loading = ref(false)
  const plan = ref({ kind: 'zoom-in' })
  const areas = shallowRef([])
  const lines = shallowRef([])

  const visible = computed(() => filterAreas(areas.value, switches, sources))
  const counts = computed(() => sourceCounts(areas.value))
  const notice = computed(() => (enabled.value ? accessNotice(plan.value, status.value) : ''))
  const attrLabel = computed(() => ACCESS_ATTRS.find((a) => a.key === attr.value)?.label || '')

  let areaLayer = null
  let lineLayer = null
  let seq = 0
  let timer = null

  function ensurePane(map) {
    if (!map.getPane('access')) {
      map.createPane('access').style.zIndex = '380'
    }
  }

  function clearLayers() {
    const map = mapRef.value
    for (const l of [areaLayer, lineLayer]) if (l && map) map.removeLayer(l)
    areaLayer = lineLayer = null
  }

  function draw() {
    const map = mapRef.value, L = LRef.value
    if (!map || !L) return
    clearLayers()
    if (!enabled.value) return
    ensurePane(map)
    const feats = visible.value.map((a) => ({
      type: 'Feature', geometry: a.geom, properties: { id: a.id, _a: a },
    }))
    areaLayer = L.geoJSON({ type: 'FeatureCollection', features: feats }, {
      pane: 'access',
      style: (f) => {
        const s = styleForArea(f.properties._a, attr.value)
        return { ...s, dashArray: s.dashArray || undefined }
      },
      onEachFeature: (f, layer) => layer.bindPopup(() => popupHtml(f.properties._a), { maxWidth: 280, className: 'acc-popup' }),
    }).addTo(map)
    if (lines.value.length && plan.value.kind === 'ok' && plan.value.lines) {
      lineLayer = L.geoJSON({ type: 'FeatureCollection', features: lines.value }, {
        pane: 'access', interactive: false,
        style: (f) => (f.properties?.kind === 'trail'
          ? { color: '#8d6e63', weight: 1.5, dashArray: '2 4', opacity: 0.9 }
          : { color: '#555', weight: 1.5, opacity: 0.8 }),
      }).addTo(map)
    }
  }

  async function load() {
    const map = mapRef.value
    if (!map || !enabled.value) return
    const b = map.getBounds()
    plan.value = planView([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], map.getZoom())
    if (plan.value.kind !== 'ok') {
      areas.value = []; lines.value = []; status.value = 'idle'
      return
    }
    const mine = ++seq
    loading.value = true
    let token = null
    try { token = await accessToken?.() } catch { /* anonymous */ }
    const r = await fetchAccess(plan.value.bbox, (...a) => fetch(...a), { token, lines: plan.value.lines })
    if (mine !== seq) return // a newer viewport superseded this one
    loading.value = false
    areas.value = r.areas
    lines.value = r.lines || []
    status.value = r.status
  }

  const scheduleLoad = () => {
    clearTimeout(timer)
    timer = setTimeout(load, LOAD_DEBOUNCE_MS)
  }

  let bound = null
  watch([enabled, mapRef], ([on, map]) => {
    if (bound) { bound.off('moveend', scheduleLoad); bound = null }
    if (on && map) {
      map.on('moveend', scheduleLoad)
      bound = map
      load()
    } else {
      seq++; loading.value = false
      areas.value = []; lines.value = []; status.value = 'idle'
    }
    draw()
  }, { immediate: true })
  watch([visible, attr, lines, plan], draw)
  watch(() => user?.value?.id, () => { if (enabled.value) load() })

  onBeforeUnmount(() => {
    clearTimeout(timer)
    if (bound) bound.off('moveend', scheduleLoad)
    clearLayers()
  })

  return {
    enabled, attr, attrLabel, switches, sources, status, loading, plan, areas, counts, visible, notice,
  }
}
