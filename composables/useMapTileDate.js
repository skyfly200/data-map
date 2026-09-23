import { onMounted, ref } from 'vue'
import { TILE_LAYERS, layerDate } from '~/composables/mapLayers'

export const MAP_MAX_ZOOM = 19

const KEY_COLLAPSED = 'map-key-collapsed'

export function useMapTileDate({ mapRef, mapView }) {
  const DEFAULT_LAG = Math.min(...TILE_LAYERS.filter((l) => l.time).map((l) => l.lag ?? 1))
  const tileDate = ref(layerDate(DEFAULT_LAG))
  const maxTileDate = layerDate(0)

  const tileErrors = ref([])
  const activeTileNotes = ref([])
  // Plain array — the opacity slider reaches these after setup; no reactivity needed.
  const tileLayers = []

  const activeTileTemplates = ref([])

  const keyCollapsed = ref(false)
  function setKeyCollapsed(v) {
    keyCollapsed.value = v
    try { localStorage.setItem(KEY_COLLAPSED, v ? '1' : '0') } catch { /* private mode */ }
  }
  onMounted(() => {
    try { keyCollapsed.value = localStorage.getItem(KEY_COLLAPSED) === '1' } catch { /* ignore */ }
  })

  function upscaleNote(n) {
    const zoom = mapView.value?.zoom
    if (!n?.native || !Number.isFinite(zoom) || zoom <= n.native) return ''
    return `Zoomed past this layer's detail — the tiles are stretched from zoom ${n.native}, not resolved finer.`
  }

  function syncActiveTemplates() {
    const map = mapRef.value
    if (!map) return
    const out = []
    map.eachLayer((l) => {
      if (!l._url || typeof l._url !== 'string' || !l._url.includes('{z}')) return
      out.push({
        template: l._url,
        id: l._spec?.ee ? l._spec.key : l._url,
        name: l._spec?.name || '',
        maxZoom: Number.isFinite(l.options?.maxNativeZoom) ? l.options.maxNativeZoom : null,
      })
    })
    activeTileTemplates.value = out
  }

  return {
    tileDate, maxTileDate, tileErrors, activeTileNotes, tileLayers,
    activeTileTemplates, keyCollapsed, setKeyCollapsed, upscaleNote, syncActiveTemplates,
  }
}
