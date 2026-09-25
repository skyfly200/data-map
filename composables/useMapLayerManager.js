import { computed, ref, watch } from 'vue'
import { drawnKeys, effectiveBlend, reorderStack } from '~/composables/blendModes'
import { useAppearance } from '~/composables/useAppearance'
import { normaliseCodes } from '~/netlify/lib/ee-tile-layers.mjs'

const BASE_KEY = 'map-basemap'

export function useMapLayerManager({ mapRef, tileOpacity, heatmaps, offline, eeTiles, maxEnt }) {
  const appearance = useAppearance()
  const { stackBlend } = appearance

  const overlayOrder = ref([])
  const layerOpacity = ref({})
  const layerBlend = ref({})
  const soloKey = ref('')
  const activeOverlays = ref(new Set())
  const overlayLayers = ref([])
  const baseLayers = ref([])
  const activeBase = ref('grey')
  const eeParams = ref({})
  const eeErrors = ref([])
  const eeLoading = ref(new Map())
  const eeLayers = new Map()

  const overlayGroups = computed(() => {
    const groups = new Map()
    for (const o of overlayLayers.value) {
      if (!groups.has(o.group)) groups.set(o.group, [])
      groups.get(o.group).push(o)
    }
    return [...groups.entries()].map(([label, items]) => ({ label, items }))
  })

  const activeBaseName = computed(() =>
    baseLayers.value.find((b) => b.key === activeBase.value)?.name || '')

  watch(maxEnt.maxentLayerSpecs, (specs) => {
    overlayLayers.value = [
      ...overlayLayers.value.filter((o) => !o.key.startsWith('maxent:')),
      ...specs.map((s) => ({
        key: s.key, name: s.name, group: s.group, note: s.note, layer: null,
      })),
    ]
  }, { immediate: true })

  function applyOverlayOrder() {
    const map = mapRef.value
    if (!map) return
    const n = overlayOrder.value.length
    overlayOrder.value.forEach((key, i) => {
      const entry = overlayLayers.value.find((o) => o.key === key)
      entry?.layer?.setZIndex?.(200 + (n - i))
    })
  }

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

  // Defers removeLayer until after any in-progress zoom animation to avoid
  // the Leaflet _updateLevels null-_map crash (rAF already queued when removal fires).
  function safeRemoveLayer(layer) {
    const map = mapRef.value
    if (!map || !map.hasLayer(layer)) return
    if (map._animatingZoom) {
      map.once('moveend', () => { if (map.hasLayer(layer)) map.removeLayer(layer) })
    } else {
      map.removeLayer(layer)
    }
  }

  function applySolo() {
    const map = mapRef.value
    if (!map) return
    const drawn = new Set(drawnKeys([...activeOverlays.value], soloKey.value))
    for (const key of activeOverlays.value) {
      const entry = overlayLayers.value.find((o) => o.key === key)
      if (!entry?.layer) continue
      const on = map.hasLayer(entry.layer)
      if (drawn.has(key) && !on) entry.layer.addTo(map)
      else if (!drawn.has(key) && on) safeRemoveLayer(entry.layer)
    }
    applyOverlayOrder()
    applyBlendModes()
  }

  function setSolo(key) {
    soloKey.value = key === soloKey.value ? '' : key
    applySolo()
  }

  function setLayerBlend(key, mode) {
    const next = { ...layerBlend.value }
    if (mode) next[key] = mode
    else delete next[key]
    layerBlend.value = next
    applyBlendModes()
  }

  function toggleOverlay(entry) {
    const map = mapRef.value
    if (!map) return
    const wasOn = activeOverlays.value.has(entry.key)
    const next = new Set(activeOverlays.value)
    if (wasOn) {
      next.delete(entry.key)
      overlayOrder.value = overlayOrder.value.filter((k) => k !== entry.key)
      if (entry.layer && map.hasLayer(entry.layer)) safeRemoveLayer(entry.layer)
      if (soloKey.value === entry.key) soloKey.value = ''
    } else {
      next.add(entry.key)
      overlayOrder.value = [entry.key, ...overlayOrder.value]
      soloKey.value = ''
      // Seed layerOpacity from the layer's spec default so the slider starts at
      // the actual rendered opacity, not always 100%. Only when no value is
      // already stored (first-ever toggle; localStorage restore sets it explicitly).
      if (layerOpacity.value[entry.key] == null && entry.layer?._baseOpacity != null) {
        layerOpacity.value = { ...layerOpacity.value, [entry.key]: entry.layer._baseOpacity }
      }
    }
    activeOverlays.value = next
    applySolo()
  }

  function toggleOverlayByKey(key) {
    const entry = overlayLayers.value.find((o) => o.key === key)
    if (entry) toggleOverlay(entry)
  }

  function moveOverlay(key, delta) {
    overlayOrder.value = reorderStack(overlayOrder.value, key, delta)
    applyOverlayOrder()
    applyBlendModes()
  }

  function setLayerOpacity(key, value) {
    const entry = overlayLayers.value.find((o) => o.key === key)
    if (!entry) return
    layerOpacity.value = { ...layerOpacity.value, [key]: value }
    // layerOpacity is the full opacity (0–1); _baseOpacity is the initial default,
    // not a persistent multiplier. Once the user drags the slider the value IS
    // the opacity — tileOpacity is the only remaining multiplier.
    entry.layer.setOpacity(value * tileOpacity.value)
    heatmaps.persist()
  }

  function clearOverlays() {
    soloKey.value = ''
    for (const key of [...activeOverlays.value]) toggleOverlayByKey(key)
  }

  watch(stackBlend, () => applyBlendModes())

  const OVERLAY_STATE_KEY = 'map-overlay-state'

  function saveOverlayState() {
    if (!import.meta.client) return
    try {
      localStorage.setItem(OVERLAY_STATE_KEY, JSON.stringify({
        active: [...activeOverlays.value],
        order: overlayOrder.value,
        opacity: layerOpacity.value,
      }))
    } catch { /* private mode or quota */ }
  }

  function restoreOverlays() {
    if (!import.meta.client) return
    let saved = null
    try { saved = JSON.parse(localStorage.getItem(OVERLAY_STATE_KEY) || 'null') } catch { return }
    if (!saved) return
    const { active = [], order = [], opacity = {} } = saved
    const available = new Set(overlayLayers.value.map((o) => o.key))
    const toRestore = active.filter((k) => available.has(k))
    for (const key of toRestore) {
      const entry = overlayLayers.value.find((o) => o.key === key)
      if (entry) toggleOverlay(entry)
    }
    const restoredSet = new Set(toRestore)
    const restoredOrder = order.filter((k) => restoredSet.has(k))
    if (restoredOrder.length) overlayOrder.value = restoredOrder
    const opacityEntries = Object.entries(opacity).filter(([k]) => restoredSet.has(k))
    if (opacityEntries.length) {
      layerOpacity.value = { ...layerOpacity.value, ...Object.fromEntries(opacityEntries) }
      for (const [key, value] of opacityEntries) {
        const entry = overlayLayers.value.find((o) => o.key === key)
        if (entry?.layer) entry.layer.setOpacity(value * tileOpacity.value)
      }
    }
    applyOverlayOrder()
  }

  function restoreEeLayer(key, layer) {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem(OVERLAY_STATE_KEY) || 'null')
      if (saved?.active?.includes(key) && !activeOverlays.value.has(key)) {
        toggleOverlay({ key, layer })
        const op = saved.opacity?.[key]
        if (op != null) {
          layerOpacity.value = { ...layerOpacity.value, [key]: op }
          layer.setOpacity(op * tileOpacity.value)
        }
      }
    } catch { /* storage unavailable */ }
  }

  watch([activeOverlays, overlayOrder, layerOpacity], saveOverlayState, { deep: true })

  function paramsFor(spec) {
    const held = eeParams.value[spec.key] || {}
    const out = {}
    for (const [name, p] of Object.entries(spec.params || {})) {
      out[name] = held[name] ?? p.default
    }
    return out
  }

  async function refreshEeLayer(spec) {
    const map = mapRef.value
    const layer = eeLayers.get(spec.key)
    if (!layer || !map?.hasLayer(layer)) return
    eeErrors.value = eeErrors.value.filter((e) => e.key !== spec.key)
    const loadingNext = new Map(eeLoading.value)
    loadingNext.set(spec.key, spec.name)
    eeLoading.value = loadingNext
    try {
      const minted = await eeTiles.template(spec.key, paramsFor(spec))
      layer.setUrl(minted.template)
      offline.registerEeTemplate(spec.key, minted.template)
    } catch (err) {
      eeErrors.value = [...eeErrors.value, { key: spec.key, name: spec.name, message: err.message }]
    } finally {
      const loadingDone = new Map(eeLoading.value)
      loadingDone.delete(spec.key)
      eeLoading.value = loadingDone
    }
  }

  const eeRefreshTimers = new Map()
  function debounceEeRefresh(spec, wait = 400) {
    clearTimeout(eeRefreshTimers.get(spec.key))
    eeRefreshTimers.set(spec.key, setTimeout(() => {
      eeRefreshTimers.delete(spec.key)
      refreshEeLayer(spec)
    }, wait))
  }

  function setEeParam(key, name, value) {
    const spec = eeTiles.catalogue.value.find((l) => l.key === key)
    if (!spec) return
    const p = spec.params?.[name]
    let next
    if (p?.type === 'enum') {
      next = (p.values || []).includes(String(value)) ? String(value) : (p.default ?? (p.values || [])[0])
    } else if (p?.type === 'codes') {
      try { next = normaliseCodes(value, p.max) } catch { return }
    } else if (p?.type === 'text') {
      const text = String(value).trim()
      if (!text) return
      next = text
    } else if (p?.type === 'date') {
      const date = String(value).trim()
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return
      next = date
    } else {
      next = Math.floor(Number(value))
      if (!Number.isFinite(next)) next = p?.default ?? 0
      if (p && Number.isFinite(p.min)) next = Math.max(p.min, next)
      if (p && Number.isFinite(p.max)) next = Math.min(p.max, next)
    }
    eeParams.value = { ...eeParams.value, [key]: { ...(eeParams.value[key] || {}), [name]: next } }
    debounceEeRefresh(spec)
  }

  const activeEeLayers = computed(() => {
    const cat = eeTiles.catalogue.value || []
    const out = []
    for (const key of activeOverlays.value) {
      const spec = cat.find((l) => l.key === key)
      if (spec) out.push({ key, params: paramsFor(spec) })
    }
    return out
  })

  function setBase(key) {
    const map = mapRef.value
    const next = baseLayers.value.find((b) => b.key === key)
    if (!next || !map) return
    for (const b of baseLayers.value) if (b.layer !== next.layer) safeRemoveLayer(b.layer)
    if (!map.hasLayer(next.layer)) next.layer.addTo(map)
    next.layer.bringToBack()
    activeBase.value = key
    try { localStorage.setItem(BASE_KEY, key) } catch { /* private mode */ }
  }

  function restoreBase() {
    let saved = null
    try { saved = localStorage.getItem(BASE_KEY) } catch { /* no storage */ }
    if (saved && saved !== activeBase.value && baseLayers.value.some((b) => b.key === saved)) {
      setBase(saved)
    }
  }

  return {
    overlayOrder, layerOpacity, layerBlend, soloKey,
    activeOverlays, overlayLayers, baseLayers, activeBase, activeBaseName,
    overlayGroups, eeParams, eeErrors, eeLoading, eeLayers, activeEeLayers,
    applyOverlayOrder, applyBlendModes, applySolo,
    setSolo, setLayerBlend, toggleOverlay, toggleOverlayByKey,
    moveOverlay, setLayerOpacity, clearOverlays,
    paramsFor, refreshEeLayer, debounceEeRefresh, setEeParam,
    setBase, restoreBase, restoreOverlays, restoreEeLayer,
  }
}
