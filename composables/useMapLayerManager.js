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

  function applySolo() {
    const map = mapRef.value
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
      if (entry.layer && map.hasLayer(entry.layer)) map.removeLayer(entry.layer)
      if (soloKey.value === entry.key) soloKey.value = ''
    } else {
      next.add(entry.key)
      overlayOrder.value = [entry.key, ...overlayOrder.value]
      soloKey.value = ''
    }
    activeOverlays.value = next
    if (!wasOn && entry.key.startsWith('maxent:')) {
      heatmaps.mode.value = 'maxent'
      heatmaps.maxentModelId.value = entry.key.replace('maxent:', '')
    }
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
    entry.layer.setOpacity(entry.layer._baseOpacity * value * tileOpacity.value)
    heatmaps.persist()
  }

  function clearOverlays() {
    soloKey.value = ''
    for (const key of [...activeOverlays.value]) toggleOverlayByKey(key)
  }

  watch(stackBlend, () => applyBlendModes())

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
    for (const b of baseLayers.value) if (b.layer !== next.layer) map.removeLayer(b.layer)
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
    setBase, restoreBase,
  }
}
