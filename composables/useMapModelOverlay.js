import { ref } from 'vue'
import { useModelOverlay } from '~/composables/useModelOverlay'
import { MAP_MAX_ZOOM } from '~/composables/useMapTileDate'

export function useMapModelOverlay({ mapRef, LRef, accessToken }) {
  const overlayHandoff = useModelOverlay()
  const modelOverlay = ref(null)
  let modelLayer = null

  function overlayAge(mintedAt) {
    if (!mintedAt) return ''
    const d = new Date(mintedAt)
    if (!Number.isFinite(d.getTime())) return ''
    const hours = Math.floor((Date.now() - d.getTime()) / 3600000)
    if (hours < 1) return 'just now'
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }

  function removeModelOverlay() {
    const map = mapRef.value
    if (modelLayer && map) map.removeLayer(modelLayer)
    modelLayer = null
    modelOverlay.value = null
  }

  function applyModelOverlay() {
    const map = mapRef.value
    const L = LRef.value
    const pending = overlayHandoff.pending.value
    if (!pending || !pending.template || !map || !L) return
    removeModelOverlay()

    const layer = L.tileLayer(pending.template, {
      opacity: 0.7,
      maxZoom: MAP_MAX_ZOOM,
      updateWhenIdle: false,
      updateWhenZooming: true,
      className: 'model-suitability',
    })

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

    const r = pending.region
    if (r && Number.isFinite(r.north)) {
      try {
        map.fitBounds(L.latLngBounds([r.south, r.west], [r.north, r.east]).pad(0.05), { animate: false })
      } catch { /* bad region */ }
    }

    overlayHandoff.clear()
  }

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

  async function loadModelById(jobId) {
    const map = mapRef.value
    const L = LRef.value
    if (!jobId || !map || !L) return
    removeModelOverlay()
    const token = await accessToken()
    let body
    try {
      const res = await fetch(`/.netlify/functions/model-tiles?job=${encodeURIComponent(jobId)}`, {
        headers: token ? { authorization: `Bearer ${token}` } : {},
      })
      body = await res.json()
      if (!res.ok || !body.ok || !body.template) throw new Error(body.error || 'Could not load model.')
    } catch (err) {
      return { error: err.message }
    }

    const layer = L.tileLayer(body.template, {
      opacity: 0.7,
      maxZoom: MAP_MAX_ZOOM,
      updateWhenIdle: false,
      updateWhenZooming: true,
      className: 'model-suitability',
    })
    let failed = 0
    layer.on('tileerror', () => {
      failed += 1
      if (failed >= 3 && modelOverlay.value && !modelOverlay.value.stale) {
        modelOverlay.value = { ...modelOverlay.value, stale: true }
      }
    })
    layer.addTo(map)
    modelLayer = layer

    const meta = body.meta || {}
    modelOverlay.value = {
      jobId,
      label: meta.label || 'Model',
      legend: meta.legend || { stops: ['#2c2f6b', '#c6301f'], min: '0', max: '1' },
      age: overlayAge(meta.mintedAt),
      cv: meta.cv || null,
      stale: false,
      refreshing: false,
    }

    const r = meta.region
    if (r && Number.isFinite(r.north)) {
      try {
        map.fitBounds(L.latLngBounds([r.south, r.west], [r.north, r.east]).pad(0.05), { animate: false })
      } catch { /* bad region */ }
    }
  }

  return { modelOverlay, applyModelOverlay, refreshModelOverlay, removeModelOverlay, loadModelById }
}
