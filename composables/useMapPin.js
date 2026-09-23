import { computed, ref, watch } from 'vue'
import { hasValue } from '~/composables/useObservations'

export function useMapPin({ mapRef, LRef, heatmapCellIndex, heatmapMode, heatmapCell, filteredData, activeEeLayers, accessToken, heatmaps }) {
  const pin = ref(null)
  const copied = ref(false)
  let pinMarker = null

  const pinElevation = ref(undefined)
  let elevTimer = null
  let elevSeq = 0

  const pinSamples = ref(null)
  const pinSampling = ref(false)
  const pinSampleError = ref('')

  function dropPinIcon() {
    const L = LRef.value
    return L.divIcon({
      className: 'drop-pin', iconSize: [28, 40], iconAnchor: [14, 38], tooltipAnchor: [0, -34],
      html: `<svg viewBox="0 0 24 34" width="28" height="40" aria-hidden="true">
        <path d="M12 0C5.4 0 0 5.3 0 11.9 0 20.6 12 34 12 34s12-13.4 12-22.1C24 5.3 18.6 0 12 0z"
              fill="#2d7ff9" stroke="#fff" stroke-width="1.5"/>
        <circle cx="12" cy="12" r="4.5" fill="#fff"/></svg>`,
    })
  }

  function setPin(lat, lon) {
    const map = mapRef.value
    const L = LRef.value
    pin.value = { lat, lon }
    copied.value = false
    if (!map || !L) return
    if (pinMarker) { pinMarker.setLatLng([lat, lon]); return }
    pinMarker = L.marker([lat, lon], {
      draggable: true,
      icon: dropPinIcon(),
      zIndexOffset: 1000,
      title: 'Dropped point — drag to move',
    }).addTo(map)
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
    } catch { /* clipboard refused */ }
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text)
      copied.value = true
      setTimeout(() => { copied.value = false }, 1600)
    } catch { /* clipboard refused */ }
  }

  const pinPlusCode = computed(() => (pin.value ? encodePlusCode(pin.value.lat, pin.value.lon, 11) : ''))

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

  const pinElevationText = computed(() => {
    const v = pinElevation.value
    if (v === undefined) return '…'
    if (v === null) return '—'
    return `${Math.round(v)} m · ${Math.round(v * 3.28084).toLocaleString()} ft`
  })

  watch(pin, () => { pinSamples.value = null; pinSampleError.value = '' }, { deep: true })

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

  function sampleText(s) {
    if (s.error) return 'unavailable'
    if (s.empty) return 'no data here'
    if (s.label) return s.label
    if (s.channels) return s.channels.map((c) => Math.round(c.value)).join(' / ')
    if (s.value === null || s.value === undefined) return '—'
    return `${typeof s.value === 'number' ? fmtNum(s.value) : s.value}${s.unit ? ` ${s.unit}` : ''}`
  }

  function heatmapCellAt(lat, lon) {
    if (!heatmapCell.value || !heatmapMode.value) return null
    return heatmapCellIndex.value.get(heatmaps.keyAt(lat, lon)) || null
  }

  const pinCell = computed(() => (pin.value ? heatmapCellAt(pin.value.lat, pin.value.lon) : null))

  const pinCellValue = computed(() => {
    const cell = pinCell.value
    if (!cell) return ''
    const v = cell.value
    if (v === null || v === undefined) return '—'
    return typeof v === 'number' ? fmtNum(v) : String(v)
  })

  const pinNearest = computed(() => {
    if (!pin.value) return null
    const feats = filteredData.value?.features || []
    if (!feats.length) return null
    const { lat, lon } = pin.value
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

  return {
    pin, copied,
    pinElevation, pinElevationText,
    pinSamples, pinSampling, pinSampleError,
    pinPlusCode, pinCell, pinCellValue, pinNearest,
    setPin, clearPin, copyPin, copyText,
    samplePinLayers, sampleText, heatmapCellAt,
  }
}

function fmtNum(v) {
  return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2)
}
