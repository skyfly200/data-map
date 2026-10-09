// Tap-to-pick reference spots for an Earth Engine layer that compares the map
// against chosen places (habitat similarity, "compare to: spots").
//
// The spots live in the layer's own `points` parameter as "lat,lng;lat,lng",
// so they travel the same way every other layer setting does: into the tile
// request, the cache key and the share link. This composable only turns taps
// into that string and draws the pins.
//
// Picking is a mode, switched on from the layer's key card, rather than every
// map tap: a plain tap already opens an observation, and on a phone a layer
// that grabbed every tap would make the map unusable for anything else.

import { ref, watch } from 'vue'

export function spotList(value) {
  return String(value || '').split(';').filter(Boolean)
}

/** One tap, rounded the way the server normalises it so the cache key matches. */
export function spotFor(lat, lng) {
  return `${Number(lat).toFixed(5)},${Number(lng).toFixed(5)}`
}

export function useMapSpotPicker({ mapRef, LRef, eeParams, setEeParam, activeOverlays, maxFor }) {
  const pickingFor = ref(null)
  let pins = null

  const spotsFor = (key) => spotList(eeParams.value[key]?.points)

  function setPickingClass(on) {
    mapRef.value?.getContainer()?.classList.toggle('picking', on)
  }

  function startPicking(key) {
    pickingFor.value = key
    if (eeParams.value[key]?.compare !== 'spots') setEeParam(key, 'compare', 'spots')
    setPickingClass(true)
  }

  function stopPicking() {
    pickingFor.value = null
    setPickingClass(false)
  }

  function addSpot(key, lat, lng) {
    const list = spotsFor(key)
    if (list.length >= maxFor(key)) return false
    setEeParam(key, 'points', [...list, spotFor(lat, lng)].join(';'))
    return true
  }

  function removeSpot(key, index) {
    setEeParam(key, 'points', spotsFor(key).filter((_, i) => i !== index).join(';'))
  }

  function clearSpots(key) {
    setEeParam(key, 'points', '')
  }

  function drawPins() {
    const map = mapRef.value
    const L = LRef.value
    if (!map || !L) return
    if (!pins) pins = L.layerGroup().addTo(map)
    pins.clearLayers()
    for (const key of activeOverlays.value) {
      const params = eeParams.value[key]
      if (params?.compare !== 'spots') continue
      spotList(params.points).forEach((spot, i) => {
        const [lat, lng] = spot.split(',').map(Number)
        const pin = L.circleMarker([lat, lng], {
          radius: 8, weight: 3, color: '#e65100', fillColor: '#fff', fillOpacity: 1,
          bubblingMouseEvents: false,
        })
        pin.bindTooltip(`Spot ${i + 1}${pickingFor.value === key ? ' (tap to remove)' : ''}`)
        // Removing is only offered while picking, so a stray tap on a pin
        // while browsing does not quietly change the layer.
        pin.on('click', () => { if (pickingFor.value === key) removeSpot(key, i) })
        pins.addLayer(pin)
      })
    }
  }

  /** Wire the map tap. Call once the map exists. */
  function attach(map) {
    map.on('click', (e) => {
      const key = pickingFor.value
      if (!key) return
      if (!addSpot(key, e.latlng.lat, e.latlng.lng)) stopPicking()
    })
  }

  watch([eeParams, activeOverlays, pickingFor], drawPins, { deep: true })
  // A layer switched off while picking ends the mode, or taps would keep
  // editing a layer nobody can see.
  watch(activeOverlays, (active) => {
    if (pickingFor.value && !active.has(pickingFor.value)) stopPicking()
  })

  return { pickingFor, spotsFor, startPicking, stopPicking, clearSpots, removeSpot, attach }
}
