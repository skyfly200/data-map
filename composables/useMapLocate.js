import { ref, watch } from 'vue'

export function useMapLocate({ mapRef, LRef }) {
  const locating = ref(false)
  const locateError = ref('')
  let locateBtn = null
  let userLayer = null

  watch(locating, (v) => { if (locateBtn) locateBtn.classList.toggle('busy', v) })
  watch(locateError, (msg) => { if (locateBtn && msg) locateBtn.title = msg })

  function setLocateBtn(btn) { locateBtn = btn }

  function locateMe() {
    const map = mapRef.value
    const L = LRef.value
    if (!map || !L) return
    if (!('geolocation' in navigator)) {
      locateError.value = 'Geolocation not supported by this browser.'
      return
    }
    locating.value = true
    locateError.value = ''
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        locating.value = false
        const { latitude: lat, longitude: lon, accuracy } = pos.coords
        if (userLayer) { userLayer.remove(); userLayer = null }
        userLayer = L.layerGroup([
          L.circle([lat, lon], { radius: accuracy || 0, color: '#2a78d6', weight: 1, fillOpacity: 0.12 }),
          L.circleMarker([lat, lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#2a78d6', fillOpacity: 1 })
            .bindTooltip('You are here', { direction: 'top' }),
        ]).addTo(map)
        map.setView([lat, lon], Math.max(map.getZoom() || 0, 11))
      },
      (err) => {
        locating.value = false
        locateError.value = err.code === err.PERMISSION_DENIED
          ? 'Location permission denied.'
          : 'Could not get your location.'
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    )
  }

  return { locating, locateError, setLocateBtn, locateMe }
}
