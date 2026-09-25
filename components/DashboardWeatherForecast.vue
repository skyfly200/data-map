<template>
  <div class="dash-widget weather-forecast">
    <div class="widget-head">
      <h3 class="widget-title">🌤️ Weather</h3>
      <div class="widget-actions">
        <button class="toggle-btn" :class="{ active: view === 'radar' }" @click="view = 'radar'">Radar</button>
        <button class="toggle-btn" :class="{ active: view === 'forecast' }" @click="view = 'forecast'">Forecast</button>
      </div>
    </div>

    <!-- Location status -->
    <div v-if="!locationGranted && !locationError" class="loc-prompt">
      <button class="loc-btn" @click="requestLocation">📍 Use my location</button>
    </div>
    <p v-if="locationError" class="widget-note error">{{ locationError }}</p>

    <!-- Radar view -->
    <div v-if="view === 'radar' && locationGranted" class="radar-wrap">
      <div class="radar-map" ref="radarEl">
        <!-- Animated radar overlay via RainViewer public tiles -->
        <iframe
          v-if="radarUrl"
          :src="radarUrl"
          class="radar-frame"
          title="Animated radar"
          loading="lazy"
          referrerpolicy="no-referrer"
          sandbox="allow-scripts allow-same-origin"
        ></iframe>
        <div v-else class="radar-placeholder">
          <div class="radar-rings">
            <div class="ring r1"></div>
            <div class="ring r2"></div>
            <div class="ring r3"></div>
          </div>
          <span class="radar-label">Loading radar…</span>
        </div>
      </div>
      <p class="radar-note">Animated precipitation radar · <a :href="radarAttrib" target="_blank" rel="noopener">RainViewer</a></p>
    </div>

    <!-- Forecast view -->
    <div v-if="view === 'forecast'" class="forecast-wrap">
      <p v-if="forecastLoading" class="widget-note">Loading forecast…</p>
      <p v-else-if="forecastError" class="widget-note error">{{ forecastError }}</p>
      <div v-else-if="!forecast.length && locationGranted" class="widget-note">No forecast data.</div>
      <div v-else-if="!locationGranted" class="widget-note muted">Enable location for forecast.</div>
      <div v-else class="forecast-days">
        <div v-for="day in forecast" :key="day.date" class="forecast-day">
          <span class="fc-dow">{{ day.dow }}</span>
          <span class="fc-icon" :title="day.description">{{ day.icon }}</span>
          <span class="fc-hi">{{ day.hi }}°</span>
          <span class="fc-lo">{{ day.lo }}°</span>
          <div class="fc-precip-bar" :title="`${day.precipMm}mm precip`">
            <div class="fc-precip-fill" :style="{ height: `${Math.min(100, day.precipMm * 10)}%` }"></div>
          </div>
          <span class="fc-precip-val">{{ day.precipMm > 0 ? day.precipMm.toFixed(1) : '' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'

const view = ref('forecast')
const lat = ref(null)
const lon = ref(null)
const locationGranted = ref(false)
const locationError = ref('')
const forecastLoading = ref(false)
const forecastError = ref('')
const forecast = ref([])

const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
const WMO_ICON = {
  0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
  45: '🌫️', 48: '🌫️',
  51: '🌦️', 53: '🌧️', 55: '🌧️',
  61: '🌧️', 63: '🌧️', 65: '🌧️',
  71: '🌨️', 73: '🌨️', 75: '❄️',
  80: '🌦️', 81: '🌧️', 82: '⛈️',
  95: '⛈️', 96: '⛈️', 99: '⛈️',
}
const WMO_DESC = {
  0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Foggy', 48: 'Icy fog',
  51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
  61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow',
  80: 'Showers', 81: 'Rain showers', 82: 'Violent showers',
  95: 'Thunderstorm', 96: 'Thunderstorm w/ hail', 99: 'Heavy thunderstorm',
}

function requestLocation() {
  if (!navigator?.geolocation) { locationError.value = 'Geolocation not supported.'; return }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      lat.value = pos.coords.latitude.toFixed(4)
      lon.value = pos.coords.longitude.toFixed(4)
      locationGranted.value = true
      locationError.value = ''
      fetchForecast()
    },
    () => { locationError.value = 'Location access denied.' }
  )
}

// Open-Meteo: free, no key required
async function fetchForecast() {
  if (!lat.value || !lon.value) return
  forecastLoading.value = true
  forecastError.value = ''
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat.value}&longitude=${lon.value}&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=7`
    const res = await fetch(url)
    if (!res.ok) throw new Error('Weather fetch failed')
    const data = await res.json()
    const daily = data.daily
    forecast.value = (daily.time || []).map((date, i) => ({
      date,
      dow: DAYS[new Date(date).getDay()],
      icon: WMO_ICON[daily.weathercode[i]] || '❓',
      description: WMO_DESC[daily.weathercode[i]] || '',
      hi: Math.round(daily.temperature_2m_max[i]),
      lo: Math.round(daily.temperature_2m_min[i]),
      precipMm: +(daily.precipitation_sum[i] || 0).toFixed(1),
    }))
  } catch (e) {
    forecastError.value = 'Could not load forecast.'
  } finally {
    forecastLoading.value = false
  }
}

// RainViewer embed URL — centers on user location
const radarUrl = computed(() => {
  if (!lat.value || !lon.value) return null
  const z = 7
  return `https://www.rainviewer.com/map.html?loc=${lat.value},${lon.value},${z}&oFa=0&oC=0&oU=0&oCS=1&oF=0&oAP=1&rmt=4&referendum=1&c=3&o=83&lm=1&layer=radar&sm=1&sn=1`
})

const radarAttrib = 'https://www.rainviewer.com'

onMounted(() => {
  // Try silently if permission was already granted
  if (navigator?.geolocation) {
    navigator.permissions?.query({ name: 'geolocation' }).then((p) => {
      if (p.state === 'granted') requestLocation()
    }).catch(() => {})
  }
})
</script>

<style scoped>
.weather-forecast { height: 100%; display: flex; flex-direction: column; gap: 0.6rem; }
.widget-head { display: flex; align-items: center; justify-content: space-between; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.3rem; }
.toggle-btn {
  font: inherit; font-size: 0.68rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #888);
  border-radius: 999px; padding: 0.12rem 0.48rem; cursor: pointer;
}
.toggle-btn.active { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }

.widget-note { font-size: 0.85rem; color: var(--muted, #888); text-align: center; padding: 0.5rem; }
.widget-note.error { color: #c0392b; }
.widget-note.muted { color: var(--muted, #aaa); }

.loc-prompt { display: flex; justify-content: center; padding: 0.5rem 0; }
.loc-btn {
  font: inherit; font-size: 0.82rem; border: 1px solid var(--border, #ddd);
  background: var(--surface-2, #f5f5f5); border-radius: 8px;
  padding: 0.4rem 0.9rem; cursor: pointer; color: var(--text, #222);
}
.loc-btn:hover { border-color: var(--accent, #2a78d6); }

/* Radar */
.radar-wrap { display: flex; flex-direction: column; gap: 0.3rem; flex: 1; min-height: 0; }
.radar-map { flex: 1; min-height: 160px; border-radius: 8px; overflow: hidden; position: relative; background: #1a2a3a; }
.radar-frame { width: 100%; height: 100%; border: none; display: block; }
.radar-placeholder { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.75rem; }
.radar-rings { position: relative; width: 80px; height: 80px; }
.ring {
  position: absolute; border-radius: 50%; border: 2px solid rgba(0, 180, 255, 0.4);
  animation: radar-pulse 2s ease-out infinite;
}
.r1 { inset: 30px; animation-delay: 0s; }
.r2 { inset: 15px; animation-delay: 0.6s; }
.r3 { inset: 0; animation-delay: 1.2s; }
@keyframes radar-pulse {
  0% { opacity: 0.8; transform: scale(0.8); }
  100% { opacity: 0; transform: scale(1.1); }
}
.radar-label { font-size: 0.75rem; color: rgba(180, 220, 255, 0.7); }
.radar-note { font-size: 0.65rem; color: var(--muted, #aaa); text-align: right; }
.radar-note a { color: var(--accent, #2a78d6); }

/* Forecast */
.forecast-wrap { flex: 1; display: flex; flex-direction: column; }
.forecast-days { display: flex; gap: 0.4rem; justify-content: space-between; flex: 1; align-items: flex-end; }
.forecast-day {
  display: flex; flex-direction: column; align-items: center; gap: 0.2rem;
  flex: 1; min-width: 0;
}
.fc-dow { font-size: 0.68rem; color: var(--muted, #888); text-transform: uppercase; letter-spacing: 0.04em; }
.fc-icon { font-size: 1.3rem; line-height: 1; }
.fc-hi { font-size: 0.82rem; font-weight: 700; color: var(--text, #222); }
.fc-lo { font-size: 0.72rem; color: var(--muted, #888); }
.fc-precip-bar {
  width: 100%; max-width: 28px; height: 32px; background: var(--surface-2, #eee);
  border-radius: 3px; overflow: hidden; display: flex; align-items: flex-end;
}
.fc-precip-fill { width: 100%; background: #4a90d9; border-radius: 3px 3px 0 0; min-height: 2px; }
.fc-precip-val { font-size: 0.62rem; color: #4a90d9; font-variant-numeric: tabular-nums; min-height: 0.8em; }
</style>
