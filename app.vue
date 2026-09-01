<template>
  <div class="app">
    <NuxtRouteAnnouncer />
    <header class="app-header">
      <NuxtLink to="/" class="brand">
        <h1>Nexstrata</h1>
      </NuxtLink>
      <div class="app-controls">
        <NuxtLink v-if="filterCount" to="/data" class="filter-flag" title="Active filters — manage on the Data tab">
          Filters: {{ filterCount }}
        </NuxtLink>
        <div class="units" role="group" aria-label="Elevation units">
          <button :class="{ active: unit === 'ft' }" @click="unit = 'ft'">ft</button>
          <button :class="{ active: unit === 'm' }" @click="unit = 'm'">m</button>
        </div>
        <div class="units" role="group" aria-label="Temperature units">
          <button :class="{ active: tempUnit === 'F' }" @click="tempUnit = 'F'">°F</button>
          <button :class="{ active: tempUnit === 'C' }" @click="tempUnit = 'C'">°C</button>
        </div>
        <button class="theme-btn" :title="`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`" @click="toggleTheme">
          {{ theme === 'dark' ? '☀' : '☾' }}
        </button>
        <nav class="app-nav">
          <NuxtLink to="/" class="nav-link">Home</NuxtLink>
          <NuxtLink to="/map" class="nav-link">Map</NuxtLink>
          <NuxtLink to="/charts" class="nav-link">Charts</NuxtLink>
          <NuxtLink to="/data" class="nav-link">Data</NuxtLink>
          <NuxtLink to="/coverage" class="nav-link">Coverage</NuxtLink>
        </nav>
        <ClientOnly>
          <div class="auth-box" v-if="configured">
            <template v-if="isAuthed">
              <span class="avatar" :title="user?.email || ''">{{ initial }}</span>
              <span class="who" :title="user?.email || ''">{{ shortEmail }}</span>
              <button class="auth-btn" @click="signOut">Sign out</button>
            </template>
            <NuxtLink v-else to="/login" class="auth-btn as-link">Sign in</NuxtLink>
          </div>
        </ClientOnly>
      </div>
    </header>
    <main class="app-main">
      <NuxtPage />
    </main>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

useHead({
  title: 'data-map · Mushroom Observations',
  meta: [{ name: 'description', content: 'Mushroom observations enriched with terrain and environmental exposure.' }],
})

// The dataset picker itself lives on the Data page; the app still loads the
// manifest here so every view knows what's available from first paint.
const { selectedDataset, setDataset, loadDatasets } = useObservations()
onMounted(loadDatasets)

const { activeCount: filterCount } = useFilters()

const { user, isAuthed, configured, signOut } = useAuth()
const shortEmail = computed(() => {
  const e = user.value?.email
  return e ? e.split('@')[0] : 'Account'
})
const initial = computed(() => {
  const e = user.value?.email
  return (e ? e[0] : '?').toUpperCase()
})

// Theme: default dark, remembered per viewer. `:root` CSS defaults to dark so
// there's no light flash before hydration; we only stamp an attribute for light.
const theme = useState('theme', () => 'dark')
function applyTheme(t) {
  if (import.meta.client) document.documentElement.setAttribute('data-theme', t)
}
function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
  applyTheme(theme.value)
  if (import.meta.client) localStorage.setItem('theme', theme.value)
}

// Units: default feet + Fahrenheit, remembered per viewer.
const { unit, tempUnit } = useUnits()
onMounted(() => {
  const savedTheme = localStorage.getItem('theme')
  theme.value = savedTheme === 'light' ? 'light' : 'dark'
  applyTheme(theme.value)
  const e = localStorage.getItem('elev-unit')
  if (e === 'm' || e === 'ft') unit.value = e
  const t = localStorage.getItem('temp-unit')
  if (t === 'F' || t === 'C') tempUnit.value = t
  // Restore the dataset choice after a hard reload (SSR can't read localStorage).
  const ds = localStorage.getItem('observations-dataset')
  if (ds && ds !== selectedDataset.value) setDataset(ds)
})
watch(unit, (v) => {
  if (import.meta.client) localStorage.setItem('elev-unit', v)
})
watch(tempUnit, (v) => {
  if (import.meta.client) localStorage.setItem('temp-unit', v)
})
</script>

<style>
/* ── Design tokens. Dark is the default (bare :root) so first paint is dark with
   no flash; [data-theme="light"] opts back into the original light palette. ── */
:root {
  color-scheme: dark;
  --bg: #0e1217;
  --surface: #171e27;
  --surface-2: #1e2732;
  --surface-3: #253040;
  --text: #e6e9ee;
  --text-strong: #f4f6f8;
  --muted: #9aa4b2;
  --border: #2a3441;
  --border-soft: #222b36;
  --accent: #34c46a;
  --accent-ink: #0e1217;
  --header-bg: #12181f;
  --input-bg: #0f151c;
  --grid: #26303c;
  --danger: #ff6b6b;
  --shadow: rgba(0, 0, 0, 0.5);
  --glow: 0.9; /* chart glow strength (0 = off) */
  --tooltip-bg: #0b0f14;
  --tooltip-fg: #f4f6f8;
  /* Soft edge-glow for chart marks (uses each mark's own colour via currentColor). */
  --chart-glow: drop-shadow(0 0 2px currentColor);
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #ffffff;
  --surface: #ffffff;
  --surface-2: #f7f8fa;
  --surface-3: #f3f4f6;
  --text: #1f2933;
  --text-strong: #10151b;
  --muted: #6b7280;
  --border: #e5e7eb;
  --border-soft: #eef0f2;
  --accent: #2b7a3d;
  --accent-ink: #ffffff;
  --header-bg: #1f2933;
  --input-bg: #ffffff;
  --grid: #eef0f2;
  --danger: #b00020;
  --shadow: rgba(16, 24, 40, 0.12);
  --glow: 0;
  --tooltip-bg: #1f2933;
  --tooltip-fg: #ffffff;
  --chart-glow: none;
}

html, body, #__nuxt { height: 100%; margin: 0; }
body { font-family: system-ui, -apple-system, sans-serif; color: var(--text); background: var(--bg); }

/* Global form-control theming so text is readable in both themes. Components
   can still override, but this ensures no light-on-white inputs in dark mode. */
input, select, textarea { background: var(--input-bg); color: var(--text); }
input::placeholder, textarea::placeholder { color: var(--muted); opacity: 1; }

.app { display: flex; flex-direction: column; height: 100vh; background: var(--bg); }

.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 10px 20px; background: var(--header-bg); color: #fff; flex: 0 0 auto;
  border-bottom: 1px solid var(--border);
}
.brand { text-decoration: none; color: inherit; }
.brand h1 { margin: 0; font-size: 1.15rem; }
.brand:hover h1 { opacity: 0.85; }

.app-controls { display: flex; align-items: center; gap: 14px; }

.theme-btn {
  border: 1px solid #52606d; background: transparent; color: #cbd2d9; cursor: pointer;
  border-radius: 6px; width: 30px; height: 28px; font-size: 0.95rem; line-height: 1;
}
.theme-btn:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }

.units { display: inline-flex; border: 1px solid #52606d; border-radius: 6px; overflow: hidden; }
.units button {
  border: 0; background: transparent; color: #cbd2d9; cursor: pointer;
  padding: 5px 11px; font-size: 0.85rem; font-weight: 600;
}
.units button:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }
.units button.active { background: #3e4c59; color: #fff; }

.app-nav { display: flex; gap: 6px; }
.nav-link {
  color: #cbd2d9; text-decoration: none; font-size: 0.9rem; font-weight: 500;
  padding: 6px 12px; border-radius: 6px;
}
.nav-link:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
.nav-link.router-link-exact-active { background: #3e4c59; color: #fff; }

.filter-flag { background: #2b7a3d; color: #fff; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; text-decoration: none; white-space: nowrap; }
.filter-flag:hover { background: #256a34; }

.auth-box { display: inline-flex; align-items: center; gap: 8px; }
.auth-box .avatar {
  display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px; border-radius: 50%; background: #3e4c59; color: #fff;
  font-size: 0.72rem; font-weight: 700; flex: 0 0 auto;
}
.auth-box .who { font-size: 0.8rem; color: #cbd2d9; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.auth-btn { border: 1px solid #52606d; background: transparent; color: #cbd2d9; border-radius: 6px; padding: 5px 11px; font-size: 0.82rem; font-weight: 600; cursor: pointer; text-decoration: none; }
.auth-btn:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }
.auth-btn.as-link { display: inline-block; }

.app-main { flex: 1 1 auto; min-height: 0; overflow: auto; }

/* ── Mobile: the header is a single row on desktop; let it wrap and give the
   nav its own horizontally-scrollable strip so nothing overflows off-screen. ── */
@media (max-width: 860px) {
  .app-header { flex-wrap: wrap; gap: 8px 10px; padding: 8px 12px; }
  .brand h1 { font-size: 1rem; }
  .app-controls { flex-wrap: wrap; gap: 8px 10px; width: 100%; }
  .app-nav {
    order: 5; width: 100%; flex-wrap: nowrap; overflow-x: auto;
    -webkit-overflow-scrolling: touch; gap: 4px; padding-bottom: 2px;
  }
  .app-nav::-webkit-scrollbar { height: 0; }
  .nav-link { white-space: nowrap; padding: 6px 10px; }
  .auth-box { margin-left: auto; }
}
@media (max-width: 480px) {
  .brand h1 { font-size: 0.95rem; }
  .units button { padding: 4px 8px; }
}
</style>
