<template>
  <div class="app">
    <NuxtRouteAnnouncer />
    <!-- Embedded in someone else's page: drop the site chrome so the iframe
         shows only the view that was shared. -->
    <header v-if="!isEmbed" class="app-header">
      <NuxtLink to="/" class="brand">
        <AppLogo :size="30" />
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
          <NuxtLink to="/analysis" class="nav-link">Analysis</NuxtLink>
          <NuxtLink to="/data" class="nav-link">Data</NuxtLink>
          <NuxtLink to="/coverage" class="nav-link">Coverage</NuxtLink>
          <NuxtLink to="/guide" class="nav-link">Guide</NuxtLink>
        </nav>
        <ClientOnly>
          <div class="auth-box" v-if="configured">
            <!-- Sync state and sign-out used to sit in the header itself, which
                 spent a third of a phone's width on two things you touch once a
                 session. They live under the name now, with the sync dot still
                 on the button so a failure is visible without opening it. -->
            <AccountMenu v-if="isAuthed" :user="user" :initial="initial"
                         :short-email="shortEmail" @sign-out="signOut" />
            <NuxtLink v-else to="/login" class="auth-btn as-link">Sign in</NuxtLink>
          </div>
        </ClientOnly>
      </div>
    </header>
    <main class="app-main">
      <NuxtPage />
    </main>
    <ClientOnly><ShortcutsHelp /></ClientOnly>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'
import { useShareState } from '~/composables/useShareState'

const { isEmbed } = useShareState()

const DESCRIPTION = 'Mushroom observations enriched with terrain and environmental exposure.'
useHead({
  title: 'Nexstrata · Mushroom Observations',
  meta: [
    { name: 'description', content: DESCRIPTION },
    { name: 'theme-color', content: '#12181f' },
    { property: 'og:title', content: 'Nexstrata' },
    { property: 'og:description', content: DESCRIPTION },
    { property: 'og:image', content: '/logo.svg' },
    { name: 'twitter:card', content: 'summary' },
  ],
  link: [
    // The logo itself, vector first: browsers that understand SVG favicons get
    // it at every size. The .ico is the same logo rendered to fixed sizes for
    // the clients that still want one — regenerate it with
    //   magick -background none public/logo.svg \
    //     -define icon:auto-resize=256,128,64,48,32,24,16 public/favicon.ico
    { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' },
    { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
    { rel: 'apple-touch-icon', href: '/logo.svg' },
    // Installable, and the same manifest tells the browser what to open when it
    // is launched from a home screen with no connection.
    { rel: 'manifest', href: '/manifest.webmanifest' },
  ],
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

// ─── Keyboard shortcuts ─────────────────────────────────────────────────────
// Registered here rather than per-page so they work everywhere, and so the help
// overlay lists them even before a page adds its own.
const shortcuts = useShortcuts()
const router = useRouter()
const go = (path) => () => router.push(path)

shortcuts.register([
  { scope: 'Navigate', keys: 'm', label: 'Map', run: go('/map') },
  { scope: 'Navigate', keys: 'c', label: 'Charts', run: go('/charts') },
  { scope: 'Navigate', keys: 'a', label: 'Analysis', run: go('/analysis') },
  { scope: 'Navigate', keys: 'd', label: 'Data', run: go('/data') },
  { scope: 'Navigate', keys: 'v', label: 'Coverage', run: go('/coverage') },
  { scope: 'Navigate', keys: 'g', label: 'Guide', run: go('/guide') },
  { scope: 'General', keys: '?', label: 'Show this help', run: () => { shortcuts.helpOpen.value = !shortcuts.helpOpen.value } },
  { scope: 'General', keys: 'escape', label: 'Close dialogs and panels', run: () => { shortcuts.helpOpen.value = false } },
  { scope: 'General', keys: 't', label: 'Light / dark theme', run: () => toggleTheme() },
  { scope: 'General', keys: 'u', label: 'Metric / imperial units', run: () => { unit.value = unit.value === 'ft' ? 'm' : 'ft' } },
])

onMounted(() => window.addEventListener('keydown', shortcuts.handle))
onBeforeUnmount(() => window.removeEventListener('keydown', shortcuts.handle))

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

/* 100dvh, not 100vh. On a phone `vh` is measured against the viewport with the
   browser's chrome retracted — its largest state — so a 100vh shell is taller
   than what you can actually see, the page picks up a scrollbar it should not
   have, and nudging it hides the top row of the header under the URL bar. `dvh`
   tracks the chrome as it comes and goes. The vh line stays as a fallback for
   browsers without dvh. */
.app { display: flex; flex-direction: column; height: 100vh; height: 100dvh; background: var(--bg); }

.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 10px 20px; background: var(--header-bg); color: #fff; flex: 0 0 auto;
  border-bottom: 1px solid var(--border);
  /* Above anything a page draws. The map alone puts Leaflet's controls at 1000
     and its observation drawer at 1100, and without a stacking order of its own
     the header's account menu opened underneath them. */
  position: relative; z-index: 2000;
}
.brand { display: inline-flex; align-items: center; gap: 9px; text-decoration: none; color: inherit; }
.brand h1 { margin: 0; font-size: 1.15rem; }
.brand:hover { opacity: 0.85; }

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
  .app-header { flex-wrap: wrap; gap: 6px 10px; padding: 6px 12px; }
  .brand { gap: 7px; }
  .brand h1 { font-size: 1rem; }
  .brand .logo { width: 24px; height: auto; }
  /* The nav lives inside .app-controls, which sits beside the brand — so the
     strip was penned into that column, starting a third of the way across and
     running out of room before the last two links. `display: contents` drops the
     wrapper out of the layout and lifts its children into the header's own flex
     line, so the units and theme sit next to the brand and the nav can claim the
     full width underneath. The wrapper still exists for desktop, where it is a
     real flex box again. */
  .app-controls { display: contents; }
  .brand { margin-right: auto; }
  .filter-flag, .units, .theme-btn, .auth-box { order: 2; }
  .app-nav {
    order: 5; flex: 1 0 100%; min-width: 0; flex-wrap: nowrap; overflow-x: auto;
    -webkit-overflow-scrolling: touch; gap: 4px; padding-bottom: 2px;
  }
  .app-nav::-webkit-scrollbar { height: 0; }
  .nav-link { white-space: nowrap; padding: 6px 10px; }
}
@media (max-width: 480px) {
  .brand h1 { font-size: 0.95rem; }
  /* The email is the first thing worth losing when space is short — the avatar
     already says who is signed in. */
  .auth-box .who { display: none; }
}

/* ── Touch: a finger is not a cursor ─────────────────────────────────────────
   Keyed on the pointer, not the viewport, so a tablet gets the same treatment
   as a phone and a small desktop window does not.

   Two approaches. Controls with room to grow get a real minimum height. Icon
   controls — the ? beside an option, a card's ⤓ and ⤢ — keep their drawn size
   and grow an invisible hit area instead, because scaling them up would turn a
   quiet affordance into a row of loud buttons. */
@media (pointer: coarse) {
  /* Every button and button-like link, unless it is one of the icon controls
     handled by hit-area expansion below. */
  button, [role="button"], .auth-btn, .nav-link, .filter-flag, .brand,
  .ref-nav a, .opt-also a, .hb-chip {
    min-height: 40px;
  }
  button, [role="button"], .auth-btn, .nav-link, .filter-flag, .brand, .ref-nav a {
    display: inline-flex; align-items: center;
  }
  /* The unit and theme toggles sit in a row of their own, so they can take the
     width as well as the height without crowding anything. */
  .app-header .units button { min-width: 46px; padding: 0 14px; justify-content: center; }
  .app-header .theme-btn { width: 42px; justify-content: center; }

  /* The icon controls opt back out: a 36px minimum would turn a card's quiet
     ⤓ and ⤢ into a pair of chunky buttons. They get reach, not bulk. */
  a.help, .card-tools .tool, .saved-tools button, .saved-tools .sh-btn,
  .sc-close, .close, .set-close {
    min-height: 0;
  }

  /* Checkboxes ship at 13px — under a sixth of a fingertip's area. 22px is
     still small to look at and large enough to hit, and the label beside one is
     a target too wherever it is wrapped in a <label>. */
  input[type="checkbox"], input[type="radio"] { width: 22px; height: 22px; }

  /* Form controls tall enough to hit, and 16px text so iOS does not zoom the
     page when one takes focus. */
  select, input[type="text"], input[type="number"], input[type="search"],
  input[type="date"], input[type="email"], input[type="password"] {
    min-height: 40px; font-size: 16px;
  }

  /* A range input's box is only as tall as its thumb, so the band you can start
     a drag in is a few pixels. Height here is hit area, not appearance: the
     browser keeps its native thumb, which restyling would throw away along with
     the track. touch-action stops the drag from scrolling the page instead. */
  input[type="range"] { height: 32px; touch-action: none; }

  /* Nothing that exists to be tapped should select its own text when held, or
     flash a grey box that outlives the tap. */
  button, [role="button"], .nav-link, .legend-row, .tabs button, summary {
    -webkit-tap-highlight-color: transparent;
    -webkit-user-select: none; user-select: none;
  }

  /* Sideways-scrolling regions — the table, wide charts — should take a
     horizontal drag without the page fighting them for it. */
  .table-wrap, .stage, [class*="scroll"] { -webkit-overflow-scrolling: touch; }

  /* Invisible hit area, centred on the icon, without disturbing the layout. */
  a.help, .card-tools .tool, .saved-tools button, .saved-tools .sh-btn {
    position: relative;
  }
  a.help::after, .card-tools .tool::after,
  .saved-tools button::after, .saved-tools .sh-btn::after {
    content: ''; position: absolute; left: 50%; top: 50%;
    width: 40px; height: 40px; transform: translate(-50%, -50%);
  }
  /* The # beside a reference heading is a desktop affordance — it appears on
     hover, which a touch screen has none of. */
  .opt .anchor { display: none; }
}
</style>
