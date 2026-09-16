<template>
  <div ref="root" class="home" :class="{ still: reduceMotion }">
    <!-- Ambient HUD field: a grid, a scanline and two drifting glows that track
         scroll and pointer through CSS custom properties the script writes. All
         decorative, all behind the content, all off when motion is reduced. -->
    <div class="hud-field" aria-hidden="true">
      <div class="hud-grid"></div>
      <div class="hud-glow glow-a"></div>
      <div class="hud-glow glow-b"></div>
      <div class="hud-scan"></div>
    </div>

    <!-- ── Hero ─────────────────────────────────────────────────────────────
         The vision first: Nexstrata is an ecosystem-modeling platform, and the
         enrichment everyone knows it for is one stage of that. -->
    <section class="hero">
      <div class="parallax layer-back">
        <AppLogo class="hero-logo" :size="132" />
      </div>
      <p class="eyebrow"><span class="tick" aria-hidden="true"></span>Ecosystem modeling platform · Beta</p>
      <h1>Model where life occurs.</h1>
      <h3>
        Nexstrata turns scattered field observations into predictive maps of habitat. It
        binds every environmental layer to each point, then models the whole surface.
        Sample enrichment is one step of that pipeline, not the destination.
      </h3>

      <div class="cta">
        <NuxtLink to="/map" class="btn primary">Open the map</NuxtLink>
        <NuxtLink to="/jobs" class="btn">Run an enrichment job</NuxtLink>
        <NuxtLink to="/guide" class="btn">Read the guide</NuxtLink>
      </div>

      <p v-if="totalCount" class="stat">
        <strong>{{ totalCount.toLocaleString() }}</strong> observations ·
        <strong>{{ ENRICHMENT_COUNT }}</strong> environmental fields on each ·
        <strong>{{ eeLayerCount }}</strong> Earth Engine layers
      </p>
    </section>

    <!-- ── The pipeline ─────────────────────────────────────────────────────
         Four stages, so enrichment reads as the second of four rather than the
         whole of it. This is the spine of the whole page. -->
    <section class="pipeline">
      <div class="sec-tag">Pipeline</div>
      <ol class="flow">
        <li v-for="(s, i) in PIPELINE" :key="s.title" class="flow-step" :style="{ '--n': i }">
          <span class="flow-k">{{ s.k }}</span>
          <span class="flow-title">{{ s.title }}</span>
          <span class="flow-body">{{ s.body }}</span>
          <span v-if="i < PIPELINE.length - 1" class="flow-arrow" aria-hidden="true">→</span>
        </li>
      </ol>
    </section>

    <!-- ── The stack ────────────────────────────────────────────────────────
         The sensor above, the layers between, the ground below. A 3D stack that
         tilts with scroll and pointer; flattened and un-tilted when motion is
         reduced or the screen is narrow. -->
    <section class="stack-sec">
      <div class="stack-grid">
        <div class="stack-text">
          <div class="sec-tag">The stack</div>
          <h2>From orbit to a single point</h2>
          <p class="etym-callout">
            <span class="etym-word"><strong class="etym">nex</strong> + <strong class="etym">strata</strong></span>
            <span class="etym-def">the binding of the layers at a point.</span>
          </p>
          <p>
            Every record is one coordinate and one date. Each layer is sampled right there,
            right then, not from a regional mean or today's ground, so a 2019 find carries
            the conditions that preceded it.
          </p>
          <p class="more-links">
            <NuxtLink to="/guide#where-the-data-comes-from">Source table, by column</NuxtLink>
            <NuxtLink to="/coverage">Cache coverage and dates</NuxtLink>
          </p>
        </div>

        <figure class="scene" aria-label="A satellite sampling environmental layers down to one point on the ground">
          <div class="scene-3d">
            <!-- Satellite in orbit at the top of the column. The dashed ellipse is
                 the orbital path; the satellite rides it, its position along the
                 path driven by how far the page is scrolled (--sat). -->
            <div class="orbit" aria-hidden="true">
              <svg class="orbit-svg" viewBox="0 0 220 92" preserveAspectRatio="none">
                <ellipse cx="110" cy="46" rx="104" ry="34" />
              </svg>
              <div class="sat">
                <span class="sat-body"></span>
                <span class="sat-panel left"></span>
                <span class="sat-panel right"></span>
              </div>
            </div>

            <!-- The beam the sensor casts straight down through the layers. -->
            <div class="beam" aria-hidden="true"></div>

            <!-- The named layers, each with a swatch of what it looks like on the
                 map, stacked in 3D. -->
            <div class="plates">
              <div v-for="(layer, i) in STRATA" :key="layer.name" class="plate"
                   :style="{ '--i': i, '--viz': layer.viz }">
                <span class="plate-viz" aria-hidden="true"></span>
                <span class="plate-text">
                  <span class="s-name">{{ layer.name }}</span>
                  <span class="s-fields">{{ layer.fields }}</span>
                </span>
              </div>
            </div>

            <!-- The ground, curved like a horizon, with the observation landing. -->
            <div class="earth" aria-hidden="true">
              <div class="earth-glow"></div>
              <div class="pin"></div>
            </div>
          </div>
          <figcaption>One observation, at the bottom of the stack.</figcaption>
        </figure>
      </div>
    </section>

    <!-- ── Modeling (the point of the enrichment) ───────────────────────────── -->
    <section class="model">
      <div class="sec-tag">Modeling <em class="road">Roadmap</em></div>
      <h2>From points to surfaces</h2>
      <p class="model-lede">
        Enrichment exists to feed a model. The stack turns each observation into a row of
        predictors; the model turns those rows into a continuous surface: suitability
        predicted everywhere, not only where someone looked.
      </p>
      <div class="model-grid">
        <div class="model-card">
          <h3>Maximum entropy (MaxEnt)</h3>
          <p>
            The data is presence-only: we know where a species was found, rarely where it
            was absent. MaxEnt fits the least-committal distribution consistent with
            conditions at the presence points, scored against background. The output is a
            suitability surface from 0 to 1.
          </p>
        </div>
        <div class="model-card">
          <h3>Features from the stack</h3>
          <p>
            Terrain, canopy, soil, weather and exposure become the predictors, each sampled
            at the record's own date. The layers you draw on the map are the features the
            model reads, so a suitability map speaks the same language as the rasters
            beneath it.
          </p>
        </div>
        <div class="model-card">
          <h3>Honest about bias</h3>
          <p>
            Presence-only modeling inherits the bias in where people look. Nexstrata treats
            that as a first-class input: background weighted by observer effort, every
            surface labelled with its confounds.
          </p>
        </div>
      </div>
      <div class="cta left">
        <NuxtLink to="/analysis" class="btn primary">See the analysis tools</NuxtLink>
        <a class="btn" :href="roadmapUrl" target="_blank" rel="noopener noreferrer">Modeling on the roadmap</a>
      </div>
    </section>

    <!-- ── Capabilities ─────────────────────────────────────────────────────
         One line each. Anything needing more than a line belongs behind the
         link rather than in front of it. -->
    <section class="features">
      <div class="sec-tag">Capabilities</div>
      <div class="feature-grid">
        <NuxtLink v-for="f in FEATURES" :key="f.title" :to="f.to" class="feature">
          <h3>{{ f.title }}</h3>
          <p>{{ f.body }}</p>
          <span class="feature-go" aria-hidden="true">→</span>
        </NuxtLink>
      </div>
    </section>

    <!-- ── Earth Engine ─────────────────────────────────────────────────── -->
    <section class="ee">
      <div class="sec-tag">Earth Engine</div>
      <h2>Compute, not storage</h2>
      <p class="ee-lede">
        No raster tiles are stored. A server function authenticates with a service account,
        calls <code>getMapId()</code> on the target asset and returns a short-lived tile
        template to the client; templates are cached per parameter set and re-minted before
        they expire.
      </p>
      <ul class="ee-list">
        <li>
          <strong>Enrich</strong>: submit a bounding box and date range. Staged sampling of
          terrain, land cover, soil, precipitation, temperature and vegetation at each point;
          output is written to object storage as GeoJSON.
        </li>
        <li>
          <strong>Render</strong>: {{ eeLayerCount }} built-in layers computed server-side:
          MODIS and MTBS fire history, GAP forest type, SOLUS100 soil, SRTM terrain analysis,
          and Sentinel-2 indices calculated per request.
        </li>
        <li>
          <strong>Publish</strong>: register an exported asset by ID, band and palette.
          Inputs are validated against a strict grammar before they reach Earth Engine, and
          each layer carries its own access tier.
        </li>
        <li>
          <strong>Analyse</strong>: load a job result as a dataset. It feeds the same map,
          charts, heatmaps and statistics as the reference data.
        </li>
      </ul>
      <div class="cta left">
        <NuxtLink to="/jobs" class="btn primary">Run a job</NuxtLink>
        <NuxtLink to="/guide#your-own-earth-engine-layers" class="btn">Publish a layer</NuxtLink>
        <a class="btn" :href="KAGGLE_URL" target="_blank" rel="noopener noreferrer">Run the pipeline yourself</a>
      </div>
    </section>

    <!-- ── Limitations ──────────────────────────────────────────────────────
         On the page rather than in the guide: it governs how every number above
         should be read, and a caveat nobody reaches is not a caveat. -->
    <section class="caveat">
      <div class="sec-tag">Known limitations</div>
      <p>
        The reference dataset is <strong>presence-only, opportunistically collected</strong>.
        Record density tracks observer effort and site access as much as it tracks
        occurrence, and absence of records is not evidence of absence. Enrichment adds
        environmental context to each record; it does not correct the sampling bias in which
        records exist, and neither does a model trained on them without care.
      </p>
      <p>
        Each view therefore states its own confounds at the point of use rather than in
        documentation. The seasonal heatmaps normalise within each cell, which cancels most of
        the effort bias; the density heatmaps cannot, and say so.
        <NuxtLink to="/guide#reference">Every control has a reference entry</NuxtLink>
        covering its behaviour and its failure modes.
      </p>
    </section>

    <section class="foot">
      <ClientOnly>
        <div v-if="configured" class="auth-cta">
          <template v-if="isAuthed">
            <span class="signed">Signed in as <strong>{{ user?.email || 'your account' }}</strong>.</span>
            <button class="btn small ghost" @click="signOut">Sign out</button>
          </template>
          <template v-else>
            <span class="hint">Read access is unauthenticated. Sign in to persist saved views and submit jobs:</span>
            <NuxtLink to="/login" class="btn small">Sign in</NuxtLink>
            <NuxtLink to="/login?mode=signup" class="btn small ghost">Sign up</NuxtLink>
          </template>
        </div>
      </ClientOnly>

      <p class="repo-line">
        <a class="repo" :href="repoUrl" target="_blank" rel="noopener noreferrer">
          <IconGithub class="repo-ico" /> <span>View the source on GitHub</span>
        </a>
      </p>
    </section>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { EE_LAYER_KEYS } from '~/netlify/lib/ee-tile-layers.mjs'

const { user, isAuthed, configured, signOut } = useAuth()
const repoUrl = 'https://github.com/skyfly200/data-map'
const roadmapUrl = 'https://github.com/skyfly200/data-map/blob/master/docs/roadmap.md'
const KAGGLE_URL = 'https://www.kaggle.com/code/skylerflywilson/nexstrata-data-enrichment-pipeline'

// Every headline number is counted from the thing it describes rather than
// written into the copy, so none of them can go stale behind the app's back.
const { availableDatasets } = useObservations()
const totalCount = computed(() =>
  availableDatasets.value?.find((d) => d.id === 'all')?.count || 0)

// The temporal fields are excluded: year, month and day-of-year come off the
// record's own date and are not something the pipeline went and sampled.
const TEMPORAL = new Set(['year', 'month', 'day_of_year'])
const ENRICHMENT_COUNT = ALL_NUMERIC.filter((f) => !TEMPORAL.has(f.key)).length

// The built-in catalogue only. Registered layers are per-deployment and behind a
// tier, so counting them here would promise a visitor something they may not see.
const eeLayerCount = EE_LAYER_KEYS.length

// The four stages of the platform, so enrichment reads as one of them. Each is a
// real place in the app except the model stage, which the section below marks as
// roadmap rather than claiming it ships today.
const PIPELINE = [
  { k: '01', title: 'Observe', body: 'Presence points from iNaturalist and your own uploads.' },
  { k: '02', title: 'Enrich', body: 'Bind every environmental layer to each point, at its own date.' },
  { k: '03', title: 'Model', body: 'Fit a presence-only distribution model over the enriched points.' },
  { k: '04', title: 'Predict', body: 'Project habitat suitability across the whole landscape.' },
]

// The layers named, coarse to fine, each with a CSS gradient that echoes how the
// layer actually paints on the map, so the stack shows the data, not just its
// name. `viz` is dropped straight into a linear-gradient in the style below.
const STRATA = [
  { name: 'Weather', fields: 'the seven days before the find', viz: '#2c7bb6, #abd9e9, #ffffbf, #fdae61, #d7191c' },
  { name: 'Canopy', fields: 'NDVI, NDMI', viz: '#a6611a, #dfc27d, #f5f5f5, #80cdc1, #018571' },
  { name: 'Ground', fields: 'soil moisture, texture, land cover', viz: '#8c510a, #d8b365, #f6e8c3, #5ab4ac, #01665e' },
  { name: 'Terrain', fields: 'elevation, slope, aspect, wetness', viz: '#1a9850, #d9ef8b, #fee08b, #fc8d59, #d73027' },
  { name: 'Exposure', fields: 'sun, wind', viz: '#2166ac, #d1e5f0, #fddbc7, #ef8a62, #b2182b' },
]

const FEATURES = [
  {
    to: '/map',
    title: 'Map and aggregate',
    body: 'Point rendering with hex or square binning from ~100 m to ~28 km, over reference '
      + 'overlays for fire, forest type, soil, weather and land ownership.',
  },
  {
    to: '/jobs',
    title: 'Point enrichment',
    body: 'Queue an Earth Engine job over a bounding box and date range. Progress reports per '
      + 'stage; results are stored and loadable as a dataset.',
  },
  {
    to: '/guide#your-own-earth-engine-layers',
    title: 'Custom raster layers',
    body: 'Register an exported Earth Engine asset by ID, band and palette. Tile URLs are minted '
      + 'server-side and gated per access tier.',
  },
  {
    to: '/analysis',
    title: 'Statistics',
    body: 'Spearman rank correlations across populated fields, per-species deviation from the '
      + 'dataset mean, with season and effort confounds reported alongside.',
  },
  {
    to: '/charts?tab=build',
    title: 'Charts',
    body: 'Ten chart types over any field pair, saved per account, with shareable URLs that '
      + 'restore the active filters, colouring and map view.',
  },
  {
    to: '/offline',
    title: 'Maps while offline',
    body: 'Named tile areas cached by a service worker, covering every layer drawn when the area '
      + 'was saved. Deletion is reference-counted, so overlapping areas keep their tiles.',
  },
]

// Parallax and the 3D tilt, written as CSS custom properties on the root rather
// than through Vue reactivity: a scroll handler that re-rendered the component on
// every frame would cost far more than setting three numbers on one element. The
// map scrolls inside .app-main, not the window, so the scroll is read there.
const root = ref(null)
const reduceMotion = ref(false)
let scroller = null
let frame = 0
let mx = 0
let my = 0
let sc = 0

function apply() {
  frame = 0
  const el = root.value
  if (!el) return
  el.style.setProperty('--mx', mx.toFixed(3))
  el.style.setProperty('--my', my.toFixed(3))
  el.style.setProperty('--sc', String(Math.round(sc)))
  // 0 at the top of the page, 1 at the bottom: the satellite rides its orbit
  // once as you scroll the whole page.
  const max = scroller ? scroller.scrollHeight - scroller.clientHeight : 0
  const progress = max > 0 ? Math.min(1, Math.max(0, sc / max)) : 0
  el.style.setProperty('--sat', progress.toFixed(4))
}
function schedule() {
  if (!frame) frame = requestAnimationFrame(apply)
}
function onPointer(e) {
  mx = e.clientX / window.innerWidth - 0.5
  my = e.clientY / window.innerHeight - 0.5
  schedule()
}
function onScroll() {
  sc = scroller ? scroller.scrollTop : (window.scrollY || 0)
  schedule()
}

onMounted(() => {
  reduceMotion.value = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches || false
  if (reduceMotion.value) return
  scroller = document.querySelector('.app-main') || null
  ;(scroller || window).addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('pointermove', onPointer, { passive: true })
  onScroll()
})
onBeforeUnmount(() => {
  ;(scroller || window).removeEventListener('scroll', onScroll)
  window.removeEventListener('pointermove', onPointer)
  if (frame) cancelAnimationFrame(frame)
})

useHead({
  title: 'Nexstrata · Ecosystem modeling on Google Earth Engine',
  meta: [{
    name: 'description',
    content: 'An ecosystem-modeling platform: bind Earth Engine environmental layers to point '
      + 'observations, then model habitat suitability across the landscape. Reference dataset: '
      + 'iNaturalist fungal observations enriched at each coordinate and date.',
  }],
})
</script>

<style scoped>
.home {
  position: relative; max-width: 980px; margin: 0 auto; padding: 40px 20px 80px;
  /* Defaults so the CSS below is valid before the script writes real values, and
     the whole of what the stylesheet reads from JS in one place. */
  --mx: 0; --my: 0; --sc: 0;
}
.home section { position: relative; z-index: 1; }
.home section + section { margin-top: 72px; }
.home h2 {
  font-size: 1.5rem; margin: 0 0 14px; color: var(--text-strong); letter-spacing: -0.01em;
}

/* ── The ambient HUD field ────────────────────────────────────────────────
   Fixed behind everything, parallaxing on scroll and pointer. Purely lit for
   the dark theme; toned right down on light. */
.hud-field {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
}
.hud-grid {
  position: absolute; inset: -20% -20% -20% -20%;
  background-image:
    linear-gradient(rgba(56, 189, 248, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56, 189, 248, 0.06) 1px, transparent 1px);
  background-size: 46px 46px;
  transform: translate3d(calc(var(--mx) * -14px), calc(var(--sc) * -0.04px + var(--my) * -14px), 0);
  mask-image: radial-gradient(ellipse at 50% 30%, #000 30%, transparent 78%);
}
.hud-glow {
  position: absolute; width: 46vw; height: 46vw; max-width: 620px; max-height: 620px;
  border-radius: 50%; filter: blur(70px); opacity: 0.5;
}
.glow-a {
  top: -8%; left: -6%; background: radial-gradient(circle, rgba(56, 189, 248, 0.5), transparent 62%);
  transform: translate3d(calc(var(--mx) * 26px), calc(var(--sc) * 0.05px + var(--my) * 26px), 0);
}
.glow-b {
  top: 34%; right: -10%; background: radial-gradient(circle, rgba(52, 211, 153, 0.42), transparent 62%);
  transform: translate3d(calc(var(--mx) * -30px), calc(var(--sc) * -0.06px + var(--my) * -20px), 0);
}
.hud-scan {
  position: absolute; inset: 0;
  background: repeating-linear-gradient(180deg, rgba(255, 255, 255, 0.018) 0 2px, transparent 2px 4px);
  mix-blend-mode: overlay;
}
:root[data-theme="light"] .hud-glow { opacity: 0.22; }
:root[data-theme="light"] .hud-grid { opacity: 0.5; }

/* ── Section tags: the little HUD labels ──────────────────────────────────── */
.sec-tag {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.18em;
  color: #38bdf8; margin-bottom: 12px;
}
.sec-tag::before {
  content: ''; width: 22px; height: 1px; background: linear-gradient(90deg, #38bdf8, transparent);
}
.sec-tag .road {
  font-style: normal; letter-spacing: 0.12em; color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.4); border-radius: 999px; padding: 1px 7px; font-size: 0.58rem;
}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero { text-align: center; padding-top: 8px; }
.hero-logo {
  margin: 0 auto 18px; border-radius: 28px;
  box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.25), 0 18px 60px rgba(56, 189, 248, 0.18);
}
.layer-back { transform: translate3d(calc(var(--mx) * 10px), calc(var(--sc) * -0.06px), 0); }
.eyebrow {
  display: inline-flex; align-items: center; gap: 8px; margin: 0 auto 14px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px;
  background: color-mix(in srgb, var(--surface) 60%, transparent);
}
.eyebrow .tick {
  width: 7px; height: 7px; border-radius: 50%; background: #34d399;
  box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.22); animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
.hero h1 {
  max-width: 760px; margin: 0 auto 14px;
  font-size: clamp(2.2rem, 6vw, 3.4rem); line-height: 1.05; color: var(--text-strong);
  letter-spacing: -0.025em;
}
.hero h3 {
  max-width: 640px; margin: 0 auto 22px;
  font-size: 1.06rem; font-weight: 400; line-height: 1.5; color: var(--muted);
}
.cta { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
.btn {
  display: inline-flex; align-items: center; gap: 8px; text-decoration: none; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface); color: var(--text); border-radius: 8px;
  padding: 10px 18px; font-size: 0.92rem; font-weight: 600;
  transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;
}
.btn:hover { background: var(--surface-2); transform: translateY(-1px); }
.btn.primary {
  background: linear-gradient(180deg, #34d399, #2b7a3d); border-color: #2b7a3d; color: #04140b;
  box-shadow: 0 8px 24px rgba(52, 211, 153, 0.28);
}
.btn.primary:hover { filter: brightness(1.05); }
.btn.small { padding: 6px 12px; font-size: 0.82rem; }
.btn.ghost { background: var(--surface); }
.stat { margin: 0; color: var(--muted); font-size: 0.85rem; }
.stat strong { color: var(--text); font-variant-numeric: tabular-nums; }

/* ── Pipeline ─────────────────────────────────────────────────────────────
   Four steps in a row on wide screens, a column on a phone. The connective
   arrow lives on each step except the last. */
.flow { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.flow-step {
  position: relative; padding: 16px 16px 18px;
  border: 1px solid var(--border); border-radius: 12px; background: color-mix(in srgb, var(--surface) 82%, transparent);
  display: flex; flex-direction: column; gap: 6px; backdrop-filter: blur(6px);
  transform: translateY(calc(var(--sc) * -0.008px * (var(--n) + 1)));
}
.flow-step::before {
  content: ''; position: absolute; left: 0; top: 14px; bottom: 14px; width: 2px; border-radius: 2px;
  background: linear-gradient(180deg, #38bdf8, #34d399);
}
.flow-k {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.72rem; font-weight: 700;
  color: #38bdf8; letter-spacing: 0.1em;
}
.flow-title { font-size: 1.02rem; font-weight: 700; color: var(--text-strong); }
.flow-body { font-size: 0.83rem; color: var(--muted); line-height: 1.5; }
.flow-arrow {
  position: absolute; right: -11px; top: 50%; transform: translateY(-50%); z-index: 2;
  color: #38bdf8; font-size: 1.1rem; font-weight: 700;
}

/* ── The stack scene ──────────────────────────────────────────────────────── */
.stack-grid { display: grid; grid-template-columns: 1fr minmax(300px, 380px); gap: 40px; align-items: center; }
.stack-text h2 { margin-top: 4px; }
.stack-text p { margin: 0 0 12px; color: var(--text); font-size: 0.95rem; line-height: 1.65; }
.etym { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #38bdf8; font-weight: 700; }

/* The name explanation, given its own weight: it is the idea the whole product
   is named for, so it reads as a definition rather than a clause buried in a
   paragraph. */
.etym-callout {
  display: flex; flex-direction: column; gap: 4px;
  margin: 0 0 16px !important; padding: 12px 16px;
  border-left: 3px solid #38bdf8; border-radius: 0 10px 10px 0;
  background: color-mix(in srgb, #38bdf8 10%, transparent);
}
.etym-word { font-size: 1.15rem; letter-spacing: 0.01em; }
.etym-word .etym { font-size: 1.15rem; }
.etym-def { color: var(--text); font-size: 0.95rem; }

.scene { margin: 0; }
.scene-3d {
  position: relative; height: 480px;
  transform-style: preserve-3d; perspective: 1100px;
}
/* The whole column tilts back and turns gently with scroll and pointer. */
.plates {
  position: absolute; left: 0; right: 0; top: 96px;
  transform-style: preserve-3d;
  transform:
    rotateX(calc(18deg + var(--sc) * 0.004deg))
    rotateZ(calc(var(--mx) * 5deg))
    rotateY(calc(var(--mx) * 8deg));
  transition: transform 0.1s linear;
}
.plate {
  position: relative; margin: 0 auto 12px; width: min(320px, 100%);
  display: flex; align-items: center; gap: 12px; padding: 9px 12px;
  border: 1px solid var(--border); border-radius: 10px;
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  box-shadow: 0 10px 24px var(--shadow), 0 0 0 1px rgba(56, 189, 248, 0.05);
  transform: translateZ(calc(var(--i) * 26px)) translateY(calc(var(--i) * -2px));
}
.plate-viz {
  flex: 0 0 auto; width: 46px; height: 30px; border-radius: 6px;
  background: linear-gradient(120deg, var(--viz)); border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.1);
}
.plate-text { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.s-name { font-size: 0.86rem; font-weight: 700; color: var(--text-strong); }
.s-fields { font-size: 0.72rem; color: var(--muted); line-height: 1.3; }

/* The sensor and its orbit at the top. */
.orbit { position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 220px; height: 92px; }
.orbit-svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.orbit-svg ellipse {
  fill: none; stroke: rgba(56, 189, 248, 0.55); stroke-width: 1; stroke-dasharray: 4 5;
  filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.35));
}
/* The satellite rides the orbit: offset-path traces the same ellipse the SVG
   draws, and offset-distance is how far the page has scrolled, so it travels
   the ring as you read down. The path coordinates match the ellipse above
   (centre 110,46, radii 104,34). */
.sat {
  position: absolute; top: 0; left: 0; width: 36px; height: 18px;
  offset-path: path('M 6 46 A 104 34 0 1 1 214 46 A 104 34 0 1 1 6 46');
  offset-distance: calc(var(--sat, 0) * 100%);
  offset-rotate: 0deg;
}
.sat-body {
  position: absolute; left: 50%; top: 50%; width: 16px; height: 12px; transform: translate(-50%, -50%);
  background: linear-gradient(180deg, #e2e8f0, #94a3b8); border-radius: 3px;
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.7);
}
.sat-panel {
  position: absolute; top: 50%; width: 13px; height: 16px; transform: translateY(-50%);
  background: repeating-linear-gradient(90deg, #1e3a8a 0 2px, #3b82f6 2px 4px); border-radius: 2px;
}
.sat-panel.left { left: 0; } .sat-panel.right { right: 0; }

/* The beam from the sensor down to the ground. */
.beam {
  position: absolute; left: 50%; top: 40px; height: 388px; width: 2px; transform: translateX(-50%);
  background: linear-gradient(180deg, rgba(56, 189, 248, 0), #38bdf8 30%, #34d399);
  box-shadow: 0 0 16px rgba(56, 189, 248, 0.7); z-index: 0;
}
.beam::after {
  content: ''; position: absolute; left: 50%; top: 0; width: 60px; height: 100%; transform: translateX(-50%);
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.18), transparent 70%);
  clip-path: polygon(48% 0, 52% 0, 100% 100%, 0 100%);
}

/* The ground: a wide, faintly glowing arc with the observation landing on it. */
.earth { position: absolute; left: 50%; bottom: 6px; width: 460px; height: 200px; transform: translateX(-50%); }
.earth::before {
  content: ''; position: absolute; left: 50%; top: 0; width: 460px; height: 460px; transform: translateX(-50%);
  border-radius: 50%;
  background: radial-gradient(circle at 50% 0, #0b3d5c 0, #0a2540 40%, transparent 62%);
  border-top: 2px solid rgba(56, 189, 248, 0.6);
  box-shadow: 0 -8px 40px rgba(56, 189, 248, 0.35);
}
.earth-glow {
  position: absolute; left: 50%; top: -6px; width: 460px; height: 40px; transform: translateX(-50%);
  background: radial-gradient(ellipse at 50% 0, rgba(52, 211, 153, 0.5), transparent 70%);
}
.pin {
  position: absolute; left: 50%; top: -6px; width: 14px; height: 14px; transform: translateX(-50%);
  border: 2.5px solid #34d399; border-radius: 50%; background: #04140b;
  box-shadow: 0 0 14px rgba(52, 211, 153, 0.9); z-index: 3;
}
.pin::after {
  content: ''; position: absolute; left: 50%; top: 50%; width: 5px; height: 5px; transform: translate(-50%, -50%);
  border-radius: 50%; background: #34d399;
}
.scene figcaption { text-align: center; font-size: 0.74rem; color: var(--muted); margin-top: 4px; }

.more-links { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 14px 0 0; }
.more-links a { color: var(--accent); font-size: 0.86rem; font-weight: 600; text-decoration: none; }
.more-links a:hover { text-decoration: underline; }

/* ── Modeling ─────────────────────────────────────────────────────────────── */
.model-lede { margin: 0 0 20px; color: var(--text); font-size: 1rem; line-height: 1.65; max-width: 720px; }
.model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-bottom: 20px; }
.model-card {
  border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
  background: color-mix(in srgb, var(--surface) 84%, transparent); backdrop-filter: blur(6px);
  position: relative; overflow: hidden;
}
.model-card::before {
  content: ''; position: absolute; left: 0; top: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #38bdf8, #34d399, transparent);
}
.model-card h3 { margin: 0 0 8px; font-size: 1rem; color: var(--text-strong); }
.model-card p { margin: 0; font-size: 0.86rem; color: var(--muted); line-height: 1.6; }

/* ── Earth Engine ─────────────────────────────────────────────────────────── */
.ee code {
  background: var(--surface-2); border: 1px solid var(--border-soft); border-radius: 4px;
  padding: 1px 5px; font-size: 0.92em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.ee-lede { margin: 0 0 14px; color: var(--muted); font-size: 0.95rem; line-height: 1.65; max-width: 680px; }
.ee-list { list-style: none; margin: 0 0 18px; padding: 0; display: grid; gap: 8px; max-width: 680px; }
.ee-list li { color: var(--muted); font-size: 0.9rem; line-height: 1.55; padding-left: 16px; position: relative; }
.ee-list li::before {
  content: ''; position: absolute; left: 0; top: 0.55em; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent);
}
.ee-list strong { color: var(--text); }
.cta.left { justify-content: flex-start; margin-bottom: 0; flex-wrap: wrap; }

/* ── Features ─────────────────────────────────────────────────────────────── */
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.feature {
  position: relative; display: block; text-decoration: none; overflow: hidden;
  border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
  background: color-mix(in srgb, var(--surface) 84%, transparent);
  transition: transform 0.14s ease, border-color 0.14s ease, background 0.14s ease;
}
.feature:hover { background: var(--surface-2); border-color: #38bdf8; transform: translateY(-2px); }
.feature h3 { margin: 0 0 6px; font-size: 0.98rem; color: var(--text-strong); }
.feature p { margin: 0; font-size: 0.85rem; color: var(--muted); line-height: 1.55; }
.feature-go {
  position: absolute; right: 14px; top: 14px; color: #38bdf8; font-weight: 700; opacity: 0;
  transform: translateX(-4px); transition: opacity 0.14s ease, transform 0.14s ease;
}
.feature:hover .feature-go { opacity: 1; transform: translateX(0); }

/* ── Caveat ───────────────────────────────────────────────────────────────── */
.caveat {
  border: 1px solid var(--border); border-left: 3px solid #fbbf24;
  border-radius: 12px; padding: 20px 22px; background: color-mix(in srgb, var(--surface) 84%, transparent);
}
.caveat p { margin: 0 0 10px; color: var(--muted); font-size: 0.9rem; line-height: 1.65; }
.caveat p:last-child { margin-bottom: 0; }
.caveat strong { color: var(--text); }
.caveat a { color: var(--accent); }

/* ── Foot ─────────────────────────────────────────────────────────────────── */
.foot { text-align: center; }
.auth-cta {
  display: inline-flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: center;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 16px; margin-bottom: 20px;
}
.auth-cta .hint, .auth-cta .signed { font-size: 0.85rem; color: var(--muted); }
.repo-line { margin: 0; }
.repo { display: inline-flex; align-items: center; gap: 8px; color: var(--text); text-decoration: none; font-size: 0.88rem; font-weight: 600; }
.repo:hover { text-decoration: underline; }
.repo-ico { width: 18px; height: 18px; }

/* ── Motion off ────────────────────────────────────────────────────────────
   Both the OS preference and the class the script sets when it sees it. Freezes
   every transform and animation; the page is fully usable static. */
.home.still .hud-grid,
.home.still .hud-glow,
.home.still .layer-back,
.home.still .flow-step,
.home.still .plates { transform: none; }
.home.still .sat, .home.still .eyebrow .tick { animation: none; }
@media (prefers-reduced-motion: reduce) {
  .hud-grid, .hud-glow, .layer-back, .flow-step, .plates { transform: none !important; }
  .sat, .eyebrow .tick, .btn { animation: none; transition: none; }
}

/* ── Narrow screens ────────────────────────────────────────────────────────
   The pipeline and the scene both need real reflow, not just smaller type: a
   3D column of plates that tilts is wrong on a phone, so it lies flat and the
   scene shrinks to a simple labelled stack with the sensor and ground still
   bracketing it. */
@media (max-width: 860px) {
  .stack-grid { grid-template-columns: 1fr; gap: 26px; }
  .flow { grid-template-columns: 1fr 1fr; }
  .flow-arrow { display: none; }
}
@media (max-width: 720px) {
  .home { padding: 26px 16px 60px; }
  .home section + section { margin-top: 52px; }
  .hero .cta { flex-direction: column; align-items: stretch; flex-wrap: nowrap; }
  .hero .cta .btn { width: 100%; box-sizing: border-box; justify-content: center; }
  .flow { grid-template-columns: 1fr; }
  /* Flatten the scene: no perspective tilt, plates as plain full-width cards so
     nothing overlaps, but keep the sensor above and the ground below. */
  .scene-3d { height: auto; perspective: none; padding: 70px 0 96px; }
  .plates { position: static; transform: none; margin-top: 8px; }
  .plate { width: 100%; transform: none; box-shadow: 0 2px 8px var(--shadow); }
  .beam { top: 46px; height: calc(100% - 120px); }
  .orbit { top: 0; }
  .earth { width: 100%; }
  .earth::before, .earth-glow { width: 320px; }
}
</style>
