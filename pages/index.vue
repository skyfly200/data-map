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
      <h1>Find where species thrive.</h1>
      <h3>
        Turn your observations into habitat maps. Nexstrata connects your data to environmental
        factors like terrain, climate, and vegetation—then predicts where else species might live.
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
            <NuxtLink to="/guide/sources">Source table, by column</NuxtLink>
            <NuxtLink to="/analysis">How complete each column is</NuxtLink>
          </p>
        </div>

        <figure class="scene" aria-label="A satellite sampling environmental layers down to one point on the ground">
          <div class="scene-3d">
            <!-- The named layers, each with a swatch of what it looks like on the
                 map, stacked in 3D above the Earth. -->
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

            <!-- The sensor beam, straight down through the layers to the surface. -->
            <div class="beam" aria-hidden="true"></div>

            <!-- The Earth, seen close.
                 Not a whole planet in a box: at that size the continents have to
                 be drawn, and line art of a coastline at 150px is the one thing
                 here that cannot look real. So the globe is enormous and mostly
                 below the frame — what you see is a limb curving across the
                 bottom, the way it looks from low orbit, where the horizon is a
                 curve and not a circle.

                 The orbit is one ellipse wide enough to enclose the whole layer
                 stack, so its top arc passes above the plates rather than around
                 a ball beneath them. The satellite rides only that top arc, and
                 sits at its apex — directly over the beam, above every layer —
                 when the section is centred in the window. -->
            <!-- The top half of the orbit only. The bottom would pass behind an
                 Earth that is not in frame, so there is nothing for it to go
                 behind and it would read as a ring lying flat on the page.

                 The viewBox is 100x100 stretched to the element's own box
                 (preserveAspectRatio="none"), so "50" means the middle of the
                 box on either axis whatever size the box is. That is what lets
                 the satellite — an HTML element, not an SVG one — be put on the
                 same ellipse from CSS: both read the same --orbit-* numbers. -->
            <svg class="orbit" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <linearGradient id="orbitFade" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0" stop-color="#38bdf8" stop-opacity="0" />
                  <stop offset="0.24" stop-color="#38bdf8" stop-opacity="0.8" />
                  <stop offset="0.76" stop-color="#38bdf8" stop-opacity="0.8" />
                  <stop offset="1" stop-color="#38bdf8" stop-opacity="0" />
                </linearGradient>
              </defs>
              <path class="orbit-arc" d="M 0 50 A 50 50 0 0 1 100 50" />
            </svg>

            <!-- The satellite, on that same ellipse. Its angle comes straight
                 from --sat, so 0.5 is the apex: dead centre, above every plate,
                 on the line the beam comes down. -->
            <div class="sat" aria-hidden="true">
              <span class="sat-body"></span>
              <span class="sat-panel left"></span>
              <span class="sat-panel right"></span>
            </div>

            <!-- The limb. One circle far bigger than its box, so only the top of
                 it is in frame. No coastlines: at this scale you are over one
                 place, and the graticule plus the atmosphere is what says Earth
                 without drawing a map badly. -->
            <div class="limb" aria-hidden="true">
              <svg class="limb-svg" viewBox="0 0 1200 1200" preserveAspectRatio="xMidYMin slice">
                <defs>
                  <clipPath id="limbClip"><circle cx="600" cy="620" r="600" /></clipPath>
                  <!-- Lit from above and to the left, and dark: the rest of the
                       page is close to black, and a daylight-bright planet in
                       the corner of it would be the only thing anyone saw.

                       Scaled to the band that is actually in frame, not to the
                       sphere. Only the top seventh of the circle is ever on
                       screen, so a gradient sized to the whole ball would put
                       its entire falloff below the bottom edge and leave the
                       visible part one flat colour. -->
                  <radialGradient id="limbOcean" cx="41%" cy="2%" r="17%">
                    <stop offset="0%" stop-color="#1a6e97" />
                    <stop offset="45%" stop-color="#0b3f60" />
                    <stop offset="100%" stop-color="#03151f" />
                  </radialGradient>
                  <!-- The atmosphere, brightest right at the edge. -->
                  <radialGradient id="limbAir" cx="50%" cy="50%" r="50%">
                    <stop offset="0.93" stop-color="#7dd3fc" stop-opacity="0" />
                    <stop offset="0.985" stop-color="#7dd3fc" stop-opacity="0.55" />
                    <stop offset="1" stop-color="#7dd3fc" stop-opacity="0" />
                  </radialGradient>
                </defs>

                <g clip-path="url(#limbClip)">
                  <circle cx="600" cy="620" r="600" fill="url(#limbOcean)" />

                  <!-- A real graticule rather than six arbitrary ellipses:
                       parallels are circles of latitude seen edge-on, so their
                       height shrinks with the cosine of the latitude and they
                       stack toward the pole. Meridians all meet there. Drawn to
                       one rule, which is what makes a wireframe read as a
                       sphere instead of as a net thrown over a disc. -->
                  <g class="grat">
                    <ellipse v-for="p in PARALLELS" :key="`p${p.lat}`"
                             cx="600" :cy="p.cy" :rx="p.rx" :ry="p.ry" />
                    <path v-for="m in MERIDIANS" :key="`m${m}`" :d="meridian(m)" />
                  </g>
                </g>

                <circle cx="600" cy="620" r="600" class="limb-air" fill="url(#limbAir)" />
                <circle cx="600" cy="620" r="599" class="limb-edge" />
              </svg>

              <!-- The observation, where the beam meets the surface. -->
              <span class="pin"></span>
            </div>
          </div>
          <figcaption>One observation on Earth, and the layers sampled above it.</figcaption>
        </figure>
      </div>
    </section>

    <!-- ── Modeling (the point of the enrichment) ───────────────────────────── -->
    <section class="model">
      <div class="sec-tag">Modeling <em class="road">Roadmap</em></div>
      <h2>From points to surfaces</h2>
      <p class="model-lede">
        Turn your observations into predictions. The model finds patterns in your data and
        creates a complete map showing where species are likely to live—even where nobody has looked.
      </p>
      <div class="model-grid">
        <div class="model-card">
          <h3>Maximum entropy (MaxEnt)</h3>
          <p>
            Works with presence-only data: where species were found, not where they weren't.
            MaxEnt learns from environmental conditions at sighting locations to predict habitat
            suitability everywhere, scored from 0 to 1.
          </p>
        </div>
        <div class="model-card">
          <h3>Features from the stack</h3>
          <p>
            Uses terrain, canopy, soil, weather and exposure as predictors. The same layers
            you see on the map become the model's input, so predictions match your data.
          </p>
        </div>
        <div class="model-card">
          <h3>Honest about bias</h3>
          <p>
            Accounts for where people look. Nexstrata weights background data by observer
            effort and clearly labels uncertainties in every prediction.
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
        <NuxtLink to="/guide/layers" class="btn">Publish a layer</NuxtLink>
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
        <NuxtLink to="/guide/reference">Every control has a reference entry</NuxtLink>
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
            <span class="hint">Read access is unauthenticated. Use the account menu above to sign in and persist saved views or submit jobs.</span>
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
  { k: '01', title: 'Observe', body: 'Your species sightings from iNaturalist or uploads.' },
  { k: '02', title: 'Enrich', body: 'Add environmental data like terrain, weather, and soil to each point.' },
  { k: '03', title: 'Model', body: 'Find patterns in where species occur based on their environment.' },
  { k: '04', title: 'Predict', body: 'See likely habitat across the entire map.' },
]

// The layers named, coarse to fine, each with a CSS gradient that echoes how the
// layer actually paints on the map, so the stack shows the data, not just its
// name. `viz` is dropped straight into a linear-gradient in the style below.

// ─── The graticule ───────────────────────────────────────────────────────────
// Drawn to one rule instead of by hand, which is the difference between a
// wireframe that reads as a sphere and one that reads as a net thrown over a
// disc. The globe is a circle of radius R centred below the frame; a parallel at
// latitude φ is that circle's cross-section seen from slightly above, so it sits
// R·sin φ up from the equator and is R·cos φ wide, squashed vertically by how
// far the viewpoint is tilted.

const GLOBE_R = 600
const GLOBE_CX = 600
const GLOBE_CY = 620

// How far the pole is tilted away from the line of sight. Near a right angle,
// so the pole sits almost at the top of the silhouette and the part of the
// sphere in frame is the high-latitude cap — which is where the observation is.
const BETA = (78 * Math.PI) / 180
const SIN_B = Math.sin(BETA)
const COS_B = Math.cos(BETA)

// Where each projected pole lands. Every meridian ends at these two points, so
// they are what makes the meridians converge instead of running parallel.
const POLE_N = Math.round(GLOBE_CY - GLOBE_R * SIN_B)
const POLE_S = Math.round(GLOBE_CY + GLOBE_R * SIN_B)

/**
 * A parallel, as the ellipse it projects to.
 *
 * Orthographic projection of a circle of latitude: it keeps its full width,
 * R·cos φ, and is flattened vertically by the cosine of the tilt. It sits
 * R·sin φ·sin β above the centre, which is what crowds the parallels together
 * as they approach the pole.
 */
const PARALLELS = Array.from({ length: 8 }, (_, i) => {
  const deg = (i + 1) * 10
  const lat = (deg * Math.PI) / 180
  return {
    lat: deg,
    cy: Math.round(GLOBE_CY - GLOBE_R * Math.sin(lat) * SIN_B),
    rx: Math.round(GLOBE_R * Math.cos(lat)),
    ry: Math.max(2, Math.round(GLOBE_R * Math.cos(lat) * COS_B)),
  }
})

const MERIDIANS = [-80, -60, -40, -20, 0, 20, 40, 60, 80]

/**
 * One meridian, as the half-ellipse it projects to.
 *
 * Every meridian runs pole to pole, so they all start and end at the same two
 * points and differ only in how far they bow out — by the sine of their
 * longitude from the one facing you. One rule for all of them is what makes the
 * wireframe read as a sphere rather than as a net thrown over a disc.
 */
function meridian(lonDeg) {
  const rx = Math.abs(Math.round(GLOBE_R * Math.sin((lonDeg * Math.PI) / 180)))
  const ry = Math.round(GLOBE_R * SIN_B)
  if (rx < 2) return `M ${GLOBE_CX} ${POLE_N} L ${GLOBE_CX} ${POLE_S}`
  const sweep = lonDeg > 0 ? 1 : 0
  return `M ${GLOBE_CX} ${POLE_N} A ${rx} ${ry} 0 0 ${sweep} ${GLOBE_CX} ${POLE_S}`
}

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
    to: '/guide/layers',
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
  // Where the satellite sits on its arc: 0 at the left horizon, 1 at the right,
  // and 0.5 — the apex, directly over the beam and above every layer — when the
  // stack section is centred in the window. Measured against that section
  // rather than the whole page, because "above the layers" is a fact about
  // where the layers are and not about how long the page happens to be.
  const stack = el.querySelector('.stack-sec')
  let along = 0.5
  if (stack) {
    const box = stack.getBoundingClientRect()
    const view = window.innerHeight || 1
    // 0 when the section's middle is a screen below the middle of the window,
    // 1 when it is a screen above it.
    const offset = (box.top + box.height / 2) - view / 2
    along = 0.5 - offset / view / 2
  }
  el.style.setProperty('--sat', Math.min(1, Math.max(0, along)).toFixed(4))
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
  position: relative; height: 560px;
  transform-style: preserve-3d; perspective: 1100px;

  /* The orbit, in one place, because two elements have to agree on it: the SVG
     that draws the ellipse and the satellite that rides it. Horizontal as a
     percentage so the ellipse stays as wide as the column; vertical in pixels
     so the apex keeps clearing the plates when the column narrows. */
  --orbit-inset: 3%;
  --orbit-cy: 248px;
  --orbit-ry: 224px;

  /* A window onto a scene rather than a diagram of one. You see a piece of the
     planet and a piece of the orbit, and both run out of the frame instead of
     ending — which is what makes it read as a view and not as an illustration
     with a border. Everything fades before it reaches an edge, so nothing has a
     cut end. */
  -webkit-mask-image:
    radial-gradient(76% 70% at 50% 54%, #000 42%, rgba(0, 0, 0, 0.62) 78%, transparent 100%);
  mask-image:
    radial-gradient(76% 70% at 50% 54%, #000 42%, rgba(0, 0, 0, 0.62) 78%, transparent 100%);
}
/* The whole column tilts back and turns gently with scroll and pointer. */
.plates {
  position: absolute; left: 0; right: 0; top: 28px;
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

/* The beam from the sensor down to the surface. Ends at the globe's north pole,
   where the pin sits. */
.beam {
  position: absolute; left: 50%; top: 30px; height: 330px; width: 2px; transform: translateX(-50%);
  background: linear-gradient(180deg, rgba(56, 189, 248, 0), #38bdf8 30%, #34d399);
  box-shadow: 0 0 16px rgba(56, 189, 248, 0.7); z-index: 0;
}
.beam::after {
  content: ''; position: absolute; left: 50%; top: 0; width: 60px; height: 100%; transform: translateX(-50%);
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.18), transparent 70%);
  clip-path: polygon(48% 0, 52% 0, 100% 100%, 0 100%);
}

/* The limb. The SVG is square and far wider than this box, and the box shows
   only its top — so the circle's curve crosses the frame as a horizon rather
   than sitting inside it as a ball. */
.limb {
  position: absolute; left: 50%; bottom: -46px; transform: translateX(-50%);
  width: 175%; height: 300px; z-index: 2;
  overflow: hidden; pointer-events: none;
  /* Its own fade downwards, on top of the scene's. Without it the planet ends
     on a straight horizontal cut at the bottom of the figure, which is the one
     edge the radial mask cannot reach — it is nearest the mask's centre. */
  -webkit-mask-image: linear-gradient(to bottom, #000 0 26%, rgba(0, 0, 0, 0.35) 68%, transparent 96%);
  mask-image: linear-gradient(to bottom, #000 0 26%, rgba(0, 0, 0, 0.35) 68%, transparent 96%);
}
/* The taller this is, the bigger the sphere is against the frame and the
   shallower the curve across it — which is the whole difference between a
   horizon and a ball. */
.limb-svg { position: absolute; left: 0; top: 0; width: 100%; height: 620%; }

.limb-svg .grat {
  fill: none; stroke: rgba(125, 211, 252, 0.22); stroke-width: 1.3;
  vector-effect: non-scaling-stroke;
}
.limb-edge {
  fill: none; stroke: rgba(125, 211, 252, 0.7); stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.limb-air { pointer-events: none; }

/* ── The orbit ────────────────────────────────────────────────────────────────
   Wide enough to enclose the whole stack, so the arc passes ABOVE the plates.
   Behind them in z, so a plate the arc crosses still reads as nearer. Both ends
   fade out rather than stopping, because an orbit with two visible ends is a
   croquet hoop. */
.orbit {
  /* Behind the plates in every layout: on a phone they are in normal flow, and
     an in-flow box paints under any positioned one, so z-index 0 would put the
     arc in front of the cards there and behind them on a desktop. */
  position: absolute; pointer-events: none; z-index: -1;
  left: var(--orbit-inset); right: var(--orbit-inset);
  top: calc(var(--orbit-cy) - var(--orbit-ry));
  height: calc(var(--orbit-ry) * 2);
}
.orbit-arc {
  fill: none; stroke: url(#orbitFade); stroke-width: 1.4;
  stroke-dasharray: 5 7; stroke-linecap: round;
  filter: drop-shadow(0 0 7px rgba(56, 189, 248, 0.45));
  vector-effect: non-scaling-stroke;
}

/* The satellite rides that same ellipse, by angle rather than by arc length: at
   --sat 0 it is on the left horizon, at 1 on the right, and at 0.5 — the middle
   of the section in the window — it is at the apex, dead centre, directly over
   the beam and above every plate. Above everything in z: it is the thing doing
   the looking. */
.sat {
  position: absolute; width: 34px; height: 18px; z-index: 6; pointer-events: none;
  --ang: calc((1 - var(--sat, 0.5)) * 180deg);
  left: calc(50% + (50% - var(--orbit-inset)) * cos(var(--ang)));
  top: calc(var(--orbit-cy) - var(--orbit-ry) * sin(var(--ang)));
  transform: translate(-50%, -50%);
  transition: left 0.12s linear, top 0.12s linear;
}
.sat-body {
  position: absolute; left: 50%; top: 50%; width: 15px; height: 11px; transform: translate(-50%, -50%);
  background: linear-gradient(180deg, #e2e8f0, #94a3b8); border-radius: 3px;
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.75);
}
.sat-panel {
  position: absolute; top: 50%; width: 12px; height: 15px; transform: translateY(-50%);
  background: repeating-linear-gradient(90deg, #1e3a8a 0 2px, #3b82f6 2px 4px); border-radius: 2px;
}
.sat-panel.left { left: 0; } .sat-panel.right { right: 0; }

/* The observation, on the surface. */
.pin {
  position: absolute; left: 50%; top: 24px; width: 13px; height: 13px; transform: translateX(-50%);
  z-index: 5;
  border: 2.5px solid #34d399; border-radius: 50%; background: #04140b;
  box-shadow: 0 0 14px rgba(52, 211, 153, 0.95); z-index: 3;
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
  .scene-3d { height: auto; perspective: none; padding: 14px 0 210px; }
  .plates { position: static; transform: none; margin-top: 8px; }
  .plate { width: 100%; transform: none; box-shadow: 0 2px 8px var(--shadow); }
  /* The beam runs from the plates down to the limb sitting below them. */
  .beam { top: 40px; height: calc(100% - 210px); }
  /* A phone column is narrow, so the same ellipse would be a thin spike. Pull
     the apex down and widen the ends past the frame, which keeps the arc
     reading as a curve rather than as a pair of vertical lines. */
  .scene-3d { --orbit-inset: -10%; --orbit-cy: 208px; --orbit-ry: 192px; }
  .limb { height: 250px; bottom: -40px; width: 210%; }
  .limb-svg { height: 760%; }
}
</style>
