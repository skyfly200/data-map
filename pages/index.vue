<template>
  <div class="home">
    <section class="hero">
      <AppLogo class="hero-logo" :size="168" />
      <h1>Nexstrata</h1>
      <h3>Point sampling, raster overlays and analysis on Google Earth Engine</h3>
      <p class="lead">
        Nexstrata samples Earth Engine raster layers at point coordinates, renders
        computed layers as map tiles, and exposes the results for filtering, charting
        and statistical analysis. The reference dataset is iNaturalist fungal
        observations, each enriched with environmental variables sampled at its own
        coordinate and date.
      </p>

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

    <!-- ── Capabilities ─────────────────────────────────────────────────────
         One line each. Anything needing more than a line belongs behind the
         link rather than in front of it. -->
    <section class="features">
      <h2>Capabilities</h2>
      <div class="feature-grid">
        <NuxtLink v-for="f in FEATURES" :key="f.title" :to="f.to" class="feature">
          <h3>{{ f.title }}</h3>
          <p>{{ f.body }}</p>
        </NuxtLink>
      </div>
    </section>

    <!-- ── The layers ───────────────────────────────────────────────────────
         The figure carries this section; the words only have to say what it
         is a picture of. The full source table lives in the guide. -->
    <section class="name">
      <h2>The sampling model</h2>
      <div class="name-grid">
        <div class="name-text">
          <p>
            Each record resolves to a single coordinate and acquisition date. Every
            layer is sampled at that coordinate on that date — not interpolated from a
            regional mean, and not taken from the current state of the ground — so a
            find from 2019 carries the 2019 conditions that preceded it.
          </p>
          <p>
            Native resolutions span 10 m to roughly 10 km depending on the source.
            Records whose coordinates are obscured or coarse are flagged, since terrain
            sampled at an obscured point describes somewhere the observation was not.
            <strong class="etym">nex</strong> + <strong class="etym">strata</strong>:
            the binding of the layers at a point.
          </p>
          <p class="more-links">
            <NuxtLink to="/guide#where-the-data-comes-from">Source table, by column</NuxtLink>
            <NuxtLink to="/coverage">Cache coverage and dates</NuxtLink>
          </p>
        </div>

        <figure class="strata" aria-label="The layers sampled at each observation">
          <div v-for="(layer, i) in STRATA" :key="layer.name" class="stratum"
               :style="{ '--i': i, '--inset': `${i * 7}px` }">
            <span class="s-name">{{ layer.name }}</span>
            <span class="s-fields">{{ layer.fields }}</span>
          </div>
          <div class="beam" aria-hidden="true"></div>
          <div class="target" aria-hidden="true"></div>
          <figcaption>One observation, at the bottom of the stack.</figcaption>
        </figure>
      </div>
    </section>

    <!-- ── Earth Engine ─────────────────────────────────────────────────── -->
    <section class="ee">
      <h2>Earth Engine integration</h2>
      <p class="ee-lede">
        No raster tiles are stored. A server function authenticates with a service
        account, calls <code>getMapId()</code> on the target asset and returns a
        short-lived tile template to the client; templates are cached per parameter set
        and re-minted before they expire.
      </p>
      <ul class="ee-list">
        <li>
          <strong>Enrich</strong> — submit a bounding box and date range. Six stages
          sample terrain, land cover, soil moisture, precipitation, temperature and NDVI
          at each point; output is written to object storage as GeoJSON.
        </li>
        <li>
          <strong>Render</strong> — {{ eeLayerCount }} built-in layers computed
          server-side: MODIS and MTBS fire history, GAP forest type at 30 m, SOLUS100
          soil at 100 m, and a Sentinel-2 dNBR composite calculated per request.
        </li>
        <li>
          <strong>Publish</strong> — register an exported asset by ID, band and palette.
          Inputs are validated against a strict grammar before they reach Earth Engine,
          and each layer carries its own access tier.
        </li>
        <li>
          <strong>Analyse</strong> — load a job result as a dataset. It feeds the same
          map, charts, heatmaps and statistics as the reference data.
        </li>
      </ul>
      <div class="cta left">
        <NuxtLink to="/jobs" class="btn primary">Run a job</NuxtLink>
        <NuxtLink to="/guide#your-own-earth-engine-layers" class="btn">Publish a layer</NuxtLink>
        <a class="btn" :href="KAGGLE_URL" target="_blank" rel="noopener noreferrer">Run the pipeline yourself</a>
      </div>
    </section>

    <!-- ── Limitations ──────────────────────────────────────────────────────
         On the page rather than in the guide: it governs how every number
         above should be read, and a caveat nobody reaches is not a caveat. -->
    <section class="caveat">
      <h2>Known limitations</h2>
      <p>
        The reference dataset is <strong>presence-only, opportunistically collected</strong>.
        Record density tracks observer effort and site access as much as it tracks
        occurrence, and absence of records is not evidence of absence. Enrichment adds
        environmental context to each record; it does not correct the sampling bias in
        which records exist.
      </p>
      <p>
        Each view therefore states its own confounds at the point of use rather than in
        documentation. The seasonal heatmaps normalise within each cell, which cancels
        most of the effort bias; the density heatmaps cannot, and say so.
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
import { computed } from 'vue'
import { EE_LAYER_KEYS } from '~/netlify/lib/ee-tile-layers.mjs'

const { user, isAuthed, configured, signOut } = useAuth()
const repoUrl = 'https://github.com/skyfly200/data-map'
const KAGGLE_URL = 'https://www.kaggle.com/code/skylerflywilson/nexstrata-data-enrichment-pipeline'

// Every headline number is counted from the thing it describes rather than
// written into the copy, so none of them can go stale behind the app's back.
const { availableDatasets } = useObservations()
const totalCount = computed(() =>
  availableDatasets.value?.find((d) => d.id === 'all')?.count || 0)

// The temporal fields are excluded: year, month and day-of-year come off the
// record's own date and are not something the pipeline went and sampled, so
// counting them would inflate the claim this line is making.
const TEMPORAL = new Set(['year', 'month', 'day_of_year'])
const ENRICHMENT_COUNT = ALL_NUMERIC.filter((f) => !TEMPORAL.has(f.key)).length

// The built-in catalogue only. Layers a society has registered of its own are
// per-deployment and behind a tier, so counting them here would promise a
// visitor something they may not be able to see.
const eeLayerCount = EE_LAYER_KEYS.length

// The layers named, coarse to fine — the same grouping the pipeline samples in
// and the observation drawer reports in. The figure is the only thing that
// reads these; the full source table lives in the guide.
const STRATA = [
  { name: 'Weather', fields: 'the seven days before the find' },
  { name: 'Canopy', fields: 'NDVI, NDMI' },
  { name: 'Ground', fields: 'soil moisture, texture, land cover' },
  { name: 'Terrain', fields: 'elevation, slope, aspect, wetness' },
  { name: 'Exposure', fields: 'sun, wind' },
]

/**
 * What the platform does, one line each.
 *
 * The page used to carry four goals and seven features at a paragraph apiece,
 * which is a page nobody finishes. Anything that needs more than a line belongs
 * behind the link rather than in front of it.
 */
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
    body: 'Queue an Earth Engine job over a bounding box and date range. Progress reports '
      + 'per stage; results are stored and loadable as a dataset.',
  },
  {
    to: '/guide#your-own-earth-engine-layers',
    title: 'Custom raster layers',
    body: 'Register an exported Earth Engine asset by ID, band and palette. Tile URLs are '
      + 'minted server-side and gated per access tier.',
  },
  {
    to: '/analysis',
    title: 'Statistics',
    body: 'Spearman rank correlations across populated fields, per-species deviation from '
      + 'the dataset mean, with season and effort confounds reported alongside.',
  },
  {
    to: '/charts?tab=build',
    title: 'Charts',
    body: 'Ten chart types over any field pair, saved per account, with shareable URLs '
      + 'that restore the active filters, colouring and map view.',
  },
  {
    to: '/offline',
    title: 'Maps while offline',
    body: 'Named tile areas cached by a service worker, covering every layer drawn when '
      + 'the area was saved. Deletion is reference-counted, so overlapping areas do not '
      + 'remove each other\'s tiles.',
  },
]


useHead({
  title: 'Nexstrata · Point sampling and raster analysis on Google Earth Engine',
  meta: [{
    name: 'description',
    content: 'Sample Earth Engine raster layers at point coordinates, render computed '
      + 'layers as map tiles, and analyse the results. Reference dataset: iNaturalist '
      + 'fungal observations enriched with environmental variables at each coordinate.',
  }],
})
</script>

<style scoped>
.home { max-width: 940px; margin: 0 auto; padding: 40px 20px 72px; }
.home section + section { margin-top: 56px; }
.home h2 {
  font-size: 1.25rem; margin: 0 0 14px; color: var(--text-strong);
  letter-spacing: -0.01em;
}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero { text-align: center; }
.hero-logo { margin: 0 auto 20px; border-radius: 30px; }
/* The title is the name now, with the tagline under it, so the two are sized
   as a pair: the name carries the weight, the tagline is a subtitle rather than
   a second heading competing with it. Capped to the lead's measure so the block
   below does not have to match a line length it cannot. */
.hero h1 {
  max-width: 700px; margin: 0 auto 6px;
  font-size: 2.6rem; line-height: 1.1; color: var(--text-strong);
  letter-spacing: -0.02em;
}
.hero h3 {
  max-width: 620px; margin: 0 auto 18px;
  font-size: 1.05rem; font-weight: 500; line-height: 1.35;
  color: var(--muted); letter-spacing: 0.01em;
}
.lead { max-width: 620px; margin: 0 auto 26px; color: var(--text); font-size: 1.05rem; line-height: 1.6; }

.cta { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-bottom: 22px; }
.btn {
  display: inline-flex; align-items: center; gap: 8px; text-decoration: none; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface); color: var(--text); border-radius: 8px;
  padding: 10px 18px; font-size: 0.92rem; font-weight: 600;
}
.btn:hover { background: var(--surface-2); }
.btn.primary { background: #2b7a3d; border-color: #2b7a3d; color: #fff; }
.btn.primary:hover { background: #246833; }
.btn.small { padding: 6px 12px; font-size: 0.82rem; }
.btn.ghost { background: var(--surface); }

.stat { margin: 0; color: var(--muted); font-size: 0.85rem; }
.stat strong { color: var(--text); font-variant-numeric: tabular-nums; }

/* ── The name ─────────────────────────────────────────────────────────── */
.name-grid { display: grid; grid-template-columns: 1fr minmax(230px, 300px); gap: 28px; align-items: start; }
.name-text p { margin: 0 0 12px; color: var(--text); font-size: 0.94rem; line-height: 1.65; }
.name-text p:last-child { margin-bottom: 0; }
.etym {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #38bdf8; font-weight: 700; letter-spacing: 0.02em;
}

/* The layer stack: the logo's idea with the layers named. Each plate is inset a
   little more than the one above so the stack reads as receding, and the beam
   runs down the middle behind them to the target — the same interleaving the
   mark uses, done in two elements rather than twenty polygons. */
.strata { position: relative; margin: 0; padding: 0 0 46px; }
.stratum {
  position: relative; z-index: 1;
  margin: 0 var(--inset) 6px; padding: 7px 12px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 7px;
  display: flex; flex-direction: column; gap: 1px;
  box-shadow: 0 1px 3px var(--shadow);
}
.s-name { font-size: 0.8rem; font-weight: 700; color: var(--text); }
.s-fields { font-size: 0.71rem; color: var(--muted); line-height: 1.35; }

.beam {
  position: absolute; left: 50%; top: 6px; bottom: 34px; width: 3px;
  transform: translateX(-50%); z-index: 0; border-radius: 2px;
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.15), #38bdf8);
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.65);
}
.target {
  position: absolute; left: 50%; bottom: 20px; width: 15px; height: 15px;
  transform: translateX(-50%); z-index: 2;
  border: 2.5px solid #38bdf8; border-radius: 50%;
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.65);
}
.target::after {
  content: ''; position: absolute; inset: 3px; border-radius: 50%; background: #38bdf8;
}
.strata figcaption {
  position: absolute; left: 0; right: 0; bottom: 0;
  text-align: center; font-size: 0.72rem; color: var(--muted);
}

/* ── Links out of a section, where the depth used to be inline ────────── */
.more-links { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 14px 0 0 !important; }
.more-links a { color: var(--accent); font-size: 0.86rem; font-weight: 600; text-decoration: none; }
.more-links a:hover { text-decoration: underline; }

/* ── Earth Engine ─────────────────────────────────────────────────────── */
.ee code {
  background: var(--surface-2); border: 1px solid var(--border-soft); border-radius: 4px;
  padding: 1px 5px; font-size: 0.92em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.ee-lede { margin: 0 0 14px; color: var(--muted); font-size: 0.94rem; line-height: 1.65; max-width: 660px; }
.ee-list { list-style: none; margin: 0 0 18px; padding: 0; display: grid; gap: 8px; max-width: 660px; }
.ee-list li {
  color: var(--muted); font-size: 0.89rem; line-height: 1.55;
  padding-left: 16px; position: relative;
}
.ee-list li::before {
  content: ''; position: absolute; left: 0; top: 0.55em;
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
}
.ee-list strong { color: var(--text); }
/* The hero centres its buttons; a section in the flow does not. */
.cta.left { justify-content: flex-start; margin-bottom: 0; }

/* ── Features ─────────────────────────────────────────────────────────── */
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.feature {
  display: block; text-decoration: none;
  border: 1px solid var(--border); border-radius: 10px; padding: 15px 17px; background: var(--surface);
}
.feature:hover { background: var(--surface-2); border-color: var(--accent); }
.feature h3 { margin: 0 0 6px; font-size: 0.96rem; color: var(--text); }
.feature p { margin: 0; font-size: 0.84rem; color: var(--muted); line-height: 1.55; }

/* ── Caveat ───────────────────────────────────────────────────────────── */
.caveat {
  border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 10px; padding: 18px 20px; background: var(--surface);
}
.caveat h2 { margin-bottom: 10px; }
.caveat p { margin: 0 0 10px; color: var(--muted); font-size: 0.89rem; line-height: 1.65; }
.caveat p:last-child { margin-bottom: 0; }
.caveat strong { color: var(--text); }
.caveat a { color: var(--accent); }

/* ── Foot ─────────────────────────────────────────────────────────────── */
.foot { text-align: center; }
.auth-cta {
  display: inline-flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: center;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 16px; margin-bottom: 20px;
}
.auth-cta .hint, .auth-cta .signed { font-size: 0.85rem; color: var(--muted); }

/* Its own line: sharing one with the sign-in box put two unrelated calls to
   action side by side, and the GitHub link lost to the buttons beside it. */
.repo-line { margin: 0; }
.repo { display: inline-flex; align-items: center; gap: 8px; color: var(--text); text-decoration: none; font-size: 0.88rem; font-weight: 600; }
.repo:hover { text-decoration: underline; }
.repo-ico { width: 18px; height: 18px; }

@media (max-width: 720px) {
  .home { padding: 28px 16px 56px; }
  .home section + section { margin-top: 42px; }
  .hero h1 { font-size: 2rem; }
  .hero h3 { font-size: 0.96rem; }
  .lead { font-size: 0.98rem; }
  /* The stack sits under the prose rather than beside it, and stops insetting —
     on a narrow screen the receding effect just eats the labels. */
  .name-grid { grid-template-columns: 1fr; gap: 22px; }
  .strata { max-width: 320px; margin: 0 auto; }
  .stratum { margin-left: calc(var(--inset) / 2); margin-right: calc(var(--inset) / 2); }
}
</style>
