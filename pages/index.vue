<template>
  <div class="home">
    <section class="hero">
      <AppLogo class="hero-logo" :size="168" />
      <h1>Nexstrata</h1>
      <h3>Data abundance distilled into actionable insights</h3>
      <p class="lead">
        Mushroom observations from iNaturalist, each one carrying the ground it was
        found on: terrain, weather, canopy, soil and exposure. So you can ask what the
        places have in common, instead of only where the dots are.
      </p>

      <div class="cta">
        <NuxtLink to="/map" class="btn primary">Open the map</NuxtLink>
        <NuxtLink to="/charts?tab=build" class="btn">Build a chart</NuxtLink>
        <NuxtLink to="/guide" class="btn">Read the guide</NuxtLink>
      </div>

      <p v-if="totalCount" class="stat">
        <strong>{{ totalCount.toLocaleString() }}</strong> observations ·
        <strong>{{ ENRICHMENT_COUNT }}</strong> environmental fields on each ·
        <strong>7</strong> taxonomic ranks
      </p>
    </section>

    <!-- ── The name ─────────────────────────────────────────────────────────
         Worth explaining rather than leaving as a word: it says what the app
         does, and the logo says the same thing in a picture. -->
    <section class="name">
      <h2>Why “Nexstrata”</h2>
      <div class="name-grid">
        <div class="name-text">
          <p>
            <strong class="etym">strata</strong>: layers. Where a mushroom grows is not
            one fact but a stack of them: the weather of the week before, the canopy over
            it, the moisture in the soil, the shape and aspect of the ground, how much sun
            and wind that shape lets through.
          </p>
          <p>
            <strong class="etym">nex</strong>: from <em>nexus</em>, a binding together.
            An observation is the one place all those layers meet. Somebody stood at a
            point, found something, and every layer had a value there at that moment.
          </p>
          <p class="name-close">
            So: the binding of the layers at a point. That is what the mark shows: a beam
            passing down through every stratum and landing on a geotag on the ground. This
            app is the beam.
          </p>
        </div>

        <!-- The same idea as the logo, with the layers actually named. -->
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

    <!-- ── Goals ───────────────────────────────────────────────────────────── -->
    <section class="goals">
      <h2>What it is trying to do</h2>
      <ol class="goal-list">
        <li v-for="goal in GOALS" :key="goal.title">
          <h3>{{ goal.title }}</h3>
          <p>{{ goal.body }}</p>
        </li>
      </ol>
    </section>

    <!-- ── What's in it ────────────────────────────────────────────────────── -->
    <section class="features">
      <h2>What is in it</h2>
      <div class="feature-grid">
        <NuxtLink v-for="f in FEATURES" :key="f.title" :to="f.to" class="feature">
          <h3>{{ f.title }}</h3>
          <p>{{ f.body }}</p>
        </NuxtLink>
      </div>
    </section>

    <!-- ── The caveat ──────────────────────────────────────────────────────
         Not a footnote. It is the thing that most changes how you should read
         everything above, so it gets a section rather than small print. -->
    <section class="caveat">
      <h2>What it cannot tell you</h2>
      <p>
        These are <strong>opportunistic observations, not surveys</strong>. Somewhere with
        many records may have many mushrooms, or may simply be near a trailhead. A map of
        finds is partly a map of where people walk, and no amount of environmental data
        attached to those finds fixes that.
      </p>
      <p>
        So the app says where it stands. Views that correct for recording effort say so;
        views that cannot say that too. Every control has an entry in the
        <NuxtLink to="/guide#reference">option reference</NuxtLink> covering what it does
        <em>and</em> where it will mislead you: the seasonal heatmaps divide by each
        cell's own total so effort cancels out, obscured coordinates are flagged because
        their terrain describes somewhere the mushroom probably was not, and a blank cell
        means nobody looked there rather than nothing grows there.
      </p>
    </section>

    <section class="foot">
      <ClientOnly>
        <div class="auth-cta" v-if="configured">
          <template v-if="isAuthed">
            <span class="signed">Signed in as <strong>{{ user?.email || 'your account' }}</strong>.</span>
            <NuxtLink to="/data?tab=fetch" class="btn small">Fetch a taxon</NuxtLink>
            <button class="btn small ghost" @click="signOut">Sign out</button>
          </template>
          <template v-else>
            <span class="hint">Browsing is open to everyone. Sign in to pull new taxa from iNaturalist:</span>
            <NuxtLink to="/login" class="btn small">Sign in</NuxtLink>
            <NuxtLink to="/login?mode=signup" class="btn small ghost">Sign up</NuxtLink>
          </template>
        </div>
      </ClientOnly>
      <br><br>
      <a class="repo" :href="repoUrl" target="_blank" rel="noopener noreferrer">
        <IconGithub class="repo-ico" /> <span>View the source on GitHub</span>
      </a>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const { user, isAuthed, configured, signOut } = useAuth()
const repoUrl = 'https://github.com/skyfly200/data-map'

// The manifest is a couple of kilobytes and app.vue already loads it, so the
// headline number is the real one rather than a figure baked into the copy that
// goes stale the next time the pipeline runs.
const { availableDatasets } = useObservations()
const totalCount = computed(() =>
  availableDatasets.value?.find((d) => d.id === 'all')?.count || 0)

// Counted from the field registry rather than written out, for the same reason.
// The temporal fields are excluded: year, month and day-of-year come off the
// record's own date and are not something the pipeline went and sampled, so
// counting them would inflate the claim this line is making.
const TEMPORAL = new Set(['year', 'month', 'day_of_year'])
const ENRICHMENT_COUNT = ALL_NUMERIC.filter((f) => !TEMPORAL.has(f.key)).length

// The layers named, coarse to fine — the same grouping the pipeline samples in
// and the observation drawer reports in.
const STRATA = [
  { name: 'Weather', fields: 'the seven days before the find' },
  { name: 'Canopy', fields: 'NDVI, NDMI' },
  { name: 'Ground', fields: 'soil moisture, land cover' },
  { name: 'Terrain', fields: 'elevation, slope, aspect, wetness' },
  { name: 'Exposure', fields: 'sun, wind' },
]

const GOALS = [
  {
    title: 'Attach the place to the find',
    body: 'A record from iNaturalist is a name, a date and a coordinate. The pipeline '
      + 'samples what the ground was like there: terrain from a digital elevation model, '
      + 'canopy and moisture from satellite imagery, the weather in the week before. So '
      + 'the observation carries its own context instead of just its position.',
  },
  {
    title: 'Make the layers legible',
    body: 'Forty-eight thousand overlapping dots show where the data is dense and nothing '
      + 'else. Binned into a hex grid they show what is actually in an area: how many '
      + 'species, when it fruits, how steep and wet and shaded the ground is. Charts and '
      + 'statistics take it further, over whatever the filters currently select.',
  },
  {
    title: 'Be honest about the limits',
    body: 'Every view says what it is distorted by, in the view itself rather than in '
      + 'documentation nobody opens. A number that cannot be trusted is worse than no '
      + 'number, so where the data will mislead you, the app says so at the point of use.',
  },
  {
    title: 'Work where it is used',
    body: 'A map of where things grow is most often read standing in the place it '
      + 'describes, which is where there is least likely to be a signal. The app, the '
      + 'observations and an area of map tiles can all be saved into the browser, and it '
      + 'installs to a home screen.',
  },
]

const FEATURES = [
  {
    to: '/map',
    title: 'Map and heatmaps',
    body: 'Every observation as a point, coloured by any dimension. Under them, a hex grid '
      + 'summarising density, species richness, seasonal activity, in-season hotspots, or '
      + 'the cell mean of any environmental field.',
  },
  {
    to: '/map',
    title: 'Reference layers',
    body: 'Rainfall and radar, land surface temperature, ESA land cover at 10 m, soil '
      + 'moisture, greenness, hillshade, hiking trails and US land ownership. Each has a '
      + 'key, and each says what it gets wrong.',
  },
  {
    to: '/data',
    title: 'Taxonomy at every rank',
    body: 'Kingdom through species, resolved from each record’s real ancestry. Filter, '
      + 'group, colour and analyse at whichever rank answers your question, and import a '
      + 'species, a family or a whole kingdom.',
  },
  {
    to: '/charts?tab=build',
    title: 'Charts you build',
    body: 'Scatter, line, area, bar, stacked bar, box, histogram, heatmap, radar and '
      + 'donut. Compose your own, save the ones worth keeping, and share a link that '
      + 'reopens exactly what you were looking at.',
  },
  {
    to: '/analysis',
    title: 'Statistics',
    body: 'Rank correlations across every populated field, species fingerprints in '
      + 'standard deviations from the dataset mean, and the confounds (season and '
      + 'recording effort) named rather than left for you to find.',
  },
  {
    to: '/coverage',
    title: 'Coverage',
    body: 'What fraction of records actually carry each enrichment field, because a mean '
      + 'over the 23% of rows that have soil moisture is a different claim from a mean '
      + 'over all of them.',
  },
]

useHead({
  title: 'Nexstrata · Mushroom observations, read through the layers',
  meta: [{
    name: 'description',
    content: 'Mushroom observations from iNaturalist, each carrying the terrain, weather, '
      + 'canopy and soil it was found in, mapped, charted and honest about its limits.',
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
.hero h1 {
  /* Capped to the lead's measure. Left to run the full 940px it set a line
     length the paragraph under it could not match, and the two read as
     unrelated blocks rather than as a headline and its subtitle. */
  max-width: 700px; margin: 0 auto 16px;
  font-size: 2.05rem; line-height: 1.15; color: var(--text-strong);
  letter-spacing: -0.02em;
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
.name-close { color: var(--muted) !important; border-top: 1px solid var(--border-soft); padding-top: 12px; }

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

/* ── Goals ────────────────────────────────────────────────────────────── */
.goal-list {
  list-style: none; counter-reset: goal; margin: 0; padding: 0;
  display: grid; gap: 18px;
}
.goal-list li { counter-increment: goal; padding-left: 42px; position: relative; }
.goal-list li::before {
  content: counter(goal); position: absolute; left: 0; top: -1px;
  width: 28px; height: 28px; border-radius: 50%;
  display: grid; place-items: center;
  background: var(--surface-2); border: 1px solid var(--border);
  color: var(--muted); font-size: 0.82rem; font-weight: 700;
}
.goal-list h3 { margin: 0 0 4px; font-size: 0.98rem; color: var(--text); }
.goal-list p { margin: 0; color: var(--muted); font-size: 0.89rem; line-height: 1.6; }

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

.repo { display: inline-flex; align-items: center; gap: 8px; color: var(--text); text-decoration: none; font-size: 0.88rem; font-weight: 600; }
.repo:hover { text-decoration: underline; }
.repo-ico { width: 18px; height: 18px; }

@media (max-width: 720px) {
  .home { padding: 28px 16px 56px; }
  .home section + section { margin-top: 42px; }
  .hero h1 { font-size: 1.65rem; }
  .lead { font-size: 0.98rem; }
  /* The stack sits under the prose rather than beside it, and stops insetting —
     on a narrow screen the receding effect just eats the labels. */
  .name-grid { grid-template-columns: 1fr; gap: 22px; }
  .strata { max-width: 320px; margin: 0 auto; }
  .stratum { margin-left: calc(var(--inset) / 2); margin-right: calc(var(--inset) / 2); }
}
</style>
