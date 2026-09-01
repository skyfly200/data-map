<template>
  <div class="analysis">
    <nav class="tabs">
      <button v-for="(t, i) in TABS" :key="t.key" :class="{ on: tab === t.key }"
              :title="tip(t.hint, String(i + 1))" @click="tab = t.key">
        {{ t.label }}
      </button>
      <ShareMenu class="tab-share" :title="shareTitle" />
    </nav>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <template v-else>
      <p class="scope">
        Analysing <strong>{{ rows.length.toLocaleString() }}</strong> observations —
        whatever the filters on the Data page currently select.
      </p>

      <!-- ─── What relates to what ─────────────────────────────────────────── -->
      <section v-if="tab === 'drivers'" class="pane">
        <ChartCard class="wide">
          <HeatmapChart title="How the environmental variables relate (Spearman ρ)"
            :rows="corrLabels" :cols="corrLabels" :matrix="corr.matrix" :format="rho" />
          <p class="note">
            Rank correlation, so a relationship counts even when it bends. Each cell uses
            only the rows where <em>both</em> variables are present, because coverage is
            very uneven — dropping rows missing any variable would compute the whole
            matrix on an unrepresentative remainder.
          </p>
        </ChartCard>

        <ChartCard class="wide">
          <h3 class="ct">Strongest relationships</h3>
          <table class="tbl">
            <thead><tr><th>Pair</th><th class="num">ρ</th><th class="num">n</th><th>Reading</th></tr></thead>
            <tbody>
              <tr v-for="p in corr.pairs.slice(0, 12)" :key="`${p.a}-${p.b}`">
                <td>{{ p.a }} ↔ {{ p.b }}</td>
                <td class="num" :style="{ color: rhoColor(p.rho) }">{{ p.rho.toFixed(2) }}</td>
                <td class="num">{{ p.n.toLocaleString() }}</td>
                <td class="muted">{{ strength(p.rho) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="note">
            Correlation is not cause, and two confounds run through every row of this
            table. <strong>Season</strong> is the big one: high-elevation finds happen in
            summer and low ones in spring and autumn, which is why elevation and
            temperature appear to rise together — hold the month still and that
            relationship flattens to roughly zero. <strong>Effort</strong> is the other:
            people record where people go.
          </p>
        </ChartCard>
      </section>

      <!-- ─── What each species prefers ────────────────────────────────────── -->
      <section v-else-if="tab === 'species'" class="pane">
        <ChartCard class="wide">
          <HeatmapChart title="How each species differs from average (standard deviations)"
            :rows="profileRows" :cols="profileCols" :matrix="profiles.matrix" :format="z" />
          <p class="note">
            Positive means this species is found higher, warmer, wetter or later than the
            dataset average. Standard deviations, not raw values, so columns in different
            units can be read side by side. Species with at least {{ MIN_SPECIES_OBS }} records.
          </p>
        </ChartCard>

        <ChartCard class="wide">
          <h3 class="ct">Species found together</h3>
          <table v-if="pairs.length" class="tbl">
            <thead><tr><th>Pair</th><th class="num">Shared places</th><th class="num">Lift</th></tr></thead>
            <tbody>
              <tr v-for="p in pairs" :key="`${p.a}|${p.b}`">
                <td><em>{{ p.a }}</em> + <em>{{ p.b }}</em></td>
                <td class="num">{{ p.cells }}</td>
                <td class="num">{{ p.lift.toFixed(1) }}×</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="note">Not enough co-located records in the current selection.</p>
          <p class="note">
            Same ~5 km cell, same month-long window. Scored by <strong>lift</strong> — how
            much more often a pair appears together than their individual frequencies
            predict — because a raw count would just rank the two commonest species first
            whether or not they have anything to do with each other.
          </p>
        </ChartCard>
      </section>

      <!-- ─── Year over year ───────────────────────────────────────────────── -->
      <section v-else-if="tab === 'seasons'" class="pane">
        <ChartCard>
          <LineChart title="Middle of the season, by year" :data="seasonTiming"
            xLabel="Year" yLabel="Median day of year"
            :xFormat="(v) => String(Math.round(v))" :yFormat="(v) => String(Math.round(v))" />
          <p class="note">
            Median, not mean: a handful of winter records would drag a mean badly.
            An upward slope means the season is arriving later.
          </p>
        </ChartCard>

        <ChartCard>
          <LineChart :title="`Median elevation of finds, by year (${unit})`" :data="elevationTrend"
            xLabel="Year" :yLabel="`Elevation (${unit})`"
            :xFormat="(v) => String(Math.round(v))" :yFormat="(v) => Math.round(v).toLocaleString()" />
        </ChartCard>

        <ChartCard>
          <BarChart title="Records per year" :data="effortByYear" :format="(v) => String(v)" />
          <p class="note">
            The control for both charts above. Recording effort has grown steeply, so a
            shift in either trend may be a shift in who is looking rather than in the
            mushrooms.
          </p>
        </ChartCard>

        <ChartCard>
          <BarChart title="Distinct species recorded per year" :data="speciesByYear" :format="(v) => String(v)" />
        </ChartCard>
      </section>

      <!-- ─── How much to trust it ─────────────────────────────────────────── -->
      <section v-else class="pane">
        <ChartCard class="wide">
          <h3 class="ct">Field coverage</h3>
          <table class="tbl">
            <thead><tr><th>Field</th><th class="num">Filled</th><th class="num">Coverage</th><th>&nbsp;</th></tr></thead>
            <tbody>
              <tr v-for="f in coverage.fields" :key="f.key">
                <td>{{ f.label }}</td>
                <td class="num">{{ f.filled.toLocaleString() }}</td>
                <td class="num">{{ (f.pct * 100).toFixed(1) }}%</td>
                <td class="barcell">
                  <span class="bar" :style="{ width: `${f.pct * 100}%`, background: covColor(f.pct) }"></span>
                </td>
              </tr>
            </tbody>
          </table>
          <p class="note">
            A chart drawn from a 20%-covered column looks exactly as confident as one drawn
            from a full column. This is which is which. Anything thin here is thin because
            the enrichment pipeline has not reached those rows yet — re-running it fills them.
          </p>
        </ChartCard>

        <ChartCard class="wide">
          <BarChart title="Enrichment coverage by year (mean across fields)"
            :data="coverageByYear" :format="(v) => `${v}%`" />
          <p class="note">
            Uneven coverage across years makes year-over-year comparison unreliable for the
            thin fields: a trend can be a change in what was measured rather than in what
            was there.
          </p>
        </ChartCard>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useUnits } from '~/composables/useUnits'

const TABS = [
  { key: 'drivers', label: 'What relates to what', hint: 'Which environmental variables move together' },
  { key: 'species', label: 'Species', hint: 'What each species prefers, and which are found together' },
  { key: 'seasons', label: 'Year over year', hint: 'Season timing and elevation by year, against recording effort' },
  { key: 'quality', label: 'Data quality', hint: 'How much of each field is actually filled in' },
]
const MIN_SPECIES_OBS = 20

const { rows, error, pending, load } = useObservations()
const { unit, elevValue } = useUnits()
const analysis = useAnalysis()
const tab = ref('drivers')

const shortcuts = useShortcuts()
const tip = (text, keys) => shortcuts.withKey(text, keys)
shortcuts.register(TABS.map((t, i) => ({
  scope: 'Analysis', keys: String(i + 1), label: t.label, run: () => { tab.value = t.key },
})))

onMounted(load)

const shareTitle = computed(() =>
  `Analysis of ${rows.value.length.toLocaleString()} mushroom observations`)

// ─── Drivers ────────────────────────────────────────────────────────────────
const corr = computed(() => analysis.correlationMatrix.value)
const corrLabels = computed(() => corr.value.fields.map((f) => f.label))
const rho = (v) => (v === null || v === undefined ? '' : Number(v).toFixed(2))
const rhoColor = (v) => (Math.abs(v) >= 0.5 ? 'var(--accent)' : 'var(--text)')

function strength(v) {
  const a = Math.abs(v)
  const dir = v > 0 ? 'rise together' : 'move opposite'
  if (a >= 0.7) return `strong — ${dir}`
  if (a >= 0.4) return `moderate — ${dir}`
  if (a >= 0.2) return `weak — ${dir}`
  return 'little or none'
}

// ─── Species ────────────────────────────────────────────────────────────────
const profiles = computed(() => {
  const p = analysis.speciesProfiles(MIN_SPECIES_OBS, 25)
  return { ...p, matrix: p.species.map((s) => s.z) }
})
const profileRows = computed(() => profiles.value.species.map((s) => `${s.species} (${s.n})`))
const profileCols = computed(() => profiles.value.fields.map((f) => f.label))
const z = (v) => (v === null || v === undefined ? '' : `${v > 0 ? '+' : ''}${Number(v).toFixed(1)}`)

const pairs = computed(() => analysis.coOccurrence({ minObs: MIN_SPECIES_OBS, limit: 15 }))

// ─── Year over year ─────────────────────────────────────────────────────────
const years = computed(() => analysis.byYear.value)
const seasonTiming = computed(() =>
  years.value.filter((y) => y.medianDoy !== null).map((y) => ({ x: y.year, y: y.medianDoy })))
const elevationTrend = computed(() =>
  years.value.filter((y) => y.medianElevation !== null)
    .map((y) => ({ x: y.year, y: elevValue(y.medianElevation) })))
const effortByYear = computed(() =>
  years.value.map((y) => ({ label: String(y.year), short: String(y.year), value: y.n })))
const speciesByYear = computed(() =>
  years.value.map((y) => ({ label: String(y.year), short: String(y.year), value: y.species })))

// ─── Quality ────────────────────────────────────────────────────────────────
const coverage = computed(() => analysis.coverage.value)
const covColor = (pct) => (pct >= 0.8 ? 'var(--accent)' : pct >= 0.4 ? '#eda100' : 'var(--danger)')
const coverageByYear = computed(() => coverage.value.years.map((y) => {
  const vals = Object.values(y.pct)
  const mean = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
  return { label: String(y.year), short: String(y.year), value: Math.round(mean * 100) }
}))
</script>

<style scoped>
.analysis { padding: 16px 18px; max-width: 1400px; margin: 0 auto; }

.tabs { display: flex; align-items: center; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.tabs button {
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  border-radius: 6px; padding: 6px 13px; font-size: 0.86rem; font-weight: 600; cursor: pointer;
}
.tabs button:hover { background: var(--surface-2); }
.tabs button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.tab-share { margin-left: auto; }

.scope { color: var(--muted); font-size: 0.82rem; margin: 0 0 12px; }
.msg { padding: 16px; color: var(--muted); }
.msg.error { color: var(--danger); }

.pane { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
/* Matrices and tables need the room; a 340px column makes them unreadable. */
.pane :deep(.wide) { grid-column: 1 / -1; }
/* These cards hold tables rather than a fixed-height chart, so they size to
   their content instead of clipping it. */
.pane :deep(.card) { height: auto; min-height: 340px; }

.ct { font-size: 0.95rem; font-weight: 600; color: var(--text); margin: 0 0 8px; }
.note { color: var(--muted); font-size: 0.76rem; line-height: 1.45; margin: 8px 0 0; }

.tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.tbl th {
  text-align: left; color: var(--muted); font-weight: 600; padding: 4px 8px;
  border-bottom: 1px solid var(--border);
}
.tbl td { padding: 4px 8px; border-bottom: 1px solid var(--border-soft); }
.tbl .num { text-align: right; font-variant-numeric: tabular-nums; }
.tbl .muted { color: var(--muted); }
.barcell { width: 34%; }
.bar { display: block; height: 8px; border-radius: 4px; }
</style>
