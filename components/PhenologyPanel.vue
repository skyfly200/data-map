<template>
  <div class="phen">
    <div class="ph-head">
      <label class="ph-pick">
        <span>Species</span>
        <select v-model="species">
          <option v-for="s in candidates" :key="s.name" :value="s.name">
            {{ s.name }} ({{ s.years }} years, {{ s.n.toLocaleString() }} obs)
          </option>
        </select>
      </label>
      <label class="ph-pick">
        <span>Timing <HelpLink option="analysis-phenology" /></span>
        <select v-model="timingKey">
          <option value="relative">Corrected for recording effort</option>
          <option value="median">Raw median date</option>
        </select>
      </label>
    </div>

    <p v-if="!candidates.length" class="ph-empty">
      No species in the current filters has enough observations across enough complete
      years to say anything about timing. This needs {{ MIN_OBS }} or more finds in each of
      at least {{ MIN_YEARS }} years.
    </p>

    <template v-else-if="rowsForSpecies.length">
      <!-- ── The headline: did it move, and by how much ──────────────────── -->
      <div class="ph-verdict">
        <div class="v-num" :class="trendClass">
          {{ trend.slope === null ? '—' : `${trend.slope > 0 ? '+' : ''}${trend.slope.toFixed(1)}` }}
          <small>days / year</small>
        </div>
        <div class="v-text">
          <strong>{{ verdict }}</strong>
          <span class="v-sub">
            Over {{ trend.n }} complete years<template v-if="trend.span">, {{ trend.span[0] }} to {{ trend.span[1] }}</template>.
            Rank correlation {{ trend.rho === null ? '—' : trend.rho.toFixed(2) }}, which says how
            consistently the order holds rather than how big the shift is.
          </span>
        </div>
      </div>

      <div class="ph-grid">
        <!-- ── When it fruited, year by year ─────────────────────────────── -->
        <figure class="ph-card wide">
          <figcaption>
            When it fruited, by year
            <span class="cap-sub">Bar is the middle half of the season; dot is the median find.</span>
          </figcaption>
          <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Fruiting timing by year">
            <g v-for="t in monthTicks" :key="t.doy">
              <line :x1="x(t.doy)" :y1="20" :x2="x(t.doy)" :y2="H - 22" class="grid" />
              <text :x="x(t.doy)" :y="H - 8" class="tick">{{ t.label }}</text>
            </g>
            <g v-for="(r, i) in rowsForSpecies" :key="r.year">
              <text :x="30" :y="yRow(i) + 4" class="ylab">{{ r.year }}</text>
              <line :x1="x(r.q25)" :y1="yRow(i)" :x2="x(r.q75)" :y2="yRow(i)" class="iqr" />
              <circle :cx="x(r.median)" :cy="yRow(i)" r="4" class="med" />
              <text :x="W - 6" :y="yRow(i) + 4" class="nlab">{{ r.n }}</text>
            </g>
          </svg>
        </figure>

        <!-- ── What moved it ─────────────────────────────────────────────── -->
        <figure class="ph-card wide">
          <figcaption>
            What moved it
            <span class="cap-sub">
              Correlation of each condition with the timing above. <strong>Held apart</strong> is the
              same correlation with the rival family (water against heat, heat against water) held
              constant, because a wet year is often a cool one and either can borrow the other's credit.
            </span>
          </figcaption>
          <table class="ph-tbl">
            <thead>
              <tr><th>Condition</th><th class="num">Correlation</th><th class="num">Held apart</th><th class="num">Years</th></tr>
            </thead>
            <tbody>
              <tr v-for="d in drivers" :key="d.key" :class="{ weak: d.n < 5 }">
                <td>
                  {{ d.label }}
                  <span class="fam" :class="d.family">{{ d.family }}</span>
                </td>
                <td class="num"><span class="bar" :style="barStyle(d.rho)"></span>{{ fmt(d.rho) }}</td>
                <td class="num">{{ fmt(d.partial) }}</td>
                <td class="num">{{ d.n }}</td>
              </tr>
            </tbody>
          </table>
          <p class="ph-note">
            Positive means more of it goes with a <em>later</em> season; negative with an earlier one.
            With {{ trend.n }} years behind these, treat anything under about 0.6 as a hint rather than
            a finding.
          </p>
        </figure>

        <!-- ── The week before a find ────────────────────────────────────── -->
        <figure class="ph-card">
          <figcaption>
            Rain in the week before a find
            <span class="cap-sub">
              This species against every observation in view, so the comparison is with a day
              somebody was out recording rather than with nothing.
            </span>
          </figcaption>
          <svg class="lead" :viewBox="`0 0 ${LW} ${LH}`" role="img" aria-label="Rainfall in the seven days before a find">
            <g v-for="(p, i) in leadUp.species" :key="p.lag">
              <rect :x="lx(i) - 11" :y="ly(p.mean)" width="10" :height="Math.max(0, LH - 26 - ly(p.mean))" class="b-sp" />
              <rect v-if="leadUp.baseline" :x="lx(i) + 1" :y="ly(leadUp.baseline[i].mean)"
                    width="10" :height="Math.max(0, LH - 26 - ly(leadUp.baseline[i].mean))" class="b-bg" />
              <text :x="lx(i)" :y="LH - 8" class="tick">{{ p.lag === 0 ? 'day of' : `-${p.lag}` }}</text>
            </g>
            <text :x="4" :y="12" class="tick">mm</text>
          </svg>
          <p class="ph-legend">
            <span class="sw sp"></span>{{ species }}
            <span class="sw bg"></span>all observations
          </p>
        </figure>

        <!-- ── Date or threshold ─────────────────────────────────────────── -->
        <figure class="ph-card">
          <figcaption>
            Calendar or conditions?
            <span class="cap-sub">
              If fruiting waits for a threshold rather than a date, the conditions at the peak should
              hold steadier across years than the date does.
            </span>
          </figcaption>
          <table class="ph-tbl compact">
            <tbody>
              <tr v-for="t in thresholds" :key="t.label">
                <td>{{ t.label }}</td>
                <td class="num">{{ t.cv === null ? '—' : t.cv.toFixed(2) }}</td>
                <td class="verdict-cell">{{ t.note }}</td>
              </tr>
            </tbody>
          </table>
          <p class="ph-note">
            Variation is measured as a coefficient of variation, which has no units, so millimetres
            and days can be set beside each other at all. Lower is steadier.
          </p>
        </figure>
      </div>

      <p class="ph-caveat">
        <strong>Read this as a hypothesis generator, not a result.</strong> These are opportunistic
        observations: a median date is partly a median of when people went out, which is what the
        effort correction is for and why it is the default. Six or eight years is very little to fit
        anything to, the weather is reconstructed from the seven-day windows the observations carry
        rather than measured at a station, and correlation across years cannot separate a cause from
        anything that moved with it. Every number here is worth following up, and none is worth
        quoting on its own.
      </p>
    </template>

    <p v-else class="ph-empty">
      {{ species }} does not have {{ MIN_OBS }} finds in enough complete years within the current
      filters. Widen the filters, or pick another species.
    </p>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import {
  completeYears, conditionsByYear, driverTable, leadUpProfile, relativeTiming,
  thresholdTest, timingByYear, timingTrend,
} from '~/composables/phenology'

// A species needs both: enough finds in a year for a median to mean anything,
// and enough years for a trend to be more than two points and a ruler.
const MIN_OBS = 20
const MIN_YEARS = 4

const props = defineProps({
  // The filtered features, so this follows the same filters as every other view.
  features: { type: Array, default: () => [] },
})

const species = ref('')
const timingKey = ref('relative')

// Years the dataset actually ran to completion. The current one stops at today,
// so its median lands wherever the season had got to and every species in it
// looks dramatically early.
const complete = computed(() => completeYears(props.features))

const bySpecies = computed(() => {
  const map = new Map()
  for (const f of props.features) {
    const s = f?.properties?.species
    if (!s) continue
    if (!map.has(s)) map.set(s, [])
    map.get(s).push(f)
  }
  return map
})

/** Only species this analysis can actually say something about. */
const candidates = computed(() => {
  const out = []
  for (const [name, rows] of bySpecies.value) {
    const years = timingByYear(rows, { minObs: MIN_OBS }).filter((r) => complete.value.has(r.year))
    if (years.length >= MIN_YEARS) out.push({ name, years: years.length, n: rows.length })
  }
  return out.sort((a, b) => b.years - a.years || b.n - a.n)
})

watch(candidates, (list) => {
  if (list.length && !list.some((c) => c.name === species.value)) species.value = list[0].name
}, { immediate: true })

// The background is every observation in view, which is what the effort
// correction subtracts out.
const background = computed(() =>
  timingByYear(props.features, { minObs: 100 }).filter((r) => complete.value.has(r.year)))

const rowsForSpecies = computed(() => {
  const rows = bySpecies.value.get(species.value)
  if (!rows) return []
  const timing = timingByYear(rows, { minObs: MIN_OBS }).filter((r) => complete.value.has(r.year))
  if (timing.length < MIN_YEARS) return []
  return conditionsByYear(rows, relativeTiming(timing, background.value), {
    allFeatures: props.features,
  })
})

const trend = computed(() => timingTrend(rowsForSpecies.value, timingKey.value))
const drivers = computed(() => driverTable(rowsForSpecies.value, { timingKey: timingKey.value }))

const trendClass = computed(() => {
  const s = trend.value.slope
  if (s === null || Math.abs(s) < 0.5) return 'flat'
  return s < 0 ? 'earlier' : 'later'
})

const verdict = computed(() => {
  const { slope, rho } = trend.value
  if (slope === null) return 'Not enough years to fit a trend'
  const dir = slope < 0 ? 'earlier' : 'later'
  const strength = Math.abs(rho ?? 0)
  if (Math.abs(slope) < 0.5) return 'Holding roughly steady'
  if (strength < 0.4) return `Drifting ${dir}, but the years do not line up`
  if (strength < 0.7) return `Trending ${dir}`
  return `Consistently ${dir}, year after year`
})

const leadUp = computed(() => {
  const rows = bySpecies.value.get(species.value) || []
  return leadUpProfile(rows, props.features)
})

const thresholds = computed(() => {
  const rows = rowsForSpecies.value
  const dates = rows.map((r) => r.median)
  const out = [{
    label: 'The calendar date itself',
    cv: thresholdTest(dates, dates).cvDate,
    note: 'the baseline to beat',
  }]
  for (const [key, label] of [['rain30', 'Rain over the previous 30 days'],
    ['rain90', 'Rain over the previous 90 days'],
    ['gdd30', 'Degree days over the previous 30']]) {
    const t = thresholdTest(rows.map((r) => r[key]), dates)
    out.push({
      label,
      cv: t.cvValue,
      note: t.steadier === null ? 'too few years'
        : t.steadier ? 'steadier than the date, which points at a threshold'
          : 'more variable than the date',
    })
  }
  return out
})

// ─── Geometry ───────────────────────────────────────────────────────────────
const W = 560
const H = computed(() => 40 + rowsForSpecies.value.length * 22 + 24)
const x = (doy) => 46 + ((doy - 1) / 365) * (W - 46 - 34)
const yRow = (i) => 28 + i * 22
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const monthTicks = MONTHS.map((label, i) => ({ label, doy: Math.round(i * 30.44) + 1 }))

const LW = 300
const LH = 150
const lx = (i) => 34 + i * ((LW - 46) / 7) + 10
const leadMax = computed(() => {
  const vs = [...leadUp.value.species, ...(leadUp.value.baseline || [])]
    .map((p) => p.mean).filter(Number.isFinite)
  return Math.max(0.5, ...vs)
})
const ly = (v) => (Number.isFinite(v) ? (LH - 26) - (v / leadMax.value) * (LH - 46) : LH - 26)

const fmt = (v) => (Number.isFinite(v) ? v.toFixed(2) : '—')
function barStyle(rho) {
  if (!Number.isFinite(rho)) return { width: '0' }
  return {
    width: `${Math.min(1, Math.abs(rho)) * 38}px`,
    background: rho < 0 ? 'var(--accent)' : '#c98b3a',
  }
}
</script>

<style scoped>
.phen { display: flex; flex-direction: column; gap: 14px; }

.ph-head { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end; }
.ph-pick { display: flex; flex-direction: column; gap: 4px; font-size: 0.8rem; }
.ph-pick span { color: var(--muted); font-weight: 600; }
.ph-pick select {
  background: var(--input-bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 8px; font-size: 0.85rem; max-width: 340px;
}

.ph-empty { color: var(--muted); font-size: 0.88rem; line-height: 1.6; margin: 0; }

.ph-verdict {
  display: flex; align-items: center; gap: 18px;
  border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; background: var(--surface);
}
.v-num { font-size: 2rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; white-space: nowrap; }
.v-num small { display: block; font-size: 0.68rem; font-weight: 600; color: var(--muted); margin-top: 4px; }
.v-num.earlier { color: var(--accent); }
.v-num.later { color: #c98b3a; }
.v-num.flat { color: var(--muted); }
.v-text { min-width: 0; }
.v-text strong { display: block; font-size: 1rem; margin-bottom: 3px; }
.v-sub { color: var(--muted); font-size: 0.8rem; line-height: 1.5; }

.ph-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.ph-card {
  margin: 0; border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; background: var(--surface); min-width: 0;
}
.ph-card.wide { grid-column: 1 / -1; }
.ph-card figcaption { font-weight: 600; font-size: 0.9rem; margin-bottom: 8px; }
.cap-sub { display: block; font-weight: 400; color: var(--muted); font-size: 0.76rem; line-height: 1.5; margin-top: 3px; }
/* Capped rather than stretched to the card. These are row charts and bar
   charts whose natural size IS their readable size; letting a 560-unit viewBox
   fill a 1364px card scaled it 2.4x and turned a 328px chart into 781px of
   mostly gap. */
.ph-card svg { width: 100%; max-width: 640px; height: auto; display: block; margin: 0 auto; }
.ph-card svg.lead { max-width: 340px; }

.grid { stroke: var(--grid); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 9px; text-anchor: middle; }
.ylab { fill: var(--muted); font-size: 10px; text-anchor: end; font-variant-numeric: tabular-nums; }
.nlab { fill: var(--muted); font-size: 9px; text-anchor: end; }
.iqr { stroke: var(--accent); stroke-width: 6; stroke-linecap: round; opacity: 0.35; }
.med { fill: var(--accent); }
.b-sp { fill: var(--accent); }
.b-bg { fill: var(--muted); opacity: 0.45; }

.ph-legend { display: flex; align-items: center; gap: 6px; font-size: 0.74rem; color: var(--muted); margin: 6px 0 0; flex-wrap: wrap; }
.sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.sw.sp { background: var(--accent); }
.sw.bg { background: var(--muted); opacity: 0.45; margin-left: 8px; }

.ph-tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.ph-tbl th { text-align: left; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; padding-bottom: 4px; }
.ph-tbl td { padding: 4px 0; border-top: 1px solid var(--border-soft); }
.ph-tbl .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ph-tbl tr.weak { opacity: 0.55; }
.ph-tbl.compact td:first-child { width: 45%; }
.verdict-cell { color: var(--muted); font-size: 0.76rem; padding-left: 10px; }
.bar { display: inline-block; height: 7px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.fam {
  display: inline-block; margin-left: 6px; font-size: 0.64rem; text-transform: uppercase;
  letter-spacing: 0.05em; padding: 1px 5px; border-radius: 3px; color: var(--accent-ink);
}
.fam.water { background: #4a86c8; }
.fam.heat { background: #c98b3a; }

.ph-note { margin: 8px 0 0; color: var(--muted); font-size: 0.75rem; line-height: 1.5; }
.ph-caveat {
  margin: 0; border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 12px 14px; background: var(--surface);
  color: var(--muted); font-size: 0.82rem; line-height: 1.6;
}
.ph-caveat strong { color: var(--text); }

@media (max-width: 780px) {
  .ph-grid { grid-template-columns: 1fr; }
  .ph-verdict { flex-direction: column; align-items: flex-start; gap: 10px; }
}
</style>
