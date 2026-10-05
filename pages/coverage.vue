<template>
  <div class="coverage-page">
    <h2>Field coverage</h2>
    <p class="sub">
      What fraction of the {{ coverage.total.toLocaleString() }} filtered records carry each
      column. A mean over the 23% of rows that have soil moisture is a different claim from a
      mean over all of them.
    </p>
    <p v-if="pending" class="sub">Loading…</p>
    <p v-else-if="!coverage.total" class="sub">No records loaded.</p>
    <template v-else>
      <table class="tbl">
        <thead><tr><th>Field</th><th class="num">Filled</th><th class="num">Coverage</th><th>&nbsp;</th></tr></thead>
        <tbody>
          <tr v-for="f in coverage.fields" :key="f.key">
            <td>{{ f.label }}</td>
            <td class="num">{{ f.filled.toLocaleString() }} / {{ f.total.toLocaleString() }}</td>
            <td class="num">{{ (f.pct * 100).toFixed(1) }}%</td>
            <td class="barcell">
              <span class="bar" :style="{ width: `${f.pct * 100}%`, background: color(f.pct) }"></span>
            </td>
          </tr>
        </tbody>
      </table>
      <h3>By year</h3>
      <div class="scroll">
        <table class="tbl">
          <thead>
            <tr><th>Year</th><th class="num">Records</th><th v-for="f in coverage.fields" :key="f.key" class="num">{{ f.label }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="y in coverage.years" :key="y.year">
              <td>{{ y.year }}</td>
              <td class="num">{{ y.n.toLocaleString() }}</td>
              <td v-for="f in coverage.fields" :key="f.key" class="num">{{ Math.round((y.pct[f.key] || 0) * 100) }}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<script setup>
const { load, pending } = useObservations()
const { coverage } = useAnalysis()
onMounted(() => { load() })
const color = (pct) => (pct >= 0.8 ? 'var(--accent)' : pct >= 0.4 ? '#eda100' : 'var(--danger)')
</script>

<style scoped>
.coverage-page { padding: 16px; max-width: 960px; margin: 0 auto; }
.sub { opacity: 0.75; }
.tbl { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.tbl th, .tbl td { padding: 4px 8px; text-align: left; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.barcell { width: 30%; }
.bar { display: block; height: 8px; border-radius: 4px; }
.scroll { overflow-x: auto; }
</style>
