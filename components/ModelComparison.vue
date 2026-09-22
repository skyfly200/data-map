<template>
  <div class="comparison">
    <div class="head">
      <h2 class="ct">Model Comparison</h2>
      <p class="sub ct">Compare metrics and predictors across different model runs to identify the most robust fit.</p>
    </div>

    <div v-if="selected.length < 2" class="empty-state">
      <p>Please select at least two models from your list to begin comparison.</p>
    </div>

    <template v-else>
      <div class="grid">
        <!-- Summary Table -->
        <section class="panel wide">
          <h3 class="ct">Performance Summary</h3>
          <table class="tbl">
            <thead>
              <tr>
                <th>Model</th>
                <th class="num">AUC</th>
                <th class="num">SD</th>
                <th>Grade</th>
                <th>Predictors</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="m in selected" :key="m.id">
                <td><strong>{{ m.title }}</strong></td>
                <td class="num">{{ m.results?.[0]?.auc || '—' }}</td>
                <td class="num">{{ m.results?.[0]?.auc_sd || '—' }}</td>
                <td><span class="grade" :class="m.results?.[0]?.grade">{{ m.results?.[0]?.grade || '—' }}</span></td>
                <td>{{ m.predictors.length }}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <!-- Comparison Matrix -->
        <section class="panel wide">
          <h3 class="ct">Predictor Overlap</h3>
          <table class="tbl">
            <thead>
              <tr>
                <th>Predictor</th>
                <th v-for="m in selected" :key="m.id" class="num">{{ m.title }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="p in allPredictors" :key="p">
                <td>{{ p }}</td>
                <td v-for="m in selected" :key="m.id" class="num">
                  <span v-if="m.predictors.includes(p)" class="check">✓</span>
                  <span v-else class="cross">✕</span>
                </td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  selected: { type: Array, required: true },
})

const allPredictors = computed(() => {
  const set = new Set()
  props.selected.forEach(m => m.predictors.forEach(p => set.add(p)))
  return [...set].sort()
})
</script>

<style scoped>
.comparison { padding: 16px 18px; max-width: 1200px; margin: 0 auto; }
.head { margin-bottom: 24px; }
.ct { text-align: center; }
.sub { color: var(--muted); font-size: 0.86rem; }

.empty-state { 
  text-align: center; padding: 40px; background: var(--surface); 
  border: 1px solid var(--border); border-radius: 12px; color: var(--muted); 
}

.grid { display: grid; gap: 20px; }
.panel { 
  background: var(--surface); border: 1px solid var(--border); 
  border-radius: 12px; padding: 20px; 
}
.panel.wide { grid-column: 1 / -1; }

.tbl { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.tbl th { 
  text-align: left; color: var(--muted); font-weight: 600; padding: 8px; 
  border-bottom: 1px solid var(--border); 
}
.tbl td { padding: 8px; border-bottom: 1px solid var(--border-soft); }
.tbl .num { text-align: right; font-variant-numeric: tabular-nums; }

.grade { 
  font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; 
  text-transform: uppercase; font-weight: 600; background: var(--border); 
}
.grade.excellent { background: var(--success); color: var(--success-ink); }
.grade.good { background: var(--accent); color: var(--accent-ink); }
.grade.fair { background: #f0a23f; color: white; }
.grade.weak { background: var(--danger); color: var(--danger-ink); }

.check { color: var(--success); font-weight: bold; }
.cross { color: var(--muted); opacity: 0.3; }
</style>
