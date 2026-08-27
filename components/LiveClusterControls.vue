<template>
  <div class="lc">
    <label class="lc-toggle" title="Cluster the loaded observations live with k-means">
      <input type="checkbox" v-model="enabled" /> Live clusters
    </label>
    <button v-if="enabled" class="lc-gear" :class="{ on: open }" title="Clustering options" @click="open = !open">⚙</button>

    <div v-if="enabled && open" class="lc-panel">
      <label class="row">
        <span>Cluster by</span>
        <select v-model="mode">
          <option value="features">Features</option>
          <option value="geographic">Location</option>
          <option value="both">Features + location</option>
        </select>
      </label>

      <label class="row">
        <span>Clusters (k)</span>
        <input type="range" min="2" max="16" v-model.number="kLocal" @change="k = kLocal" />
        <b>{{ kLocal }}</b>
      </label>

      <label v-if="mode === 'both'" class="row">
        <span>Location weight</span>
        <input type="range" min="0" max="100" v-model.number="geoPct" @change="geoWeight = geoPct / 100" />
        <b>{{ geoPct }}%</b>
      </label>

      <div v-if="mode !== 'geographic'" class="feats">
        <div class="feats-head">
          <span>Features</span>
          <button class="mini" @click="setAll(true)">all</button>
          <button class="mini" @click="setAll(false)">none</button>
        </div>
        <div v-if="presentFeatures.length" class="feats-grid">
          <label v-for="f in presentFeatures" :key="f.key" class="chk">
            <input type="checkbox" :value="f.key" v-model="features" /> {{ f.label }}
          </label>
        </div>
        <p v-else class="empty">No environmental features in this dataset yet — enrich to enable feature clustering.</p>
      </div>

      <div v-if="sizes.length" class="sizes">
        <span v-for="s in sizes" :key="s.label" class="sz">
          <span class="dot" :style="{ background: colorFor(Number(s.label.slice(1))) }"></span>{{ s.label }}: {{ s.n }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { colorFor } from '~/composables/useObservations'

const { enabled, k, mode, features, geoWeight, presentFeatures, sizes } = useLiveClusters()
const open = ref(false)

// Slide freely; only re-cluster on release (k-means over the whole dataset is
// cheap but not free), keeping the drag smooth.
const kLocal = ref(k.value)
watch(k, (v) => { kLocal.value = v })
const geoPct = ref(Math.round(geoWeight.value * 100))
watch(geoWeight, (v) => { geoPct.value = Math.round(v * 100) })

function setAll(on) { features.value = on ? presentFeatures.value.map((f) => f.key) : [] }
</script>

<style scoped>
.lc {
  position: relative; display: inline-flex; align-items: center; gap: 8px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 10px; font-size: 0.82rem; color: var(--text);
}
.lc-toggle { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; cursor: pointer; }
.lc-toggle input { accent-color: var(--accent); }
.lc-gear {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text); cursor: pointer;
  width: 24px; height: 24px; border-radius: 6px; font-size: 0.85rem; line-height: 1; padding: 0;
}
.lc-gear.on, .lc-gear:hover { background: var(--surface-3); }

.lc-panel {
  position: absolute; top: calc(100% + 6px); left: 0; z-index: 1100; width: 260px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 10px 12px; display: grid; gap: 8px;
}
.row { display: grid; grid-template-columns: 92px 1fr auto; align-items: center; gap: 8px; color: var(--muted); }
.row span { font-weight: 600; }
.row b { color: var(--text); min-width: 2.4em; text-align: right; }
.row select { border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-size: 0.8rem; background: var(--input-bg); color: var(--text); }
.row input[type="range"] { width: 100%; accent-color: var(--accent); }

.feats-head { display: flex; align-items: center; gap: 8px; color: var(--muted); font-weight: 600; }
.feats-head .mini { margin-left: auto; border: 1px solid var(--border); background: var(--surface-2); color: var(--text); border-radius: 5px; padding: 1px 7px; font-size: 0.72rem; cursor: pointer; }
.feats-head .mini + .mini { margin-left: 4px; }
.feats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 10px; margin-top: 5px; }
.chk { display: inline-flex; align-items: center; gap: 5px; font-size: 0.78rem; color: var(--text); }
.chk input { accent-color: var(--accent); }
.empty { margin: 4px 0 0; font-size: 0.76rem; color: var(--muted); }

.sizes { display: flex; flex-wrap: wrap; gap: 3px 10px; border-top: 1px solid var(--border-soft); padding-top: 7px; font-size: 0.75rem; }
.sizes .sz { display: inline-flex; align-items: center; gap: 4px; color: var(--text); }
.sizes .dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
</style>
