<template>
  <!-- One button, one panel. This used to be a checkbox that had to be ticked
       before a cog appeared beside it, which was then the thing that opened the
       options — two controls and two clicks to reach a panel whose first row
       could just as well be the switch. -->
  <PopoverMenu icon="◈" label="Clusters" title="Cluster the loaded observations live with k-means"
               :active="enabled" :badge="enabled ? `k=${k}` : ''">
    <label class="lc-on">
      <input type="checkbox" v-model="enabled" />
      <span>Live clustering</span>
    </label>

    <!-- The settings are shown whether or not it is on, so you can set it up
         and then switch it on, rather than having to switch it on to find out
         what it will do. -->
    <div class="lc-body" :class="{ off: !enabled }">
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
        <p v-else class="empty">No environmental features in this dataset yet, enrich to enable feature clustering.</p>
      </div>

      <div v-if="sizes.length" class="sizes">
        <span v-for="s in sizes" :key="s.label" class="sz">
          <span class="dot" :style="{ background: colorFor(Number(s.label.slice(1))) }"></span>{{ s.label }}: {{ s.n }}
        </span>
      </div>
    </div>
  </PopoverMenu>
</template>

<script setup>
import { colorFor } from '~/composables/useObservations'

const { enabled, k, mode, features, geoWeight, presentFeatures, sizes } = useLiveClusters()

// Slide freely; only re-cluster on release (k-means over the whole dataset is
// cheap but not free), keeping the drag smooth.
const kLocal = ref(k.value)
watch(k, (v) => { kLocal.value = v })
const geoPct = ref(Math.round(geoWeight.value * 100))
watch(geoWeight, (v) => { geoPct.value = Math.round(v * 100) })

function setAll(on) { features.value = on ? presentFeatures.value.map((f) => f.key) : [] }
</script>

<style scoped>
.lc-on {
  display: flex; align-items: center; gap: 8px;
  font-weight: 600; font-size: 0.85rem; cursor: pointer;
  padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.lc-on input { accent-color: var(--accent); }

.lc-body { display: grid; gap: 8px; }
/* Dimmed rather than hidden or disabled: it says these do nothing yet without
   the panel changing height as you tick the box, which would move the controls
   out from under the cursor that just ticked it. */
.lc-body.off { opacity: 0.55; }

.row { display: grid; grid-template-columns: 92px 1fr auto; align-items: center; gap: 8px; color: var(--muted); }
.row span { font-weight: 600; font-size: 0.8rem; }
.row b { color: var(--text); min-width: 2.4em; text-align: right; }
.row select { border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-size: 0.8rem; background: var(--input-bg); color: var(--text); }
.row input[type="range"] { width: 100%; accent-color: var(--accent); }

.feats-head { display: flex; align-items: center; gap: 8px; color: var(--muted); font-weight: 600; font-size: 0.8rem; }
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
