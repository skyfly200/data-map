<template>
  <div class="lc" :class="{ open: enabled }">
    <label class="lc-toggle" :title="'Cluster the loaded observations live with k-means'">
      <input type="checkbox" v-model="enabled" /> Live clusters
    </label>
    <template v-if="enabled">
      <select v-model="mode" class="lc-mode" aria-label="Cluster by">
        <option value="features">by features</option>
        <option value="geographic">by location</option>
      </select>
      <label class="lc-k">k
        <input type="range" min="2" max="12" v-model.number="kLocal" @change="k = kLocal" aria-label="Number of clusters" />
        <span>{{ kLocal }}</span>
      </label>
    </template>
  </div>
</template>

<script setup>
const { enabled, k, mode } = useLiveClusters()
// Slide freely; only re-cluster on release (k-means over the whole dataset is
// cheap but not free), keeping the drag smooth.
const kLocal = ref(k.value)
watch(k, (v) => { kLocal.value = v })
</script>

<style scoped>
.lc {
  display: inline-flex; align-items: center; gap: 10px; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 10px; font-size: 0.82rem; color: var(--text);
}
.lc-toggle { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; cursor: pointer; }
.lc-toggle input { accent-color: var(--accent); }
.lc-mode { border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-size: 0.8rem; background: var(--input-bg); color: var(--text); }
.lc-k { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
.lc-k input[type="range"] { width: 84px; accent-color: var(--accent); }
.lc-k span { color: var(--text); font-weight: 600; min-width: 1.2em; text-align: right; }
</style>
