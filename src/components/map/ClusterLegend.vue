<template>
  <div :class="['cluster-legend', { vertical }]">
    <div
      v-for="(count, cluster) in clusterCounts"
      :key="cluster"
      class="legend-item"
    >
      <span class="dot" :style="{ background: CLUSTER_COLORS[cluster % CLUSTER_COLORS.length] }" />
      <span class="label">C{{ cluster }} ({{ count }})</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  clusterCounts: { type: Object, default: () => ({}) },
  vertical: { type: Boolean, default: false }
})

const CLUSTER_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
  '#ff7f00', '#a65628', '#f781bf', '#999999'
]
</script>

<style scoped>
.cluster-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.cluster-legend.vertical {
  flex-direction: column;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
}
.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
</style>
