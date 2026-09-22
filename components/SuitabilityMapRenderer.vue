<template>
  <div class="suitability-renderer">
    <div class="controls">
      <div class="control-group">
        <label>Visualization</label>
        <select v-model="vizMode">
          <option value="probability">Probability (0–1)</option>
          <option value="binary">Binary (Presence/Absence)</option>
        </select>
      </div>

      <div v-if="vizMode === 'binary'" class="control-group">
        <label>Threshold: {{ threshold }}</label>
        <input type="range" v-model.number="threshold" min="0.05" max="0.95" step="0.05" />
      </div>

      <div class="control-group">
        <label>Confidence Interval</label>
        <label class="checkbox">
          <input type="checkbox" v-model="showCI" />
          <span>Overlay Confidence</span>
        </label>
      </div>
    </div>

    <div class="map-container">
      <!-- The actual Earth Engine tile layer would be rendered here 
           via the map's layer manager. This component provides the 
           controls and state for that layer. -->
      <div class="placeholder-map">
        <p>MaxEnt Suitability Layer Active</p>
        <span class="mode-tag">{{ vizMode }}</span>
        <span v-if="vizMode === 'binary'" class="threshold-tag">Threshold: {{ threshold }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const vizMode = ref('probability')
const threshold = ref(0.5)
const showCI = ref(false)

// In a real implementation, these values would be pushed to 
// the useMapLayers composable to update the tile URL 
// (e.g., adding a threshold param to the GEE tile request).
</script>

<style scoped>
.suitability-renderer {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-soft);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.control-group label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
}

.checkbox {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.86rem;
  cursor: pointer;
}

.map-container {
  height: 300px;
  background: var(--surface-2);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.placeholder-map {
  text-align: center;
  color: var(--muted);
}

.mode-tag {
  display: block;
  margin-top: 8px;
  font-size: 0.7rem;
  background: var(--border);
  padding: 2px 6px;
  border-radius: 4px;
  text-transform: uppercase;
}

.threshold-tag {
  display: block;
  margin-top: 4px;
  font-size: 0.7rem;
  background: var(--accent-soft);
  color: var(--accent);
  padding: 2px 6px;
  border-radius: 4px;
}
</style>
