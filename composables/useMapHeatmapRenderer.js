import { computed, ref, watch } from 'vue'
import { useMapHeatmaps } from '~/composables/useMapHeatmaps'
import { fieldValue } from '~/composables/statistics'

export function useMapHeatmapRenderer({ mapRef, LRef, geoLayerRef, filteredData }) {
  const heatmaps = useMapHeatmaps()
  const {
    mode: heatmapMode, cellSize: heatmapCell, cellShape, seasonDay, seasonWindow,
    activeMode: heatmapMeta, groupedModes, heatmapOpacity, tileOpacity, CELL_SIZES,
  } = heatmaps
  let heatmapLayer = null

  // Debounced rebuild: filteredData changes on every filter interaction, and
  // computeHeatmap is O(n) over all observations. Collapsing rapid-fire updates
  // into one rebuild keeps the map responsive during slider drags.
  const heatmapResult = ref(heatmaps.computeHeatmap(filteredData.value?.features || [], heatmapMode.value))
  let heatmapRebuildTimer = null
  watch(
    [filteredData, heatmapMode],
    () => {
      clearTimeout(heatmapRebuildTimer)
      heatmapRebuildTimer = setTimeout(() => {
        heatmapResult.value = heatmaps.computeHeatmap(
          filteredData.value?.features || [],
          heatmapMode.value,
        )
      }, 50)
    },
    { immediate: true },
  )
  const heatmapLegend = computed(() => heatmapResult.value.legend)

  const emptyHeatmapReason = computed(() => {
    const field = heatmapMeta.value?.field
    const feats = filteredData.value?.features || []
    if (!feats.length) return 'No observations match the current filters.'
    if (!field) return 'Nothing to show for the current filters.'
    const present = feats.some((f) => Number.isFinite(fieldValue(f.properties || {}, field)))
    if (!present) {
      return `No ${heatmapMeta.value.label} values in this dataset. `
        + 'The pipeline has not filled this column in yet, so it will stay blank until it is re-run.'
    }
    return 'No cells at this zoom. Zoom in, or widen the filters.'
  })

  function arrowFor(c) {
    const L = LRef.value
    const span = (c.lat1 - c.lat0) * 0.42
    const len = span * (0.35 + 0.65 * (c.t ?? 0.5))
    const kx = 1 / Math.max(0.2, Math.cos((c.lat * Math.PI) / 180))
    const tipLat = c.lat + c.dy * len
    const tipLon = c.lon + c.dx * len * kx
    const tailLat = c.lat - c.dy * len
    const tailLon = c.lon - c.dx * len * kx
    const barb = len * 0.38
    const head = (deg) => {
      const a = Math.atan2(c.dx, c.dy) + (deg * Math.PI) / 180
      return [tipLat - Math.cos(a) * barb, tipLon - Math.sin(a) * barb * kx]
    }
    const style = { color: c.color, weight: 1.6, opacity: 0.9, interactive: false }
    return [
      L.polyline([[tailLat, tailLon], [tipLat, tipLon]], style),
      L.polyline([head(-28), [tipLat, tipLon], head(28)], style),
    ]
  }

  function renderHeatmap() {
    const map = mapRef.value
    const L = LRef.value
    if (!map || !L) return
    if (heatmapLayer) { heatmapLayer.remove(); heatmapLayer = null }
    const { cells } = heatmapResult.value
    if (!cells.length) return

    const shapes = heatmapResult.value.legend?.type === 'vector'
      ? cells.flatMap((c) => arrowFor(c))
      : cells.map((c) => L.polygon(c.polygon, {
        stroke: false, fillColor: c.color, fillOpacity: heatmapOpacity.value, interactive: false,
      }))

    heatmapLayer = L.layerGroup(shapes)
    heatmapLayer.addTo(map)
    // Keep observation points above the shading.
    if (geoLayerRef.value) geoLayerRef.value.bringToFront()
  }

  watch(heatmapResult, () => renderHeatmap())
  watch(heatmapOpacity, () => renderHeatmap())
  watch([heatmapMode, heatmapCell, cellShape, seasonDay, seasonWindow], () => heatmaps.persist())

  const heatmapCellIndex = computed(() => {
    const index = new Map()
    for (const c of heatmapResult.value.cells || []) index.set(c.key, c)
    return index
  })

  function heatmapCellAt(lat, lon) {
    if (!heatmapResult.value.cells?.length || !heatmapMode.value) return null
    return heatmapCellIndex.value.get(heatmaps.keyAt(lat, lon)) || null
  }

  return {
    heatmaps,
    heatmapMode, heatmapCell, cellShape, seasonDay, seasonWindow,
    heatmapMeta, groupedModes, heatmapOpacity, tileOpacity, CELL_SIZES,
    heatmapResult, heatmapLegend, emptyHeatmapReason,
    renderHeatmap, heatmapCellIndex, heatmapCellAt,
  }
}
