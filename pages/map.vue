<template>
  <div class="map-page">
    <ClientOnly>
      <MushroomMap />
      <template #fallback>
        <div class="map-fallback">Loading map…</div>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
// The map itself is a client-only component (Leaflet needs the DOM).
const route = useRoute()
const datasetsApi = useDatasets()
const { addInlineDataset } = useObservations()

// `?dataset=slug` opens a saved dataset (e.g. a finished job's output) on the map.
onMounted(async () => {
  const slug = route.query.dataset
  if (!slug || typeof slug !== 'string') return
  try {
    const ds = await datasetsApi.activate(slug)
    if (!ds) return
    const geojson = await datasetsApi.loadActiveGeojson()
    if (geojson) {
      addInlineDataset(
        { id: `dataset-${ds.id}`, label: ds.title, path: `mem:dataset-${ds.id}` },
        geojson,
      )
    }
  } catch { /* the map still works with whatever dataset was already loaded */ }
})
</script>

<style scoped>
.map-page { height: 100%; }
.map-fallback { display: grid; place-items: center; height: 100%; color: var(--muted); }
</style>
