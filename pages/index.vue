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
</script>

<style scoped>
.map-page { height: 100%; }
.map-fallback { display: grid; place-items: center; height: 100%; color: #555; }
</style>
