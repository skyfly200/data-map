<template>
  <div class="app">
    <NuxtRouteAnnouncer />
    <header class="app-header">
      <h1>Mushroom Observations</h1>
      <p>iNaturalist finds enriched with terrain &amp; environmental exposure, clustered by similarity.</p>
    </header>
    <main class="app-main">
      <ClientOnly>
        <MushroomMap />
        <template #fallback>
          <div class="map-fallback">Loading map…</div>
        </template>
      </ClientOnly>
    </main>
  </div>
</template>

<script setup>
useHead({
  title: 'data-map · Mushroom Observations',
  meta: [{ name: 'description', content: 'Mushroom observations enriched with terrain and environmental exposure.' }],
})
</script>

<style>
html, body, #__nuxt { height: 100%; margin: 0; }
body { font-family: system-ui, -apple-system, sans-serif; }

.app { display: flex; flex-direction: column; height: 100vh; }

.app-header {
  padding: 12px 20px; background: #1f2933; color: #fff; flex: 0 0 auto;
}
.app-header h1 { margin: 0; font-size: 1.15rem; }
.app-header p { margin: 2px 0 0; font-size: 0.82rem; opacity: 0.8; }

.app-main { flex: 1 1 auto; min-height: 0; }

.map-fallback { display: grid; place-items: center; height: 100%; color: #555; }
</style>
