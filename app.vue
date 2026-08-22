<template>
  <div class="app">
    <NuxtRouteAnnouncer />
    <header class="app-header">
      <div class="brand">
        <h1>Mushroom Observations</h1>
        <p>iNaturalist finds enriched with terrain &amp; environmental exposure, clustered by similarity.</p>
      </div>
      <nav class="app-nav">
        <NuxtLink to="/" class="nav-link">Map</NuxtLink>
        <NuxtLink to="/table" class="nav-link">Table</NuxtLink>
        <NuxtLink to="/charts" class="nav-link">Charts</NuxtLink>
      </nav>
    </header>
    <main class="app-main">
      <NuxtPage />
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
body { font-family: system-ui, -apple-system, sans-serif; color: #1f2933; }

.app { display: flex; flex-direction: column; height: 100vh; }

.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 10px 20px; background: #1f2933; color: #fff; flex: 0 0 auto;
}
.brand h1 { margin: 0; font-size: 1.15rem; }
.brand p { margin: 2px 0 0; font-size: 0.82rem; opacity: 0.8; }

.app-nav { display: flex; gap: 6px; }
.nav-link {
  color: #cbd2d9; text-decoration: none; font-size: 0.9rem; font-weight: 500;
  padding: 6px 12px; border-radius: 6px;
}
.nav-link:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
.nav-link.router-link-exact-active { background: #3e4c59; color: #fff; }

.app-main { flex: 1 1 auto; min-height: 0; overflow: auto; }
</style>
