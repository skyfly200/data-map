<template>
  <div class="error-page">
    <header class="err-header">
      <NuxtLink to="/" class="brand">
        <AppLogo :size="28" />
        <span class="brand-name">Nexstrata</span>
      </NuxtLink>
    </header>
    <main class="err-body">
      <p class="err-code">{{ error.statusCode }}</p>
      <h1 class="err-title">{{ title }}</h1>
      <p class="err-msg">{{ error.message || 'Something went wrong.' }}</p>
      <NuxtLink to="/" class="err-home">Go to home ›</NuxtLink>
    </main>
  </div>
</template>

<script setup>
import AppLogo from '~/components/AppLogo.vue'

const props = defineProps({ error: { type: Object, required: true } })
const title = computed(() => {
  if (props.error.statusCode === 404) return 'Page not found'
  if (props.error.statusCode === 403) return 'Access denied'
  return 'An error occurred'
})
</script>

<style scoped>
.error-page {
  min-height: 100vh; display: flex; flex-direction: column;
  font-family: inherit; background: var(--bg, #fff); color: var(--text, #222);
}
.err-header {
  padding: 14px 20px; border-bottom: 1px solid var(--border, #e5e5e5);
  display: flex; align-items: center;
}
.brand {
  display: flex; align-items: center; gap: 8px; text-decoration: none; color: inherit;
}
.brand-name { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; }
.err-body {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; padding: 40px 20px;
}
.err-code {
  font-size: 5rem; font-weight: 800; color: var(--accent, #2a78d6); opacity: 0.15;
  line-height: 1; margin: 0 0 -0.5rem;
}
.err-title { font-size: 1.6rem; font-weight: 700; margin: 0 0 0.5rem; }
.err-msg { color: var(--muted, #888); font-size: 0.95rem; margin: 0 0 1.5rem; max-width: 360px; }
.err-home {
  color: var(--accent, #2a78d6); font-weight: 600; text-decoration: none;
  border: 1px solid var(--accent, #2a78d6); border-radius: 8px; padding: 8px 20px;
  font-size: 0.9rem;
}
.err-home:hover { background: color-mix(in srgb, var(--accent, #2a78d6) 10%, transparent); }
</style>
