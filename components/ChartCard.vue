<template>
  <section ref="card" class="card">
    <button class="expand" :title="isFull ? 'Exit full screen' : 'Full screen'" @click="toggle">
      {{ isFull ? '✕' : '⤢' }}
    </button>
    <slot />
  </section>
</template>

<script setup>
const card = ref(null)
const isFull = ref(false)

function toggle() {
  const el = card.value
  if (!document.fullscreenElement) el?.requestFullscreen?.()
  else document.exitFullscreen?.()
}

function onChange() { isFull.value = document.fullscreenElement === card.value }
onMounted(() => document.addEventListener('fullscreenchange', onChange))
onBeforeUnmount(() => document.removeEventListener('fullscreenchange', onChange))
</script>

<style scoped>
.card {
  position: relative; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}
.expand {
  position: absolute; top: 8px; right: 8px; z-index: 2; border: 0; background: transparent;
  cursor: pointer; font-size: 0.95rem; color: var(--muted); padding: 4px 6px; border-radius: 6px; line-height: 1;
}
.expand:hover { background: var(--surface-2); color: var(--text); }

.card:fullscreen {
  padding: 48px 64px; display: flex; flex-direction: column; justify-content: center;
  border-radius: 0;
}
.card:fullscreen :deep(svg) { max-height: 78vh; }
.card:fullscreen :deep(.chart-title) { font-size: 1.3rem; }
</style>
