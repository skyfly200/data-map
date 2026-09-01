<template>
  <section ref="card" class="card">
    <button class="expand" :title="isFull ? 'Exit full screen' : 'Full screen'" @click="toggle">
      {{ isFull ? '✕' : '⤢' }}
    </button>
    <div class="card-body">
      <slot />
    </div>
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
/* Every card is the same height, so the gallery is a regular grid instead of a
   ragged one whose rows jump as charts with different category counts load.
   A chart taller than the card (a box plot with 40 species, a long horizontal
   bar chart) is clipped rather than allowed to stretch its row; the fade at the
   bottom marks that there is more, and the ⤢ button opens it full screen. */
.card {
  position: relative; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
  height: var(--chart-card-height, 340px);
  display: flex; flex-direction: column;
}
.card-body {
  flex: 1; min-height: 0; overflow: hidden; position: relative;
}
/* Fade only when the content actually overflows: a shorter chart sits on the
   card background, which the gradient matches at full opacity anyway. */
.card-body::after {
  content: ''; position: absolute; left: 0; right: 0; bottom: 0; height: 26px;
  background: linear-gradient(to bottom, transparent, var(--surface));
  pointer-events: none;
}
.expand {
  position: absolute; top: 8px; right: 8px; z-index: 2; border: 0; background: transparent;
  cursor: pointer; font-size: 0.95rem; color: var(--muted); padding: 4px 6px; border-radius: 6px; line-height: 1;
}
.expand:hover { background: var(--surface-2); color: var(--text); }

/* Full screen is the escape hatch for anything the card clips: no fixed height,
   no clipping, and the body scrolls if the chart is still taller than the
   viewport. */
.card:fullscreen {
  padding: 48px 64px; height: 100%; justify-content: center; border-radius: 0;
}
.card:fullscreen .card-body { overflow: auto; }
.card:fullscreen .card-body::after { display: none; }
.card:fullscreen :deep(svg) { max-height: 78vh; }
.card:fullscreen :deep(.chart-title) { font-size: 1.3rem; }
</style>
