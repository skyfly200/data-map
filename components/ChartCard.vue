<template>
  <section ref="card" class="card">
    <div class="card-tools">
      <button class="tool" :title="saving ? 'Saving…' : 'Save as PNG (shift-click for SVG)'"
              :disabled="saving" @click="save($event)">
        {{ saving ? '…' : '⤓' }}
      </button>
      <button class="tool" :title="isFull ? 'Exit full screen' : 'Full screen'" @click="toggle">
        {{ isFull ? '✕' : '⤢' }}
      </button>
    </div>
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

// Save the card's chart. PNG by default; shift-click gives SVG, which stays
// sharp at any size and is the better thing to drop into a document.
const saving = ref(false)
const exporter = useImageExport()

async function save(event) {
  const svg = card.value?.querySelector('svg')
  if (!svg || saving.value) return
  saving.value = true
  try {
    const title = card.value.querySelector('.chart-title')?.textContent?.trim()
    const stem = `${exporter.slugify(title, 'chart')}-${exporter.stamp()}`
    // The card background, so the saved image matches the theme on screen
    // rather than arriving with a transparent (or black) plate.
    const background = getComputedStyle(card.value).backgroundColor || '#ffffff'
    if (event?.shiftKey) {
      exporter.download(exporter.svgBlob(svg, { background }), `${stem}.svg`)
    } else {
      exporter.download(await exporter.svgToPng(svg, { scale: 2, background }), `${stem}.png`)
    }
  } catch (err) {
    console.error('Chart export failed:', err)
  } finally {
    saving.value = false
  }
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
.card-tools { position: absolute; top: 8px; right: 8px; z-index: 2; display: flex; gap: 2px; }
.tool {
  border: 0; background: transparent; cursor: pointer; font-size: 0.95rem;
  color: var(--muted); padding: 4px 6px; border-radius: 6px; line-height: 1;
}
.tool:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.tool:disabled { opacity: 0.5; cursor: default; }

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
