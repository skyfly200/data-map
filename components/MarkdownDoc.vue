<template>
  <div class="legal">
    <!-- eslint-disable-next-line vue/no-v-html — source is our own static Markdown -->
    <article v-html="html" />
  </div>
</template>

<script setup>
import { renderMarkdown } from '~/composables/useMarkdown'

const props = defineProps({ source: { type: String, required: true } })
const html = computed(() => renderMarkdown(props.source))
</script>

<style scoped>
/* Wide enough for the guide's tables; the legal pages stay comfortably
   within it at their natural measure. */
.legal { max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }
.legal :deep(article) { color: var(--text); line-height: 1.55; }
.legal :deep(h1) { font-size: 1.5rem; margin: 0 0 16px; }
.legal :deep(h2) { font-size: 1.05rem; margin: 26px 0 8px; }
.legal :deep(h3) { font-size: 0.95rem; margin: 20px 0 6px; }
.legal :deep(p) { margin: 0 0 12px; }
.legal :deep(ul) { margin: 0 0 12px; padding-left: 20px; }
.legal :deep(li) { margin: 4px 0; }
.legal :deep(h4) { font-size: 0.9rem; margin: 16px 0 6px; }
/* var(--accent), not a fixed green: the hardcoded colour was near-invisible on
   the dark theme, which is the default. */
.legal :deep(a) { color: var(--accent); }
/* Emphasis should stand out, so it keeps the body colour rather than being
   dimmed to muted. */
.legal :deep(em) { font-style: italic; }
.legal :deep(strong) { color: var(--text-strong); }
.legal :deep(hr) { border: 0; border-top: 1px solid var(--border); margin: 24px 0; }

.legal :deep(code) {
  font: 0.86em/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--surface-2); border: 1px solid var(--border-soft);
  border-radius: 4px; padding: 1px 5px; white-space: nowrap;
}

/* Tables can be wider than the column on a narrow screen, so they scroll inside
   their own box rather than pushing the page sideways. */
.legal :deep(table) {
  width: 100%; border-collapse: collapse; margin: 0 0 16px;
  font-size: 0.88rem; display: block; overflow-x: auto;
}
.legal :deep(thead th) {
  text-align: left; color: var(--muted); font-weight: 600;
  padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.legal :deep(tbody td) {
  padding: 6px 10px; border-bottom: 1px solid var(--border-soft); vertical-align: top;
}
.legal :deep(tbody tr:last-child td) { border-bottom: 0; }
</style>
