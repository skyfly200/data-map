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
.legal { max-width: 760px; margin: 0 auto; padding: 32px 20px 64px; }
.legal :deep(article) { color: #1f2933; line-height: 1.55; }
.legal :deep(h1) { font-size: 1.5rem; margin: 0 0 16px; }
.legal :deep(h2) { font-size: 1.05rem; margin: 26px 0 8px; }
.legal :deep(h3) { font-size: 0.95rem; margin: 20px 0 6px; }
.legal :deep(p) { margin: 0 0 12px; }
.legal :deep(ul) { margin: 0 0 12px; padding-left: 20px; }
.legal :deep(li) { margin: 4px 0; }
.legal :deep(a) { color: #2b7a3d; }
.legal :deep(em) { color: #6b7280; }
.legal :deep(hr) { border: 0; border-top: 1px solid #e5e7eb; margin: 24px 0; }
</style>
