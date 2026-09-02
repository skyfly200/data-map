<template>
  <div class="guide">
    <MarkdownDoc :source="source" />
    <OptionReference class="ref-block" />
  </div>
</template>

<script setup>
import source from '~/content/guide.md?raw'

useHead({
  title: 'Guide · data-map',
  meta: [{ name: 'description', content: 'What each part of the app does, and what each number means.' }],
})

// Anchored links arrive from tooltips all over the app. Nuxt restores the hash
// on a normal navigation, but a link to a section of the page you are already on
// does not re-render, so the jump is done here as well.
const route = useRoute()
watch(() => route.hash, (hash) => {
  if (!import.meta.client || !hash) return
  nextTick(() => {
    document.querySelector(hash)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}, { immediate: true })
</script>

<style scoped>
/* The reference sits inside the same column the prose uses, so the page reads
   as one document rather than two stacked ones. */
.ref-block { max-width: 880px; margin: 0 auto; padding: 0 20px 64px; }
</style>
