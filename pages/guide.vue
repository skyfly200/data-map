<template>
  <MarkdownDoc :source="source" />
</template>

<script setup>
import source from '~/content/guide.md?raw'

useHead({
  title: 'Guide · data-map',
  meta: [{ name: 'description', content: 'What each part of the app does, and what each number means.' }],
})
</script>
