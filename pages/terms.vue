<template>
  <MarkdownDoc :source="source" />
</template>

<script setup>
import source from '~/content/terms.md?raw'

useHead({ title: 'Terms of Service · data-map' })
</script>
