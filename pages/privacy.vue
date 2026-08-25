<template>
  <MarkdownDoc :source="source" />
</template>

<script setup>
import source from '~/content/privacy.md?raw'

useHead({ title: 'Privacy Policy · data-map' })
</script>
