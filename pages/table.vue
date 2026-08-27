<script setup>
// The observations table now lives as a tab on the Data page.
await navigateTo({ path: '/data', query: { tab: 'table' } }, { replace: true })
</script>

<template>
  <div />
</template>
