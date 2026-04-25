<template>
  <v-app>
    <AppNavbar />
    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script setup>
import AppNavbar from '@/components/layout/AppNavbar.vue'
</script>
