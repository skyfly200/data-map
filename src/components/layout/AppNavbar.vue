<template>
  <v-app-bar elevation="2" color="green-darken-3">
    <v-app-bar-title>
      <v-icon icon="mdi-mushroom" class="mr-2" />
      Mushroom Data Map
    </v-app-bar-title>
    <template #append>
      <v-btn :to="'/'" variant="text" prepend-icon="mdi-map">Map</v-btn>
      <v-btn :to="'/datasets'" variant="text" prepend-icon="mdi-database">Datasets</v-btn>
    </template>
  </v-app-bar>
</template>
