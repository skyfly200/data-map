<template>
  <div class="plus-code-input">
    <label for="plusCodes" class="block text-sm font-medium text-gray-700">Plus‑Codes (semicolon‑separated)</label>
    <input
      id="plusCodes"
      v-model="input"
      @keyup.enter="apply"
      placeholder="e.g. 8FW4V7V2+5G;8FQ3V0J2+2M"
      class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
    />
    <button @click="apply" class="mt-2 inline-flex items-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
      Apply
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const emitFilters = defineEmits(['update-plus-codes']);

const input = ref('');

function apply() {
  emitFilters('update-plus-codes', input.value.trim());
}
</script>

<style scoped>
.plus-code-input {
  max-width: 400px;
}
</style>
