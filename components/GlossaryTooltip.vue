<template>
  <span class="gloss-wrap">
    <span
      class="gloss-term"
      tabindex="0"
      :aria-describedby="uid"
      @mouseenter="show = true"
      @mouseleave="show = false"
      @focus="show = true"
      @blur="show = false"
      @keydown.escape="show = false"
    ><slot /></span>
    <transition name="gloss">
      <div
        v-if="show && definition"
        :id="uid"
        class="gloss-popup"
        role="tooltip"
      >
        <strong class="gloss-head">{{ term ?? $slots.default?.()[0]?.children }}</strong>
        <span class="gloss-def">{{ definition }}</span>
      </div>
    </transition>
  </span>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  term: { type: String, default: null },
  definition: { type: String, required: true },
})

const show = ref(false)
const uid = `gloss-${Math.random().toString(36).slice(2)}`
</script>

<style scoped>
.gloss-wrap { position: relative; display: inline; }

.gloss-term {
  border-bottom: 1px dotted var(--muted, #888);
  cursor: help;
  outline: none;
}
.gloss-term:focus-visible { border-bottom-style: solid; border-bottom-color: #38bdf8; }

.gloss-popup {
  position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
  z-index: 900;
  width: max(220px, 28ch); max-width: min(340px, 90vw);
  background: var(--surface, #fff); color: var(--text, #1a1a1a);
  border: 1px solid var(--border, #ddd);
  border-radius: 8px;
  padding: 10px 12px 10px;
  box-shadow: 0 8px 28px rgba(0,0,0,0.18);
  pointer-events: none;
}

.gloss-head {
  display: block; font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: #38bdf8; margin-bottom: 4px;
}
.gloss-def { font-size: 0.87rem; line-height: 1.5; color: var(--muted, #555); }

/* tiny caret */
.gloss-popup::after {
  content: '';
  position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: var(--border, #ddd);
}
.gloss-popup::before {
  content: '';
  position: absolute; top: calc(100% - 1px); left: 50%; transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: var(--surface, #fff);
  z-index: 1;
}

.gloss-enter-active, .gloss-leave-active { transition: opacity 0.12s ease, transform 0.12s ease; }
.gloss-enter-from, .gloss-leave-to { opacity: 0; transform: translateX(-50%) translateY(4px); }
</style>
