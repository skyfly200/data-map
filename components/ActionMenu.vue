<template>
  <!-- The rest of a card's actions, one tap away. The card keeps its main
       action as a real button; everything someone does now and then (save,
       archive, delete, open somewhere else) goes in here, so a list of ten
       cards reads as ten things rather than sixty buttons. -->
  <PopoverMenu icon="⋯" icon-only menu :align="align" :title="title" :label="title"
               btn-class="act-btn" class="act">
    <slot />
  </PopoverMenu>
</template>

<script setup>
defineProps({
  title: { type: String, default: 'More actions' },
  align: { type: String, default: 'right' },
})
</script>

<style scoped>
.act :deep(.act-btn) {
  min-height: 30px; padding: 0 8px; font-size: 1rem; letter-spacing: 0.05em;
  background: transparent; border-color: transparent; color: var(--muted, #777);
}
.act :deep(.act-btn:hover),
.act :deep(.act-btn.on) { background: var(--surface-2, #f1f1f1); border-color: var(--border, #ddd); color: var(--text, #222); box-shadow: none; }
.act :deep(.pop-panel) { padding: 4px; gap: 0; min-width: 190px; }

/* Items: a button or link per action, a label over a group, a rule between. */
.act :deep(.mi) {
  display: flex; align-items: center; gap: 8px; width: 100%;
  padding: 8px 10px; border: 0; border-radius: 6px; background: none;
  color: var(--text, #222); font: inherit; font-size: 0.85rem; text-align: left;
  text-decoration: none; cursor: pointer; white-space: nowrap;
}
.act :deep(.mi:hover:not(:disabled)) { background: var(--surface-2, #f1f1f1); }
.act :deep(.mi:disabled) { opacity: 0.5; cursor: default; }
.act :deep(.mi.danger) { color: #b3492f; }
.act :deep(.mi.on)::after { content: '✓'; margin-left: auto; color: var(--accent, #2b7a3d); }
.act :deep(.mi-sep) { border: 0; border-top: 1px solid var(--border, #e5e5e5); margin: 4px 2px; }
.act :deep(.mi-label) {
  padding: 6px 10px 2px; font-size: 0.66rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted, #777); font-weight: 700;
}
</style>
