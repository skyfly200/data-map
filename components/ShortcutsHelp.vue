<template>
  <transition name="fade">
    <div v-if="shortcuts.helpOpen.value" class="sc-backdrop" @click.self="close">
      <div class="sc-modal" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <div class="sc-head">
          <h2>Keyboard shortcuts</h2>
          <button class="sc-close" aria-label="Close" @click="close">×</button>
        </div>

        <p class="sc-note">
          Shortcuts are ignored while you are typing in a field, so they never
          eat a search term. Page-specific keys only work on that page.
        </p>

        <div class="sc-groups">
          <section v-for="g in shortcuts.grouped.value" :key="g.scope">
            <h3>{{ g.scope }}</h3>
            <dl>
              <div v-for="s in g.items" :key="s.id">
                <dt><kbd>{{ shortcuts.prettyKey(s.keys) }}</kbd></dt>
                <dd>{{ s.label }}</dd>
              </div>
            </dl>
          </section>
        </div>
      </div>
    </div>
  </transition>
</template>

<script setup>
const shortcuts = useShortcuts()
const close = () => { shortcuts.helpOpen.value = false }
</script>

<style scoped>
.sc-backdrop {
  position: fixed; inset: 0; z-index: 2000; background: rgba(0, 0, 0, 0.5);
  display: grid; place-items: center; padding: 20px;
}
.sc-modal {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  box-shadow: 0 8px 32px var(--shadow); padding: 20px 22px;
  width: min(720px, 100%); max-height: 82vh; overflow-y: auto; color: var(--text);
}
.sc-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.sc-head h2 { margin: 0; font-size: 1.1rem; }
.sc-close {
  border: 0; background: transparent; color: var(--muted);
  font-size: 1.6rem; line-height: 1; cursor: pointer; padding: 0 4px;
}
.sc-close:hover { color: var(--text); }
.sc-note { color: var(--muted); font-size: 0.8rem; line-height: 1.45; margin: 8px 0 16px; }

.sc-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; }
.sc-groups h3 {
  font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted); margin: 0 0 8px;
}
.sc-groups dl { margin: 0; display: grid; gap: 6px; }
.sc-groups dl > div { display: grid; grid-template-columns: 74px 1fr; gap: 10px; align-items: baseline; }
.sc-groups dt { margin: 0; }
.sc-groups dd { margin: 0; font-size: 0.85rem; }

kbd {
  display: inline-block; font: 600 0.75rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--surface-2); border: 1px solid var(--border);
  border-bottom-width: 2px; border-radius: 4px; padding: 4px 6px; color: var(--text);
}

.fade-enter-active, .fade-leave-active { transition: opacity 0.15s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

@media (max-width: 560px) {
  .sc-modal { padding: 16px; }
  .sc-groups { grid-template-columns: 1fr; }
}
</style>
