<template>
  <!-- Two forms of the same list. A column beside the prose where there is room
       for one, and a drawer over it where there is not. Rendering both and
       hiding one keeps a single source for the links. -->
  <nav class="docnav" :class="{ open }" aria-label="On this page">
    <button class="dn-toggle" :aria-expanded="String(open)"
            title="Jump to a section" @click="open = !open">
      <span class="dn-icon" aria-hidden="true">☰</span>
      <span>{{ current ? current.text : 'On this page' }}</span>
    </button>

    <div class="dn-list" :class="{ shown: open }">
      <div class="dn-head">On this page</div>
      <a v-for="h in headings" :key="h.id" :href="`#${h.id}`"
         class="dn-link" :class="[`lvl-${h.level}`, { active: h.id === activeId }]"
         @click="go(h.id, $event)">{{ h.text }}</a>
      <slot />
    </div>
  </nav>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  headings: { type: Array, default: () => [] },
})

const open = ref(false)
const activeId = ref('')

const current = computed(() => props.headings.find((h) => h.id === activeId.value) || null)

/**
 * Which section the reader is in.
 *
 * The last heading that has scrolled past the top of the viewport, rather than
 * whichever is most visible: on a long page with short sections the "most
 * visible" answer flickers between two neighbours as you scroll, and a contents
 * list that jitters is worse than one that lags slightly.
 */
function updateActive() {
  const marker = 90                       // below the sticky app header
  let found = ''
  for (const h of props.headings) {
    const el = document.getElementById(h.id)
    if (!el) continue
    if (el.getBoundingClientRect().top <= marker) found = h.id
    else break
  }
  // Before the first heading, the first entry is the honest answer rather than
  // no entry at all.
  activeId.value = found || props.headings[0]?.id || ''
}

function go(id, e) {
  const el = document.getElementById(id)
  if (!el) return                          // let the browser try the raw anchor
  e.preventDefault()

  // Close first, then scroll on the next frame. On a narrow screen the open
  // drawer is 60vh of the layout, so scrolling and then closing moves the
  // target out from under the scroll — it landed a section or two past where
  // it was asked to go.
  const wasOpen = open.value
  open.value = false
  activeId.value = id
  history.replaceState(null, '', `#${id}`)

  const scroll = () => el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  if (wasOpen) requestAnimationFrame(() => requestAnimationFrame(scroll))
  else scroll()
}

let ticking = false
function onScroll() {
  if (ticking) return
  ticking = true
  requestAnimationFrame(() => { updateActive(); ticking = false })
}

onMounted(() => {
  window.addEventListener('scroll', onScroll, { passive: true })
  // The page scrolls inside .app-main rather than the window, so that is where
  // the event actually fires.
  document.querySelector('.app-main')?.addEventListener('scroll', onScroll, { passive: true })
  updateActive()
})
onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
  document.querySelector('.app-main')?.removeEventListener('scroll', onScroll)
})
</script>

<style scoped>
.docnav { --dn-w: 220px; }

.dn-toggle { display: none; }

.dn-list {
  position: sticky; top: 16px;
  width: var(--dn-w); max-height: calc(100vh - 90px); overflow-y: auto;
  display: flex; flex-direction: column; gap: 1px;
  padding-right: 8px;
}
.dn-head {
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 700; margin-bottom: 6px;
}
.dn-link {
  display: block; padding: 4px 8px; border-radius: 5px;
  color: var(--muted); text-decoration: none; font-size: 0.82rem; line-height: 1.35;
  border-left: 2px solid transparent;
}
.dn-link:hover { color: var(--text); background: var(--surface-2); }
.dn-link.lvl-3 { padding-left: 18px; font-size: 0.78rem; }
.dn-link.active {
  color: var(--accent); border-left-color: var(--accent); font-weight: 600;
  background: var(--surface-2);
}

/* Below this there is no room for a column beside the text, so the same list
   becomes a drawer opened from a bar pinned under the header. */
@media (max-width: 940px) {
  .docnav { position: sticky; top: 0; z-index: 60; }
  .dn-toggle {
    display: flex; align-items: center; gap: 8px; width: 100%;
    min-height: 40px; padding: 0 14px;
    background: var(--surface); color: var(--text);
    border: 0; border-bottom: 1px solid var(--border);
    font: inherit; font-size: 0.85rem; font-weight: 600; cursor: pointer;
    text-align: left;
  }
  .dn-icon { color: var(--muted); }
  .dn-list {
    position: static; width: auto; max-height: 60vh;
    display: none; padding: 10px 14px 14px;
    background: var(--surface); border-bottom: 1px solid var(--border);
  }
  .dn-list.shown { display: flex; }
  .dn-link { padding: 8px 8px; font-size: 0.88rem; }
  .dn-link.lvl-3 { padding-left: 22px; font-size: 0.84rem; }
}
</style>
