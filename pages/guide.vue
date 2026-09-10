<template>
  <div class="guide-shell">
    <DocNav :headings="headings">
      <!-- The option reference is generated rather than written, so it has no
           heading in the Markdown to be picked up with the rest. -->
      <a href="#reference" class="dn-extra" @click="jump('reference', $event)">Option reference</a>
    </DocNav>

    <div class="guide">
      <MarkdownDoc :source="source" />
      <OptionReference class="ref-block" />
    </div>
  </div>
</template>

<script setup>
import source from '~/content/guide.md?raw'
import { extractHeadings } from '~/composables/useMarkdown'

useHead({
  title: 'Guide · Nexstrata',
  meta: [{ name: 'description', content: 'What each part of the app does, and what each number means.' }],
})

// Sections and subsections, but not the h1 (which is the page) or h4s (which are
// details inside a subsection and would triple the list's length).
const headings = extractHeadings(source, { min: 2, max: 3 })

function jump(id, e) {
  const el = document.getElementById(id)
  if (!el) return
  e.preventDefault()
  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  history.replaceState(null, '', `#${id}`)
}

// Anchored links arrive from tooltips all over the app. Nuxt restores the hash
// on a normal navigation, but a link to a section of the page you are already on
// does not re-render, so the jump is done here as well.
const route = useRoute()
watch(() => route.hash, (hash) => {
  if (!import.meta.client || !hash) return
  nextTick(() => {
    document.querySelector(hash)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}, { immediate: true })
</script>

<style scoped>
/* The nav is a column beside the prose, and the prose keeps its own measure
   rather than stretching to fill what is left — a 100-character line is harder
   to read than a narrow one, whatever the window is doing. */
.guide-shell {
  display: grid; grid-template-columns: 220px minmax(0, 1fr);
  gap: 28px; max-width: 1180px; margin: 0 auto; padding: 24px 20px 0;
  /* Not align-items:start. That shrinks the nav column to its own content, and
     a sticky child can only travel inside its parent's box — so the contents
     list scrolled away with the first section instead of staying put. */
}
.guide { min-width: 0; }

/* Inside the nav's slot, so it matches the generated links. */
.dn-extra {
  display: block; margin-top: 8px; padding: 4px 8px;
  border-top: 1px solid var(--border); padding-top: 10px;
  color: var(--muted); text-decoration: none; font-size: 0.82rem;
}
.dn-extra:hover { color: var(--text); }

/* MarkdownDoc centres itself in its own column; inside this grid that column IS
   the text column, so the extra centring only adds a second margin. */
.guide :deep(.legal) { margin: 0; padding: 8px 0 48px; max-width: 780px; }
.ref-block { max-width: 780px; margin: 0; padding: 0 0 64px; }

@media (max-width: 940px) {
  .guide-shell { display: block; padding: 0; max-width: none; }
  .guide :deep(.legal) { padding: 20px 18px 40px; margin: 0 auto; }
  .ref-block { margin: 0 auto; padding: 0 18px 56px; }
  .dn-extra { padding: 8px; font-size: 0.88rem; }
}
</style>
