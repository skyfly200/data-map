<template>
  <div class="guide-shell">
    <DocNav :headings="headings">
      <!-- The guide is nine documents now, so the column has to answer "where
           am I in the guide" before it answers "where am I on this page". -->
      <template #top>
        <div class="dn-head">Guide</div>
        <NuxtLink v-for="p in GUIDE_PAGES" :key="p.slug" :to="guidePath(p.slug)"
                  class="gp-link" :class="{ here: p.slug === slug }"
                  :title="p.blurb">{{ p.title }}</NuxtLink>
        <div class="gp-rule"></div>
      </template>
    </DocNav>

    <div class="guide">
      <MarkdownDoc v-if="source" :source="source" />
      <OptionReference v-if="slug === 'reference'" class="ref-block" />

      <!-- A reader who finishes a page of a split document has nowhere to go,
           where a reader who finished the single page had simply finished. -->
      <nav v-if="prev || next" class="guide-more" aria-label="Guide pages">
        <NuxtLink v-if="prev" :to="guidePath(prev.slug)" class="gm">
          <span class="gm-dir">← Previous</span>
          <span class="gm-title">{{ prev.title }}</span>
          <span class="gm-blurb">{{ prev.blurb }}</span>
        </NuxtLink>
        <NuxtLink v-if="next" :to="guidePath(next.slug)" class="gm next">
          <span class="gm-dir">Next →</span>
          <span class="gm-title">{{ next.title }}</span>
          <span class="gm-blurb">{{ next.blurb }}</span>
        </NuxtLink>
      </nav>
    </div>
  </div>
</template>

<script setup>
import { GUIDE_PAGES, guidePage, guidePath } from '~/composables/guidePages'
import { findGuideAnchor, guideSource } from '~/composables/guideContent'
import { extractHeadings } from '~/composables/useMarkdown'

const route = useRoute()
const router = useRouter()

// [[slug]] makes the parameter optional, so /guide and /guide/map are one page.
// An array arrives when a route has repeated segments; taking the first keeps
// the lookup below total rather than letting an odd URL produce `undefined`.
const slug = computed(() => {
  const raw = route.params.slug
  return String(Array.isArray(raw) ? raw[0] || '' : raw || '')
})

const page = computed(() => guidePage(slug.value))
if (!page.value) {
  throw createError({ statusCode: 404, statusMessage: 'No such guide page', fatal: true })
}

const source = computed(() => guideSource(slug.value))

// Sections and subsections, but not the h1 (which is the page) or the h4s
// (which are details inside a subsection and would double the list's length).
const headings = computed(() => {
  const own = extractHeadings(source.value, { min: 2, max: 3 })
  // The reference builds its own headings from the option registry, and the
  // one anchor worth offering is the top of it.
  if (slug.value === 'reference') return [{ level: 2, text: 'Option reference', id: 'reference' }]
  return own
})

const at = computed(() => GUIDE_PAGES.findIndex((p) => p.slug === slug.value))
const prev = computed(() => GUIDE_PAGES[at.value - 1] || null)
const next = computed(() => GUIDE_PAGES[at.value + 1] || null)

useHead(() => ({
  title: `${page.value?.title || 'Guide'} · Nexstrata guide`,
  meta: [{ name: 'description', content: page.value?.blurb || '' }],
}))

/**
 * Send a hash this page does not own to the page that does.
 *
 * Every control in the app links to /guide#opt-<id>, the home page links to two
 * old section anchors, and readers have bookmarked the rest. Splitting the
 * document would have broken all of them; this is what keeps them landing on
 * the words they named rather than on whichever page inherited the URL.
 */
function settleHash(hash) {
  if (!import.meta.client) return
  const id = String(hash || '').replace(/^#/, '')
  if (!id) return
  if (document.getElementById(id)) {
    nextTick(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    return
  }
  const target = findGuideAnchor(id)
  // Nothing claims it: leave the reader at the top of the page they asked for
  // rather than sending them somewhere invented.
  if (!target || target.slug === slug.value) return
  router.replace(`${guidePath(target.slug)}${target.anchor ? `#${target.anchor}` : ''}`)
}

watch(() => `${route.path}${route.hash}`, () => settleHash(route.hash), { immediate: true })

// After a redirect the new page mounts with the hash already in the URL, and no
// hash change follows, so the watcher above does not fire for it.
onMounted(() => nextTick(() => settleHash(route.hash)))
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

/* Inside the nav's slot, so these match the generated section links. */
.gp-link {
  display: block; padding: 4px 8px; border-radius: 5px;
  color: var(--muted); text-decoration: none; font-size: 0.82rem; line-height: 1.35;
  border-left: 2px solid transparent;
}
.gp-link:hover { color: var(--text); background: var(--surface-2); }
.gp-link.here {
  color: var(--text-strong); font-weight: 700;
  border-left-color: var(--border); background: var(--surface-2);
}
.gp-rule { border-top: 1px solid var(--border); margin: 10px 0 12px; }

/* MarkdownDoc centres itself in its own column; inside this grid that column IS
   the text column, so the extra centring only adds a second margin. */
.guide :deep(.legal) { margin: 0; padding: 8px 0 24px; max-width: 780px; }
.ref-block { max-width: 780px; margin: 0; padding: 0 0 24px; }

.guide-more {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px; max-width: 780px; margin: 8px 0 64px;
}
.gm {
  display: grid; gap: 2px; padding: 12px 14px; text-decoration: none;
  border: 1px solid var(--border); border-radius: 8px; background: var(--surface-2);
}
.gm:hover { border-color: var(--accent); }
.gm.next { text-align: right; }
.gm-dir { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.gm-title { color: var(--text-strong); font-weight: 700; font-size: 0.95rem; }
.gm-blurb { color: var(--muted); font-size: 0.8rem; line-height: 1.45; }

@media (max-width: 940px) {
  .guide-shell { display: block; padding: 0; max-width: none; }
  .guide :deep(.legal) { padding: 20px 18px 24px; margin: 0 auto; }
  .ref-block { margin: 0 auto; padding: 0 18px 24px; }
  .gp-link { padding: 8px; font-size: 0.88rem; }
  .guide-more { margin: 8px 18px 56px; }
}
</style>
