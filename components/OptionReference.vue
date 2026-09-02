<template>
  <section class="ref">
    <h2 id="reference">Option reference</h2>
    <p class="ref-intro">
      Every control in the app, what it means, and where it can mislead you. The
      <span class="q">?</span> beside a control links straight to its entry here.
    </p>

    <nav class="ref-nav" aria-label="Reference sections">
      <a v-for="g in groups" :key="g.name" :href="`#group-${slug(g.name)}`">{{ g.name }}</a>
    </nav>

    <section v-for="g in groups" :key="g.name" class="ref-group">
      <h3 :id="`group-${slug(g.name)}`">{{ g.name }}</h3>

      <article v-for="doc in g.items" :id="docAnchor(doc.id)" :key="doc.id" class="opt">
        <h4>
          {{ doc.title }}
          <a class="anchor" :href="`#${docAnchor(doc.id)}`" :aria-label="`Link to ${doc.title}`">#</a>
        </h4>
        <p class="opt-summary">{{ doc.summary }}</p>
        <!-- eslint-disable-next-line vue/no-v-html -- rendered from our own
             static registry, escaped by the same renderer the guide uses. -->
        <p v-for="(para, i) in doc.detail" :key="i" class="opt-detail" v-html="render(para)"></p>
        <p v-if="doc.caveat" class="opt-caveat">
          <strong>Watch out:</strong> <span v-html="render(doc.caveat)"></span>
        </p>
        <p v-if="related(doc).length" class="opt-also">
          See also:
          <a v-for="(r, i) in related(doc)" :key="r.id" :href="`#${docAnchor(r.id)}`">{{ r.title }}<span v-if="i < related(doc).length - 1">, </span></a>
        </p>
      </article>
    </section>
  </section>
</template>

<script setup>
import { docAnchor, docFor, docGroups } from '~/composables/optionDocs'
import { renderInline, slugify } from '~/composables/useMarkdown'

const groups = docGroups()
const slug = slugify
const render = renderInline
const related = (doc) => (doc.also || []).map(docFor).filter(Boolean)
</script>

<style scoped>
.ref { margin-top: 40px; border-top: 1px solid var(--border); padding-top: 8px; }
.ref h2 { font-size: 1.5rem; margin: 24px 0 8px; }
.ref-intro { color: var(--muted); line-height: 1.6; margin: 0 0 18px; }
.q {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; border: 1px solid var(--border); border-radius: 50%;
  font-size: 9px; font-weight: 700;
}

.ref-nav { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 26px; }
.ref-nav a {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 999px; padding: 4px 12px; font-size: 0.8rem; font-weight: 600; text-decoration: none;
}
.ref-nav a:hover { background: var(--surface-3); }

.ref-group { margin-bottom: 30px; }
.ref-group > h3 {
  font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin: 0 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border-soft);
}

.opt {
  margin: 0 0 18px; padding-left: 14px; border-left: 2px solid var(--border-soft);
  /* Anchored from a tooltip, so the heading must not land under a sticky header. */
  scroll-margin-top: 76px;
}
.opt:target { border-left-color: var(--accent); }
.opt h4 { font-size: 1rem; margin: 0 0 4px; color: var(--text); }
.anchor { color: var(--border); text-decoration: none; margin-left: 6px; font-weight: 400; }
.opt h4:hover .anchor { color: var(--muted); }
.opt-summary { margin: 0 0 8px; color: var(--text); font-weight: 600; line-height: 1.5; }
.opt-detail { margin: 0 0 8px; color: var(--muted); line-height: 1.65; }
.opt-caveat {
  margin: 8px 0; padding: 8px 12px; border-radius: 6px; line-height: 1.6;
  background: var(--surface-2); border-left: 3px solid var(--warn, #d08a00); color: var(--text);
  font-size: 0.9rem;
}
.opt-also { margin: 6px 0 0; font-size: 0.82rem; color: var(--muted); }
.opt-also a { color: var(--accent); text-decoration: none; }
.opt-also a:hover { text-decoration: underline; }

@media (max-width: 560px) {
  .opt { padding-left: 10px; }
}
</style>
