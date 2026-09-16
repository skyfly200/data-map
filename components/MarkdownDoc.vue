<template>
  <div class="legal">
    <!-- eslint-disable-next-line vue/no-v-html — source is our own static Markdown -->
    <article v-html="html" />
  </div>
</template>

<script setup>
import { FIGURES } from '~/composables/docFigures'
import { renderMarkdown } from '~/composables/useMarkdown'

const props = defineProps({ source: { type: String, required: true } })
const html = computed(() => renderMarkdown(props.source, { figures: FIGURES }))
</script>

<style scoped>
/* Wide enough for the guide's tables; the legal pages stay comfortably
   within it at their natural measure. */
.legal { max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }
.legal :deep(article) { color: var(--text); line-height: 1.65; }

/* Four levels that are told apart at a glance, not by measuring. The guide runs
   to four deep in places, and at the old sizes an h3 and a bold sentence were
   the same thing — so a reader scrolling for a section had to read rather than
   scan. Size, weight, colour and space above all move together, and the h2 also
   takes a rule, because that is the one boundary worth seeing from the corner
   of the eye. */
.legal :deep(h1) {
  font-size: 1.9rem; font-weight: 700; line-height: 1.2;
  margin: 0 0 18px; color: var(--text-strong); letter-spacing: -0.01em;
}
.legal :deep(h2) {
  font-size: 1.32rem; font-weight: 700; line-height: 1.3;
  margin: 44px 0 14px; padding-bottom: 7px; color: var(--text-strong);
  border-bottom: 1px solid var(--border);
}
.legal :deep(h3) {
  font-size: 1.06rem; font-weight: 700; line-height: 1.35;
  margin: 30px 0 8px; color: var(--text-strong);
}
.legal :deep(h4) {
  font-size: 0.82rem; font-weight: 700; margin: 20px 0 6px;
  text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
}
/* Anchored from a tooltip and from the contents list, so a heading must not
   land under the sticky app header. */
.legal :deep(h1), .legal :deep(h2), .legal :deep(h3), .legal :deep(h4) {
  scroll-margin-top: 76px;
}
.legal :deep(h2 + h3), .legal :deep(h3 + h4) { margin-top: 16px; }

.legal :deep(p) { margin: 0 0 14px; }
.legal :deep(ul), .legal :deep(ol) { margin: 0 0 14px; padding-left: 24px; }
.legal :deep(li) { margin: 6px 0; }
/* A procedure reads as a procedure: the step numbers are the thing you scan
   for, so they are not the same weight as the sentence beside them. */
.legal :deep(ol) { counter-reset: step; list-style: none; padding-left: 0; }
.legal :deep(ol > li) {
  counter-increment: step; position: relative; padding-left: 34px; margin: 10px 0;
}
.legal :deep(ol > li)::before {
  content: counter(step);
  position: absolute; left: 0; top: 1px;
  width: 22px; height: 22px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--surface-2); border: 1px solid var(--border);
  color: var(--text-strong); font-size: 0.75rem; font-weight: 700;
  font-variant-numeric: tabular-nums;
}
/* var(--accent), not a fixed green: the hardcoded color was near-invisible on
   the dark theme, which is the default. */
.legal :deep(a) { color: var(--accent); }
/* Emphasis should stand out, so it keeps the body color rather than being
   dimmed to muted. */
.legal :deep(em) { font-style: italic; }
.legal :deep(strong) { color: var(--text-strong); }
.legal :deep(hr) { border: 0; border-top: 1px solid var(--border); margin: 24px 0; }

/* Callouts. Set apart rather than woven in, because a caveat inside a paragraph
   reads as commentary and the same caveat in a box reads as something to act
   on — which, for most of the ones in the guide, it is. */
.legal :deep(blockquote.callout) {
  margin: 0 0 14px; padding: 10px 14px;
  border-left: 3px solid var(--border);
  background: var(--surface-2); border-radius: 0 6px 6px 0;
  font-size: 0.9rem; line-height: 1.5; color: var(--text);
}
.legal :deep(.callout-label) {
  color: var(--text-strong); font-weight: 700;
}
.legal :deep(.callout-label)::after { content: ''; }
/* Three kinds, by the word the author used. Anything else stays neutral rather
   than picking a color at random. */
.legal :deep(blockquote.callout-note) { border-left-color: var(--accent); }
.legal :deep(blockquote.callout-caution),
.legal :deep(blockquote.callout-warning) { border-left-color: #b3822f; }
.legal :deep(blockquote.callout-caution) .callout-label,
.legal :deep(blockquote.callout-warning) .callout-label { color: #b3822f; }
.legal :deep(blockquote.callout-tip) { border-left-color: #3d8b5f; }
.legal :deep(blockquote.callout-tip) .callout-label { color: #3d8b5f; }

/* Figures. The drawing carries classes and no colours of its own, so the theme
   tokens below are the only place either theme is described. */
.legal :deep(.doc-figure) {
  margin: 18px 0 20px; padding: 14px 14px 10px;
  border: 1px solid var(--border-soft, var(--border)); border-radius: 8px;
  background: var(--surface-2);
}
.legal :deep(.doc-figure-art) { display: block; }
.legal :deep(.doc-figure svg) { display: block; width: 100%; height: auto; }
.legal :deep(.doc-figure figcaption) {
  margin-top: 8px; font-size: 0.82rem; color: var(--muted); line-height: 1.5;
}

.legal :deep(.doc-figure .box) { fill: var(--surface); stroke: var(--border); stroke-width: 1; }
.legal :deep(.doc-figure .box.accent) { stroke: var(--accent); stroke-width: 1.5; }
.legal :deep(.doc-figure .lbl) {
  fill: var(--text-strong); font-size: 12px; font-weight: 600;
  font-family: inherit;
}
.legal :deep(.doc-figure .sub) { fill: var(--muted); font-size: 10.5px; font-family: inherit; }
.legal :deep(.doc-figure .arw) { stroke: var(--muted); stroke-width: 1.4; fill: none; }
.legal :deep(.doc-figure .arw.dash) { stroke-dasharray: 4 4; }
.legal :deep(.doc-figure .arw-head) { fill: var(--muted); stroke: none; }
.legal :deep(.doc-figure .cell) { fill: none; stroke: var(--border); stroke-width: 1.2; }
.legal :deep(.doc-figure .cell.fill) { fill: var(--accent); fill-opacity: 0.35; stroke: var(--accent); }
.legal :deep(.doc-figure .cell.dim) { stroke-dasharray: 3 3; }
.legal :deep(.doc-figure .ramp) { stroke: var(--border); stroke-width: 1; }
.legal :deep(.doc-figure .stop) { fill: var(--surface); stroke: var(--text); stroke-width: 1.2; }
.legal :deep(.doc-figure .dot) { stroke: var(--surface); stroke-width: 1.5; }
.legal :deep(.doc-figure .d1) { fill: #d1603d; }
.legal :deep(.doc-figure .d2) { fill: #3d8b5f; }
.legal :deep(.doc-figure .d3) { fill: #4a6fa5; }
.legal :deep(.doc-figure .rs0) { stop-color: #e8f1fb; }
.legal :deep(.doc-figure .rs1) { stop-color: #7fb3d5; }
.legal :deep(.doc-figure .rs2) { stop-color: #2a6f97; }
.legal :deep(.doc-figure .rs3) { stop-color: #0b3d91; }

.legal :deep(code) {
  font: 0.86em/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--surface-2); border: 1px solid var(--border-soft);
  border-radius: 4px; padding: 1px 5px; white-space: nowrap;
}

/* A block of code is meant to be copied, so it scrolls sideways rather than
   wrapping: a wrapped line changes where the line breaks are, and in a language
   that does not care about that it still misleads the eye about the structure. */
.legal :deep(pre.doc-code) {
  margin: 0 0 16px; padding: 12px 14px; overflow-x: auto;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
}
.legal :deep(pre.doc-code code) {
  display: block; background: none; border: 0; padding: 0;
  white-space: pre; font-size: 0.8rem; line-height: 1.55; color: var(--text);
}

/* Tables can be wider than the column on a narrow screen, so they scroll inside
   their own box rather than pushing the page sideways. */
.legal :deep(table) {
  width: 100%; border-collapse: collapse; margin: 0 0 16px;
  font-size: 0.88rem; display: block; overflow-x: auto;
}
.legal :deep(thead th) {
  text-align: left; color: var(--muted); font-weight: 600;
  padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.legal :deep(tbody td) {
  padding: 6px 10px; border-bottom: 1px solid var(--border-soft); vertical-align: top;
}
.legal :deep(tbody tr:last-child td) { border-bottom: 0; }
</style>
