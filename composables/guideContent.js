// The guide's Markdown, and an index of every anchor in it.
//
// Separate from guidePages.js because of the ?raw imports: those are a bundler
// feature, so a plain Node test cannot load this file. The manifest stays
// loadable, this does not, and the tests read the Markdown off disk instead.
//
// The index exists for one job. A link into the guide can be older than the
// split, or can point at a section of a page the reader is not on, and either
// way it has to land on the right words rather than on a page that happens to
// share the URL prefix.

import { GUIDE_PAGES, LEGACY_ANCHORS, splitTarget } from '~/composables/guidePages'
import { extractHeadings } from '~/composables/useMarkdown'

import indexMd from '~/content/guide/index.md?raw'
import mapMd from '~/content/guide/map.md?raw'
import dataMd from '~/content/guide/data.md?raw'
import chartsMd from '~/content/guide/charts.md?raw'
import analysisMd from '~/content/guide/analysis.md?raw'
import jobsMd from '~/content/guide/jobs.md?raw'
import layersMd from '~/content/guide/layers.md?raw'
import learningMd from '~/content/guide/learning.md?raw'
import sourcesMd from '~/content/guide/sources.md?raw'

/** slug → Markdown. The reference page has no entry; it is generated. */
export const GUIDE_SOURCES = {
  '': indexMd,
  map: mapMd,
  data: dataMd,
  charts: chartsMd,
  analysis: analysisMd,
  jobs: jobsMd,
  layers: layersMd,
  learning: learningMd,
  sources: sourcesMd,
}

/** The Markdown of a page, or '' for the generated reference. */
export function guideSource(slug) {
  return GUIDE_SOURCES[String(slug || '')] || ''
}

/**
 * anchor → slug, for every heading in every page.
 *
 * Levels 1 to 4, not the 2 and 3 the contents list shows: a link can point at
 * an h4, and this decides where a link goes rather than what the sidebar lists.
 */
const ANCHOR_INDEX = (() => {
  const index = new Map()
  for (const page of GUIDE_PAGES) {
    if (page.file === null) continue
    for (const h of extractHeadings(guideSource(page.slug), { min: 1, max: 4 })) {
      // First page wins. Two pages can hold a "Species" heading, and sending a
      // bare #species somewhere stable beats sending it somewhere alphabetical.
      if (!index.has(h.id)) index.set(h.id, page.slug)
    }
  }
  return index
})()

/**
 * Where an anchor lives, as `{ slug, anchor }`, or null when nothing claims it.
 *
 * Checked in order of confidence: the reference's own namespace, then a heading
 * that exists today, then the record of what the single-page guide used to call
 * things. A heading beats a legacy entry, so re-adding a section under its old
 * name silently retires the redirect instead of fighting it.
 */
export function findGuideAnchor(id) {
  const anchor = String(id || '').replace(/^#/, '')
  if (!anchor) return null
  if (anchor === 'reference' || anchor.startsWith('opt-')) {
    return { slug: 'reference', anchor }
  }
  if (ANCHOR_INDEX.has(anchor)) return { slug: ANCHOR_INDEX.get(anchor), anchor }
  if (LEGACY_ANCHORS[anchor]) return splitTarget(LEGACY_ANCHORS[anchor])
  return null
}
