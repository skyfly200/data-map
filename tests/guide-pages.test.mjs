/* The guide, after it was split into pages.
 *
 * Splitting a document is cheap and breaking its links is silent, so this is
 * mostly about the links: every anchor the single-page guide published, every
 * internal cross-reference the new pages added, and every figure they call for.
 *
 * The manifest is imported and the Markdown is read off disk, which is the only
 * way to check the two agree — a page listed with no file, or a file no page
 * lists, is a page that either 404s or cannot be reached at all.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { readFileSync, readdirSync } from 'node:fs'

import {
  GUIDE_PAGES, LEGACY_ANCHORS, guidePage, guidePath, guidePaths, splitTarget,
} from '../composables/guidePages.js'
import { FIGURES } from '../composables/docFigures.js'
import { extractHeadings, renderMarkdown } from '../composables/useMarkdown.js'
import { OPTION_DOCS, docAnchor, docHref } from '../composables/optionDocs.js'

const dir = new URL('../content/guide/', import.meta.url)
const read = (file) => readFileSync(new URL(file, dir), 'utf8')

/** Every anchor on every page: id → slug. */
const anchors = new Map()
for (const page of GUIDE_PAGES) {
  if (page.file === null) continue
  for (const h of extractHeadings(read(page.file), { min: 1, max: 4 })) {
    if (!anchors.has(h.id)) anchors.set(h.id, page.slug)
  }
}

// ── The manifest and the files ───────────────────────────────────────────────

test('every page in the manifest has its file, and every file has its page', () => {
  const onDisk = readdirSync(dir).filter((f) => f.endsWith('.md')).sort()
  const listed = GUIDE_PAGES.map((p) => p.file).filter(Boolean).sort()
  assert.deepEqual(listed, onDisk)
})

test('slugs are unique, and exactly one page is the index', () => {
  const slugs = GUIDE_PAGES.map((p) => p.slug)
  assert.equal(new Set(slugs).size, slugs.length)
  assert.equal(slugs.filter((s) => s === '').length, 1)
  assert.equal(slugs[0], '', 'the index is not first, so prev/next start in the middle')
})

test('each page has a title and a blurb, since both are rendered', () => {
  for (const p of GUIDE_PAGES) {
    assert.ok(p.title, `${p.slug} has no title`)
    assert.ok(p.blurb, `${p.slug} has no blurb`)
  }
})

test('a URL is built for each page, and an unknown slug is not a page', () => {
  assert.equal(guidePath(''), '/guide')
  assert.equal(guidePath('map'), '/guide/map')
  assert.equal(guidePaths().length, GUIDE_PAGES.length)
  assert.ok(guidePage('map'))
  assert.equal(guidePage('nope'), null)
  // The route passes '' for /guide itself, and undefined before the param
  // resolves. Both have to find the index rather than 404.
  assert.ok(guidePage(''))
  assert.ok(guidePage(undefined))
})

// ── Old links ────────────────────────────────────────────────────────────────

test('every anchor the single-page guide published still resolves', () => {
  for (const [old, target] of Object.entries(LEGACY_ANCHORS)) {
    const { slug, anchor } = splitTarget(target)
    assert.ok(guidePage(slug), `#${old} points at page "${slug}", which does not exist`)
    if (!anchor) continue
    assert.ok(anchors.has(anchor) || slug === 'reference',
      `#${old} points at #${anchor}, which is not a heading anywhere`)
    if (anchors.has(anchor)) {
      assert.equal(anchors.get(anchor), slug,
        `#${old} sends the reader to /${slug} but #${anchor} is on /${anchors.get(anchor)}`)
    }
  }
})

test('the anchors the rest of the app links to are among them', () => {
  // These are hardcoded in pages/index.vue and in the option registry, so they
  // are the ones that break loudest.
  for (const id of ['your-own-earth-engine-layers', 'where-the-data-comes-from', 'reference']) {
    assert.ok(LEGACY_ANCHORS[id] || anchors.has(id), `nothing claims #${id}`)
  }
})

test('every option links into the reference page', () => {
  for (const doc of OPTION_DOCS) {
    assert.equal(docHref(doc.id), `/guide/reference#${docAnchor(doc.id)}`)
  }
})

// ── New links ────────────────────────────────────────────────────────────────

test('every cross-reference between guide pages lands on something', () => {
  const bad = []
  for (const page of GUIDE_PAGES) {
    if (page.file === null) continue
    const src = read(page.file)
    for (const [, href] of src.matchAll(/\]\((\/guide[^)\s]*)\)/g)) {
      const [path, anchor = ''] = href.split('#')
      const slug = path.replace(/^\/guide\/?/, '')
      if (!guidePage(slug)) { bad.push(`${page.file}: no page ${href}`); continue }
      if (!anchor) continue
      if (!anchors.has(anchor)) { bad.push(`${page.file}: no anchor ${href}`); continue }
      if (anchors.get(anchor) !== slug) {
        bad.push(`${page.file}: ${href} — #${anchor} is on /${anchors.get(anchor)}`)
      }
    }
  }
  assert.deepEqual(bad, [])
})

test('every figure a page asks for has a drawing, and actually draws', () => {
  const asked = new Set()
  for (const page of GUIDE_PAGES) {
    if (page.file === null) continue
    const src = read(page.file)
    const html = renderMarkdown(src, { figures: FIGURES })
    for (const [, key] of src.matchAll(/\]\(figure:([a-z0-9-]+)\)/g)) {
      assert.ok(FIGURES[key], `${page.file} asks for figure "${key}", which does not exist`)
      asked.add(key)
    }
    // A figure is a block, so its whole reference has to be on one line. A
    // caption wrapped onto a second line reads as an inline link instead, which
    // renders a literal "!" and a dead href rather than the drawing — and every
    // check on the source alone passes while the page is wrong.
    assert.equal(
      (html.match(/<figure class="doc-figure">/g) || []).length,
      (src.match(/\]\(figure:/g) || []).length,
      `${page.file}: a figure reference did not render as a figure (wrapped caption?)`,
    )
    assert.ok(!/href="figure:/.test(html), `${page.file}: a figure rendered as a link`)
  }
  // And nothing is drawn that no page shows, which would be dead weight in the
  // bundle rather than a broken page.
  for (const key of Object.keys(FIGURES)) {
    assert.ok(asked.has(key), `figure "${key}" is drawn but never used`)
  }
})

test('each drawing is self-contained SVG with a text alternative', () => {
  for (const [key, svg] of Object.entries(FIGURES)) {
    assert.match(svg, /^<svg /, `${key} does not start with <svg`)
    assert.match(svg, /viewBox="/, `${key} has no viewBox, so it cannot scale`)
    assert.match(svg, /aria-label="/, `${key} has no label for a screen reader`)
    // Colours live in the stylesheet, so the drawing is right in both themes.
    assert.ok(!/fill="#|stroke="#/.test(svg), `${key} hardcodes a colour`)
  }
})

// ── The pages themselves ─────────────────────────────────────────────────────

test('every page renders, with a heading list and no duplicate anchors', () => {
  for (const page of GUIDE_PAGES) {
    if (page.file === null) continue
    const src = read(page.file)
    const html = renderMarkdown(src, { figures: FIGURES })
    const heads = extractHeadings(src, { min: 1, max: 4 })
    assert.ok(heads.length >= 2, `${page.file} has ${heads.length} headings`)
    assert.equal(heads[0].level, 1, `${page.file} does not start with a title`)
    const ids = heads.map((h) => h.id)
    assert.equal(new Set(ids).size, ids.length, `${page.file} has a duplicate anchor`)
    for (const h of heads) {
      assert.ok(html.includes(`id="${h.id}"`), `${page.file}: ${h.text} has no anchor`)
    }
  }
})

test('the guide still uses its callouts, on the pages that carry the warnings', () => {
  const html = renderMarkdown(read('map.md'), { figures: FIGURES })
  assert.ok(html.includes('callout-caution'), 'the map page lost its cautions')
  assert.ok(html.includes('callout-note'), 'the map page lost its notes')
})

test('external links are absolute and reach a named host', () => {
  // A bare "www." or a relative path in the learning page's reading lists would
  // resolve against /guide/ and 404 quietly.
  for (const page of GUIDE_PAGES) {
    if (page.file === null) continue
    for (const [, href] of read(page.file).matchAll(/\]\(([^)\s]+)\)/g)) {
      if (href.startsWith('/') || href.startsWith('#') || href.startsWith('figure:')) continue
      assert.match(href, /^https:\/\/[a-z0-9.-]+\//i, `${page.file}: odd link ${href}`)
    }
  }
})
