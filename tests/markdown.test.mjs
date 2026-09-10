import test from 'node:test'
import assert from 'node:assert/strict'

import { readFileSync } from 'node:fs'

import { extractHeadings, renderInline, renderMarkdown, slugify } from '../composables/useMarkdown.js'

test('headings, paragraphs, lists and rules still render', () => {
  const html = renderMarkdown('# Title\n\nSome text.\n\n- one\n- two\n\n---')
  assert.match(html, /<h1 id="title">Title<\/h1>/)
  assert.match(html, /<p>Some text\.<\/p>/)
  assert.match(html, /<ul><li>one<\/li><li>two<\/li><\/ul>/)
  assert.match(html, /<hr>/)
})

test('a pipe table becomes a table', () => {
  const html = renderMarkdown('| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |')
  assert.match(html, /<table>/)
  assert.match(html, /<th>A<\/th><th>B<\/th>/)
  assert.match(html, /<td>1<\/td><td>2<\/td>/)
  assert.match(html, /<td>3<\/td><td>4<\/td>/)
})

test('text after a table is not swallowed by it', () => {
  const html = renderMarkdown('| A |\n| --- |\n| 1 |\n\nAfter.')
  assert.match(html, /<\/table>/)
  assert.match(html, /<p>After\.<\/p>/)
})

test('pipes without a rule stay a paragraph', () => {
  // Otherwise any sentence containing a pipe would silently become a table.
  const html = renderMarkdown('| not | a table |')
  assert.ok(!html.includes('<table>'))
  assert.match(html, /<p>/)
})

test('inline code renders and is not re-read as emphasis', () => {
  const html = renderMarkdown('Set `wind_u` and `**not bold**`.')
  assert.match(html, /<code>wind_u<\/code>/)
  // The asterisks inside backticks must survive literally.
  assert.match(html, /<code>\*\*not bold\*\*<\/code>/)
  assert.ok(!html.includes('<strong>not bold</strong>'))
})

test('code works inside a table cell', () => {
  const html = renderMarkdown('| Col |\n| --- |\n| `ndvi` |')
  assert.match(html, /<td><code>ndvi<\/code><\/td>/)
})

test('emphasis, bold and links still work', () => {
  const html = renderMarkdown('**bold**, *italic*, [link](https://example.com)')
  assert.match(html, /<strong>bold<\/strong>/)
  assert.match(html, /<em>italic<\/em>/)
  assert.match(html, /<a href="https:\/\/example\.com">link<\/a>/)
})

test('HTML in the source is escaped, including inside tables and code', () => {
  // The renderer is for our own content, but escaping is the property that
  // keeps it from becoming an injection route if that ever changes.
  const html = renderMarkdown('<script>alert(1)</script>\n\n| <b>x</b> |\n| --- |\n| `<i>y</i>` |')
  assert.ok(!html.includes('<script>'))
  assert.ok(!html.includes('<b>x</b>'))
  assert.ok(!html.includes('<i>y</i>'))
  assert.match(html, /&lt;script&gt;/)
})

test('empty and malformed input does not throw', () => {
  assert.equal(renderMarkdown(''), '')
  assert.equal(renderMarkdown(null), '')
  assert.equal(renderMarkdown(undefined), '')
  // A table header with no body rows is still a valid table.
  assert.match(renderMarkdown('| A |\n| --- |'), /<table>/)
})

test('headings carry an anchor id, so a tooltip can link to a section', () => {
  const html = renderMarkdown('## Map\n\n### Colouring and sizing')
  assert.match(html, /<h2 id="map">Map<\/h2>/)
  assert.match(html, /<h3 id="colouring-and-sizing">/)
})

test('slugs ignore inline markup, so styled headings still anchor', () => {
  assert.equal(slugify('**Bold** heading'), 'bold-heading')
  assert.equal(slugify('`code` in a heading'), 'code-in-a-heading')
  assert.equal(slugify('A [link](/x) here'), 'a-link-here')
  // Punctuation and spacing collapse rather than leaking into the id.
  assert.equal(slugify('Wind / aspect vectors'), 'wind-aspect-vectors')
  assert.equal(slugify('  Trailing —  '), 'trailing')
  assert.equal(slugify('!!!'), '')
})

test('a heading that slugs to nothing is still rendered, just unanchored', () => {
  assert.equal(renderMarkdown('# !!!'), '<h1>!!!</h1>')
})

test('renderInline applies the same escaping as the block renderer', () => {
  assert.equal(renderInline('<script>'), '&lt;script&gt;')
  assert.equal(renderInline('**bold** and `code`'), '<strong>bold</strong> and <code>code</code>')
})

// ── Contents extraction ──────────────────────────────────────────────────────

test('headings come back with the same ids the renderer gives them', () => {
  // The whole point: a contents entry that does not match its anchor scrolls
  // nowhere, which is quiet and infuriating.
  const src = '# Page\n\n## Map\n\n### Colour and size\n\n## Where the data comes from\n'
  const heads = extractHeadings(src)
  const html = renderMarkdown(src)
  for (const h of heads) {
    assert.ok(html.includes(`id="${h.id}"`), `no anchor rendered for ${h.text}`)
  }
})

test('only the requested heading levels are listed', () => {
  const src = '# Page\n\n## Two\n\n### Three\n\n#### Four\n'
  assert.deepEqual(extractHeadings(src, { min: 2, max: 3 }).map((h) => h.text), ['Two', 'Three'])
  assert.deepEqual(extractHeadings(src, { min: 2, max: 2 }).map((h) => h.text), ['Two'])
  // The h1 is the page itself, never a section of it.
  assert.ok(!extractHeadings(src).some((h) => h.text === 'Page'))
})

test('inline markup is stripped from a contents entry', () => {
  const heads = extractHeadings('## The **Points** control\n\n## A `code` heading\n')
  assert.deepEqual(heads.map((h) => h.text), ['The Points control', 'A code heading'])
})

test('a heading inside a fenced block is not a section', () => {
  // Otherwise "# comment" in an example would appear in the sidebar.
  const src = '## Real\n\n```\n# not a heading\n## also not\n```\n\n## Also real\n'
  assert.deepEqual(extractHeadings(src).map((h) => h.text), ['Real', 'Also real'])
})

test('extraction survives an empty or absent document', () => {
  assert.deepEqual(extractHeadings(''), [])
  assert.deepEqual(extractHeadings(null), [])
  assert.deepEqual(extractHeadings('just a paragraph'), [])
})

// ── Callouts ────────────────────────────────────────────────────────────────

test('a labelled blockquote becomes a callout of that kind', () => {
  const html = renderMarkdown('> **Caution** Layers are not heatmaps.')
  assert.match(html, /blockquote class="callout callout-caution"/)
  assert.match(html, /callout-label">Caution</)
  assert.match(html, /Layers are not heatmaps\./)
})

test('consecutive quote lines are one callout', () => {
  // A caveat worth setting apart is usually longer than one line.
  const html = renderMarkdown('> **Note** First line\n> and its continuation.')
  assert.equal((html.match(/<blockquote/g) || []).length, 1)
  assert.match(html, /First line and its continuation\./)
})

test('an unlabelled blockquote is still a callout, just untyped', () => {
  const html = renderMarkdown('> Plain aside.')
  assert.match(html, /blockquote class="callout"/)
  assert.ok(!/callout-label/.test(html))
})

test('a callout ends where the quoting ends', () => {
  const html = renderMarkdown('> **Note** Inside.\n\nOutside.')
  assert.match(html, /<blockquote[^>]*>.*Inside\..*<\/blockquote>/s)
  assert.match(html, /<p>Outside\.<\/p>/)
})

test('the shipped guide renders its callouts and lists every section', () => {
  // The real document, since a renderer that passes on fixtures and fails on
  // the one file it exists for is not much use.
  const src = readFileSync(new URL('../content/guide.md', import.meta.url), 'utf8')
  const html = renderMarkdown(src)
  const heads = extractHeadings(src)
  assert.ok(heads.length > 15, `only ${heads.length} sections found`)
  assert.ok(html.includes('callout-caution'), 'no caution callouts rendered')
  assert.ok(html.includes('callout-note'), 'no note callouts rendered')
  for (const h of heads) assert.ok(html.includes(`id="${h.id}"`), `${h.text} has no anchor`)
})
