import test from 'node:test'
import assert from 'node:assert/strict'

import { renderMarkdown } from '../composables/useMarkdown.js'

test('headings, paragraphs, lists and rules still render', () => {
  const html = renderMarkdown('# Title\n\nSome text.\n\n- one\n- two\n\n---')
  assert.match(html, /<h1>Title<\/h1>/)
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
