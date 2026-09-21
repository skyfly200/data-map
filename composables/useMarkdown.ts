// Tiny, dependency-free Markdown → HTML renderer for the app's own static
// documents (the feature guide, privacy policy, terms). It supports only the
// subset those files use — headings, paragraphs, unordered lists, tables,
// links, bold, italic, inline code, and horizontal rules — and is NOT a
// general-purpose or safe renderer for untrusted input. Source text is
// HTML-escaped first, so our controlled Markdown renders literally.

export interface Heading {
  level: number
  text: string
  id: string
}

export interface HeadingOptions {
  min?: number
  max?: number
}

/**
 * A heading's anchor id. Headings get one so the guide can be linked to a
 * section rather than to the top of a long page — which is what the ? beside
// every control does.
 */
export function slugify(text: string): string {
  return String(text)
    .toLowerCase()
    // Drop the inline markup before slugging, so "**Bold** heading" and "Bold
    // heading" reach the same anchor.
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*+/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/**
 * The headings of a document, for building a contents list beside it.
 *
 * Reads the same lines the renderer does and slugs them the same way, so a
// contents entry and the anchor it points at cannot drift apart — the failure
// would be a link that scrolls nowhere, which is quiet and annoying.
 *
 * Fenced code is skipped: a `# comment` inside one is not a heading, and
// treating it as a section would put nonsense in the sidebar.
 */
export function extractHeadings(source: string | null | undefined, options: HeadingOptions = { min: 2, max: 3 }): Heading[] {
  const out: Heading[] = []
  const unique = uniqueIds()
  let fenced = false
  for (const line of String(source || '').split(/\r?\n/)) {
    if (/^```/.test(line.trim())) { fenced = !fenced; continue }
    if (fenced) continue
    const m = /^(#{1,4})\s+(.*)$/.exec(line)
    if (!m) continue
    // Every heading is counted, including the ones outside [min, max], because
    // the renderer counts them too. Skipping one here would number a later
    // duplicate differently from its own anchor.
    const id = unique(slugify(m[2]))
    const level = m[1].length
    if (level < (options.min ?? 2) || level > (options.max ?? 3)) continue
    const text = m[2]
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\*+/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .trim()
    if (id && text) out.push({ level, text, id })
  }
  return out
}

/**
 * A slug-to-id function that never repeats itself within one document.
 *
 * Two headings with the same words are ordinary in a long document — three
// sections of the learning page could each end "where to read more" — and two
// elements with one id means the second is unreachable and the contents list
// silently sends you to the first.
 */
function uniqueIds() {
  const seen = new Map<string, number>()
  return (base: string) => {
    if (!base) return ''
    const n = (seen.get(base) || 0) + 1
    seen.set(base, n)
    return n === 1 ? base : `${base}-${n}`
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/**
 * Inline markup only — bold, italic, code and links — with no block structure.
 * The option reference renders its paragraphs through this, so the same escaping
// and the same small Markdown subset apply there as in the guide itself.
 */
export function renderInline(text: string): string {
  return inline(text)
}

function inline(text: string): string {
  let s = escapeHtml(text)
  // Code first: whatever is inside backticks must not then be read as emphasis.
  const code: string[] = []
  s = s.replace(/`([^`]+)`/g, (_, body) => `\u0000${code.push(body) - 1}\u0000`)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => `<a href="${url}">${label}</a>`)
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${code[Number(i)]}</code>`)
  return s
}

const isTableRow = (line: string) => line.startsWith('|') && line.endsWith('|')
// The |---|---| line that separates a table's header from its body.
const isTableRule = (line: string) => /^\|[\s|:-]+\|$/.test(line) && line.includes('-')

function tableCells(line: string): string[] {
  return line.slice(1, -1).split('|').map((c) => c.trim())
}

// A figure on its own line: `![caption](figure:key)`. The key names a drawing in
// composables/docFigures, rather than an image file, because these diagrams are
// line art over the app's own colour tokens — an exported PNG would be wrong in
// one of the two themes, and an SVG file could not read the tokens at all.
const FIGURE_LINE = /^!\[([^\]]*)\]\(figure:([a-z0-9-]+)\)$/

/**
 * Markdown → HTML for the app's own documents.
 *
 * `figures` maps a figure key to an SVG string. A key with no drawing renders
// its caption as ordinary text rather than disappearing, a typo is visible
// in the page instead of silently removing a paragraph.
 */
export function renderMarkdown(md: string | null | undefined, { figures = {} } = {}): string {
  const lines = String(md || '').replace(/\r\n/g, '\n').split('\n')
  const out: string[] = []
  const unique = uniqueIds()
  let para: string[] = []
  let list: string[] = []
  // 'ul' or 'ol'. A guide is mostly procedures, and a procedure that renders as
// a paragraph beginning "1. Open the Map. 2. Open Points." is a procedure
// nobody can follow a step at a time.
  let listTag = 'ul'
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = [] } }
  const flushList = () => {
    if (!list.length) return
    out.push(`<${listTag}>${list.map((li) => `<li>${inline(li)}</li>`).join('')}</${listTag}>`)
    list = []
  }
  const pushItem = (tag: string, text: string) => {
    // A bulleted list directly under a numbered one is two lists, not one with
// a confused tag.
    if (list.length && listTag !== tag) flushList()
    listTag = tag
    list.push(text)
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()

    // A fenced code block, taken verbatim. Without this the guide's Earth
    // Engine example rendered as a paragraph of run-together statements, which
    // is exactly the content that has to be copied character for character.
    if (line.startsWith('```')) {
      flushPara(); flushList()
      const codeLines: string[] = []
      i += 1
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i])
        i += 1
      }
      out.push(`<pre class="doc-code"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`)
      continue
    }

    if (!line) { flushPara(); flushList(); continue }

    // A pipe table: a header row, a |---| rule, then body rows until the block
    // ends. Without the rule it is just a paragraph containing pipes.
    if (isTableRow(line) && isTableRule((lines[i + 1] || '').trim())) {
      flushPara(); flushList()
      const head = tableCells(line)
      const body: string[][] = []
      i += 2
      while (i < lines.length && isTableRow(lines[i].trim())) {
        body.push(tableCells(lines[i].trim()))
        i++
      }
      i--
      out.push(
        '<table><thead><tr>'
        + head.map((c) => `<th>${inline(c)}</th>`).join('')
        + '</tr></thead><tbody>'
        + body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')
        + '</tbody></table>',
      )
      continue
    }

    // A blockquote, used for the callouts the guide leans on. Consecutive
    // "> " lines become one, so a callout can run to a few sentences.
    //
    // When the first line starts with a bold word — "> **Note**" — that word
    // becomes the callout's label and the rest its body, which is the shape
// every documentation site uses and the reason this exists: a caveat woven
// into a paragraph reads as commentary, and the same caveat set apart reads
// as something to act on.
    if (/^>\s?/.test(line)) {
      flushPara(); flushList()
      const quoted: string[] = []
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) {
        quoted.push(lines[i].trim().replace(/^>\s?/, ''))
        i++
      }
      i--
      const text = quoted.join(' ').trim()
      const labelled = /^\*\*([^*]+)\*\*[:.]?\s*(.*)$/.exec(text)
      const kind = labelled ? slugify(labelled[1]) : ''
      out.push(labelled
        ? `<blockquote class="callout callout-${kind}">`
          + `<strong class="callout-label">${inline(labelled[1])}</strong> `
          + `<span>${inline(labelled[2])}</span></blockquote>`
        : `<blockquote class="callout">${inline(text)}</blockquote>`)
      continue
    }

    let m
    if ((m = FIGURE_LINE.exec(line))) {
      flushPara(); flushList()
      const svg = figures[m[2]]
      out.push(svg
        ? `<figure class="doc-figure"><div class="doc-figure-art">${svg}</div>`
          + `<figcaption>${inline(m[1])}</figcaption></figure>`
        : `<p>${inline(m[1])}</p>`)
    } else if ((m = /^(#{1,4})\s+(.*)$/.exec(line))) {
      flushPara(); flushList()
      const level = m[1].length
      const id = unique(slugify(m[2]))
      out.push(`<h${level}${id ? ` id="${id}"` : ''}>${inline(m[2])}</h${level}>`)
    } else if (/^---+$/.test(line)) {
      flushPara(); flushList()
      out.push('<hr>')
    } else if ((m = /^[-*]\s+(.*)$/.exec(line))) {
      flushPara()
      pushItem('ul', m[1])
    // "1. " and "1) ". The space is what keeps a sentence starting "1.41 times
    // further" from becoming a step.
    } else if ((m = /^\d+[.)]\s+(.*)$/.exec(line))) {
      flushPara()
      pushItem('ol', m[1])
    } else if (list.length) {
      // A wrapped bullet. Markdown lets a list item run over several lines, and
      // the guide's do: they are full sentences, and a source file with 120
      // character lines is unreadable. Without this the second line escaped the
      // list and became a paragraph below it — the item's own sentence, cut
      // in half and set at a different indent.
      list[list.length - 1] += ` ${line}`
    } else {
      para.push(line)
    }
  }
  flushPara(); flushList()
  return out.join('\n')
}
