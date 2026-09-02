// Tiny, dependency-free Markdown → HTML renderer for the app's own static
// documents (the feature guide, privacy policy, terms). It supports only the
// subset those files use — headings, paragraphs, unordered lists, tables,
// links, bold, italic, inline code, and horizontal rules — and is NOT a
// general-purpose or safe renderer for untrusted input. Source text is
// HTML-escaped first, so our controlled Markdown renders literally.

/**
 * A heading's anchor id. Headings get one so the guide can be linked to a
 * section rather than to the top of a long page — which is what the ? beside
 * every control does.
 */
export function slugify(text) {
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

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/**
 * Inline markup only — bold, italic, code and links — with no block structure.
 * The option reference renders its paragraphs through this, so the same escaping
 * and the same small Markdown subset apply there as in the guide itself.
 */
export function renderInline(text) {
  return inline(text)
}

function inline(text) {
  let s = escapeHtml(text)
  // Code first: whatever is inside backticks must not then be read as emphasis.
  const code = []
  s = s.replace(/`([^`]+)`/g, (_, body) => `\u0000${code.push(body) - 1}\u0000`)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => `<a href="${url}">${label}</a>`)
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${code[Number(i)]}</code>`)
  return s
}

const isTableRow = (line) => line.startsWith('|') && line.endsWith('|')
// The |---|---| line that separates a table's header from its body.
const isTableRule = (line) => /^\|[\s|:-]+\|$/.test(line) && line.includes('-')

function tableCells(line) {
  return line.slice(1, -1).split('|').map((c) => c.trim())
}

export function renderMarkdown(md) {
  const lines = String(md || '').replace(/\r\n/g, '\n').split('\n')
  const out = []
  let para = []
  let list = []
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = [] } }
  const flushList = () => { if (list.length) { out.push(`<ul>${list.map((li) => `<li>${inline(li)}</li>`).join('')}</ul>`); list = [] } }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) { flushPara(); flushList(); continue }

    // A pipe table: a header row, a |---| rule, then body rows until the block
    // ends. Without the rule it is just a paragraph containing pipes.
    if (isTableRow(line) && isTableRule((lines[i + 1] || '').trim())) {
      flushPara(); flushList()
      const head = tableCells(line)
      const body = []
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

    let m
    if ((m = /^(#{1,4})\s+(.*)$/.exec(line))) {
      flushPara(); flushList()
      const level = m[1].length
      const id = slugify(m[2])
      out.push(`<h${level}${id ? ` id="${id}"` : ''}>${inline(m[2])}</h${level}>`)
    } else if (/^---+$/.test(line)) {
      flushPara(); flushList()
      out.push('<hr>')
    } else if ((m = /^[-*]\s+(.*)$/.exec(line))) {
      flushPara()
      list.push(m[1])
    } else {
      flushList()
      para.push(line)
    }
  }
  flushPara(); flushList()
  return out.join('\n')
}
