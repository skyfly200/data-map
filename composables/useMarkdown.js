// Tiny, dependency-free Markdown → HTML renderer for the app's own static
// documents (privacy policy, terms). It supports only the subset those files
// use — headings, paragraphs, unordered lists, links, bold, italic, and
// horizontal rules — and is NOT a general-purpose or safe renderer for
// untrusted input. Source text is HTML-escaped first, so our controlled
// Markdown renders literally.

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function inline(text) {
  let s = escapeHtml(text)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => `<a href="${url}">${label}</a>`)
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  return s
}

export function renderMarkdown(md) {
  const lines = String(md || '').replace(/\r\n/g, '\n').split('\n')
  const out = []
  let para = []
  let list = []
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = [] } }
  const flushList = () => { if (list.length) { out.push(`<ul>${list.map((li) => `<li>${inline(li)}</li>`).join('')}</ul>`); list = [] } }

  for (const raw of lines) {
    const line = raw.trim()
    if (!line) { flushPara(); flushList(); continue }

    let m
    if ((m = /^(#{1,4})\s+(.*)$/.exec(line))) {
      flushPara(); flushList()
      const level = m[1].length
      out.push(`<h${level}>${inline(m[2])}</h${level}>`)
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
