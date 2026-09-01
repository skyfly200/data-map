// Save a chart or the map as an image.
//
// Two different problems wearing one button:
//
//   Charts are SVG. Serialising one loses every style, because the rules live in
//   a scoped stylesheet the detached copy never sees — so the computed styles
//   have to be baked onto the elements first. PNG and SVG are both offered:
//   SVG stays sharp at any size and is the better thing to put in a document.
//
//   The map is a stack of DOM layers — tile <img>s and one or more <canvas>
//   panes — that Leaflet positions with CSS transforms. Rather than recomputing
//   those transforms, each layer is measured against the map container with
//   getBoundingClientRect and drawn where it actually appears. Iterating in
//   document order reproduces Leaflet's own pane stacking for free.

// Presentation properties that carry a chart's appearance. Copying all computed
// styles would bloat the file and drag in irrelevant layout properties.
const SVG_STYLE_PROPS = [
  'fill', 'fill-opacity', 'fill-rule',
  'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray',
  'stroke-linecap', 'stroke-linejoin',
  'opacity', 'font-family', 'font-size', 'font-weight', 'font-style',
  'text-anchor', 'dominant-baseline', 'letter-spacing', 'visibility', 'display',
]

function inlineStyles(source, clone) {
  const from = source.querySelectorAll('*')
  const to = clone.querySelectorAll('*')
  for (let i = 0; i < from.length; i++) {
    const computed = getComputedStyle(from[i])
    let css = ''
    for (const prop of SVG_STYLE_PROPS) {
      const value = computed.getPropertyValue(prop)
      if (value) css += `${prop}:${value};`
    }
    to[i].setAttribute('style', css)
    // A drop-shadow filter references a stylesheet the clone will not have.
    to[i].removeAttribute('filter')
  }
  const rootStyle = getComputedStyle(source)
  let rootCss = ''
  for (const prop of SVG_STYLE_PROPS) {
    const value = rootStyle.getPropertyValue(prop)
    if (value) rootCss += `${prop}:${value};`
  }
  clone.setAttribute('style', rootCss)
}

export function useImageExport() {
  /** Trigger a download for a blob. */
  function download(blob, filename) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Revoke on the next tick; revoking synchronously can cancel the download
    // in some browsers.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  /**
   * A standalone, self-contained copy of an on-screen <svg>, with its styles
   * baked in and an explicit size and background.
   */
  function serializeSvg(svg, { background = '#ffffff' } = {}) {
    const rect = svg.getBoundingClientRect()
    const width = Math.max(1, Math.round(rect.width))
    const height = Math.max(1, Math.round(rect.height))

    const clone = svg.cloneNode(true)
    inlineStyles(svg, clone)
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
    clone.setAttribute('width', width)
    clone.setAttribute('height', height)
    if (!clone.getAttribute('viewBox')) {
      clone.setAttribute('viewBox', `0 0 ${width} ${height}`)
    }
    if (background) {
      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      bg.setAttribute('width', '100%')
      bg.setAttribute('height', '100%')
      bg.setAttribute('fill', background)
      clone.insertBefore(bg, clone.firstChild)
    }
    return { markup: new XMLSerializer().serializeToString(clone), width, height }
  }

  function svgBlob(svg, options) {
    const { markup } = serializeSvg(svg, options)
    return new Blob([markup], { type: 'image/svg+xml;charset=utf-8' })
  }

  /** Rasterise an on-screen <svg> to a PNG blob at `scale`x. */
  async function svgToPng(svg, { scale = 2, background = '#ffffff' } = {}) {
    const { markup, width, height } = serializeSvg(svg, { background })
    const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = () => reject(new Error('Could not rasterise the chart'))
      img.src = url
    })
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(width * scale)
    canvas.height = Math.round(height * scale)
    const ctx = canvas.getContext('2d')
    if (background) {
      ctx.fillStyle = background
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    return new Promise((resolve, reject) => {
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the image'))), 'image/png')
    })
  }

  /**
   * Flatten a Leaflet map container into a PNG.
   *
   * Tiles must be served with CORS headers and requested with crossOrigin set,
   * or the canvas is tainted and toBlob throws — that case is reported rather
   * than failing silently, since it depends on the basemap in use.
   */
  async function mapToPng(container, { scale = 2, background = '#ffffff' } = {}) {
    const mapRect = container.getBoundingClientRect()
    const width = Math.max(1, Math.round(mapRect.width))
    const height = Math.max(1, Math.round(mapRect.height))

    const canvas = document.createElement('canvas')
    canvas.width = Math.round(width * scale)
    canvas.height = Math.round(height * scale)
    const ctx = canvas.getContext('2d')
    ctx.scale(scale, scale)
    ctx.fillStyle = background
    ctx.fillRect(0, 0, width, height)

    // Document order == Leaflet's pane order (tiles, then overlays, then
    // markers), so drawing in that order reproduces the on-screen stacking.
    const layers = container.querySelectorAll('.leaflet-pane img, .leaflet-pane canvas, .leaflet-pane svg')
    for (const el of layers) {
      const r = el.getBoundingClientRect()
      if (!r.width || !r.height) continue
      const x = r.left - mapRect.left
      const y = r.top - mapRect.top
      try {
        if (el.tagName === 'SVG' || el.tagName === 'svg') {
          const { markup } = serializeSvg(el, { background: null })
          const img = new Image()
          await new Promise((res) => { img.onload = res; img.onerror = res; img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}` })
          ctx.drawImage(img, x, y, r.width, r.height)
        } else if (el.tagName === 'IMG') {
          if (!el.complete || !el.naturalWidth) continue   // still loading
          ctx.drawImage(el, x, y, r.width, r.height)
        } else {
          ctx.drawImage(el, x, y, r.width, r.height)
        }
      } catch {
        // One unusable layer should not lose the whole export.
      }
    }

    return new Promise((resolve, reject) => {
      try {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the image'))), 'image/png')
      } catch {
        reject(new Error('The basemap blocked image export (its tiles disallow cross-origin reads). '
                       + 'Switch basemap and try again.'))
      }
    })
  }

  /** A filesystem-safe filename stem from a chart title or view name. */
  function slugify(name, fallback = 'export') {
    const slug = String(name || '').toLowerCase()
      .replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-').slice(0, 60)
    return slug || fallback
  }

  const stamp = () => new Date().toISOString().slice(0, 10)

  return { download, serializeSvg, svgBlob, svgToPng, mapToPng, slugify, stamp }
}
