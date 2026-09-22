// Pure logic for mapping MaxEnt result data to SVG coordinates and styles.
//
// By extracting this from the Vue components, we can unit-test the coordinate
// mapping, scaling, and color interpolation without a browser.
//
// All functions assume a fixed viewport for the base calculation, which
// the components then apply to their specific SVG viewBox.

export interface Point { x: number; y: number }
export interface ScaledPoint extends Point { cx: number; cy: number }

export interface ChartDims {
  W: number
  H: number
  padL: number
  padR: number
  padT: number
  padB: number
}

/** Standard viewport dimensions used across most MaxEnt charts. */
export const DEFAULT_DIMS: ChartDims = {
  W: 640, H: 260, padL: 52, padR: 16, padT: 12, padB: 34
}

// ── Response & ROC Curves ──────────────────────────────────────────────────

/** Maps a value to a pixel coordinate based on a domain and viewport. */
export function mapValue(v: number, dom: [number, number], padStart: number, padEnd: number, total: number): number {
  return padStart + ((v - dom[0]) / (dom[1] - dom[0] || 1)) * (total - padStart - padEnd)
}

/** Scales a data point to a coordinate. */
export function scalePoint(p: Point, xDom: [number, number], yDom: [number, number], dims: ChartDims = DEFAULT_DIMS): ScaledPoint {
  return {
    x: p.x, y: p.y,
    cx: mapValue(p.x, xDom, dims.padL, dims.padR, dims.W),
    cy: (dims.H - dims.padB) - ((p.y - yDom[0]) / (yDom[1] - yDom[0] || 1)) * (dims.H - dims.padT - dims.padB)
  }
}

/** Generates evenly spaced tick marks for an axis. */
export function generateTicks(dom: [number, number], n: number, isX: boolean, dims: ChartDims = DEFAULT_DIMS): { v: number, p: number, label: string }[] {
  const [lo, hi] = dom
  const out = []
  for (let i = 0; i <= n; i++) {
    const v = lo + ((hi - lo) * i) / n
    const p = isX 
      ? mapValue(v, dom, dims.padL, dims.padR, dims.W)
      : (dims.H - dims.padB) - ((v - dom[0]) / (dom[1] - dom[0] || 1)) * (dims.H - dims.padT - dims.padB)
    out.push({ v, p, label: v.toFixed(2) })
  }
  return out
}

// ── Confusion Matrix ──────────────────────────────────────────────────────────

export interface MatrixCell {
  r: string
  c: string
  v: number
}

/** Calculates a color for a matrix cell based on the range of values in the matrix. */
export function matrixCellColor(v: number, lo: number, hi: number): string {
  if (!Number.isFinite(v)) return 'var(--surface-2)'
  const t = (v - lo) / ((hi - lo) || 1)
  const a = [232, 241, 251], b = [11, 61, 145]
  const c = a.map((ch2, k) => Math.round(ch2 + (b[k] - ch2) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}

/** Determines if the text in a cell should be white or dark. */
export function matrixTextColor(v: number, lo: number, hi: number): string {
  if (!Number.isFinite(v)) return 'var(--muted)'
  const t = (v - lo) / ((hi - lo) || 1)
  return t > 0.55 ? '#fff' : 'var(--text)'
}

/** Calculates the coordinates for a cell in the matrix. */
export function matrixCellCoord(rowIdx: number, colIdx: number, rowCount: number, colCount: number, dims: ChartDims): { cx: number, cy: number, cw: number, ch: number } {
  const padL = Math.max(90, 130 - Math.min(28, Math.max(0, rowCount - 5) * 4))
  const padR = 12
  const padT = 26
  const padB = 8
  const ch = 30
  const W = Math.max(640, colCount * 90 + 180)
  const cw = (W - padL - padR) / Math.max(1, colCount)

  return {
    cx: padL + colIdx * cw,
    cy: padT + rowIdx * ch,
    cw, ch
  }
}

// ── Variable Contribution ────────────────────────────────────────────────────

export function scaleContribution(v: number, maxV: number, dims: ChartDims = DEFAULT_DIMS): number {
  return dims.padL + (v / maxV) * (dims.W - dims.padL - dims.padR)
}
