// How a drawn layer combines with the layers below it.
//
// Stacked rasters have one problem that opacity does not solve. Two layers at
// 50% is each layer half washed out, and the thing you wanted — the shape of
// the hillshade over the colour of the land cover — is exactly what 50% of each
// destroys. A blend mode keeps both at full strength and combines them by
// value instead: multiply keeps the dark parts of both, screen keeps the light
// parts, and the relief reads through the colour rather than under it.
//
// These are CSS mix-blend-mode values, set on each layer's own container in the
// tile pane. The browser does the compositing, so there is no cost per tile and
// nothing to re-fetch when the mode changes.

export interface BlendMode {
  key: string
  label: string
  note: string
}

/**
 * The modes worth offering, in the order they are useful rather than
 * alphabetically.
 *
 * Not the full CSS list: hue, saturation and color operate on a colour wheel
 * and produce results nobody can predict from a raster, and the separable
// modes below cover what a map stack actually needs.
 */
export const BLEND_MODES: BlendMode[] = [
  { key: 'normal', label: 'Normal', note: 'Draw over, nothing combined.' },
  { key: 'multiply', label: 'Multiply', note: 'Keeps what is dark in both. Relief over colour.' },
  { key: 'screen', label: 'Screen', note: 'Keeps what is light in both. Burn scars over terrain.' },
  { key: 'overlay', label: 'Overlay', note: 'Multiply the darks, screen the lights. More contrast.' },
  { key: 'darken', label: 'Darken', note: 'The darker of the two, channel by channel.' },
  { key: 'lighten', label: 'Lighten', note: 'The lighter of the two, channel by channel.' },
  { key: 'difference', label: 'Difference', note: 'What the two disagree and opposite.' },
  { key: 'luminosity', label: 'Luminosity', note: "This layer's brightness, the colour below." },
]

const KEYS = BLEND_MODES.map((m) => m.key)

/** The default, and what an unusable value falls back to. */
export const NORMAL = 'normal'

/** Whether a mode is one this app draws with. */
export function isBlendMode(mode: string | undefined): boolean {
  return KEYS.includes(mode || '')
}

/** The label for a mode, for a button that has to say what it is set to. */
export function blendLabel(mode: string | undefined): string {
  return BLEND_MODES.find((m) => m.key === mode)?.label || 'Normal'
}

/**
 * The mode one layer actually draws with.
 *
 * Three inputs, in order of how specific they are:
 *
 * - what this layer was set to by hand, which always wins;
 * - the stack default, which applies only when there is a stack — one layer
 *   over the basemap is not what "when several layers are drawn" means, and a
 *   default that fired on a single layer would silently change every layer
 *   anyone switched on;
 * - normal.
 *
 * An unrecognised value behaves as if it were absent rather than reaching the
 * browser, because a stored preference outlives the list it was chosen from.
 */
export function effectiveBlend(key: string, { overrides = {}, fallback = NORMAL, drawn = 0 } = {}): string {
  const own = overrides?.[key]
  if (isBlendMode(own)) return own
  if (drawn > 1 && isBlendMode(fallback)) return fallback
  return NORMAL
}

/**
 * Which layers are drawn, given the active set and whichever is soloed.
 *
 * Solo hides the rest without switching them off: the question it answers is
 * "what is this one contributing", and answering it must not cost you the stack
// you built. A solo key that is not on is ignored rather than drawing nothing,
// which is what happens if the soloed layer is switched off from elsewhere.
 */
export function drawnKeys(active: string[] = [], solo = ''): string[] {
  const keys = [...active]
  if (solo && keys.includes(solo)) return [solo]
  return keys
}

/**
 * An index moved to the top or the bottom of the stack, or by one step.
 *
 * Returns a new array. `delta` is a number of places, or 'top' or 'bottom' —
 * with eight layers on, "up" eight times to reach the top is not an ordering
// control, but a counting exercise.
 */
export function reorderStack(order: string[] = [], key: string, delta: string | number): string[] {
  const list = [...order]
  const i = list.indexOf(key)
  if (i < 0) return list
  let to
  if (delta === 'top') to = 0
  else if (delta === 'bottom') to = list.length - 1
  else {
    to = i + Number(delta || 0)
    if (!Number.isFinite(to) || to < 0 || to >= list.length) return list
  }
  if (to === i) return list
  list.splice(to, 0, ...list.splice(i, 1))
  return list
}
