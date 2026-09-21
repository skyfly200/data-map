// Which taxa the app asks iNaturalist for.
//
// The list was a comma-separated environment variable, read by the Python
// pipeline and by nothing else. That is the right place for it to END — the
// pipeline is what runs a forty-genus fetch overnight — but the wrong place for
// it to be EDITED, because editing it meant finding a .env file, and because
// nothing in the app could then tell you what it was set to.
//
// So the list lives here, in settings, and the app does two things with it: it
// drives the on-demand fetch in the browser, and it writes the environment
// variable for the pipeline to be pasted somewhere. One list, both paths.
//
// A taxon here is a NAME at any rank. iNaturalist resolves "Amanita" to a
// genus, "Amanitaceae" to a family and "Fungi" to a kingdom, and returns
// everything below it — which is what makes "all fungi" a list of one.

/**
 * The genera FRMS has been fetching, which is what the pipeline's .env shipped
 * with. Kept as the default so an app with no saved settings behaves the way
// the project already did.
 */
export const FRMS_GENERA: string[] = [
  'Auricularia', 'Tremella', 'Xylaria', 'Cordyceps', 'Ganoderma', 'Tricholoma',
  'Hydnellum', 'Hydnum', 'Clavariadelphus', 'Pleurotus', 'Hericium', 'Morchella',
  'Amanita', 'Boletus', 'Sarcodon', 'Russula', 'Lactarius', 'Cortinarius',
  'Clitocybe', 'Hypholoma', 'Hygrophorus', 'Psilocybe', 'Coprinus', 'Coprinopsis',
  'Marasmius', 'Mycena', 'Lepiota', 'Agaricus', 'Flammulina', 'Boletopsis',
  'Albatrellus', 'Fomitopsis', 'Trametes', 'Trichaptum', 'Lanmaoa', 'Hypomyces',
  'Cantharellus', 'Grifola', 'Laetiporus', 'Armillaria', 'Rhizopogon', 'Panaeolus',
]

interface TaxonPreset {
  key: string
  label: string
  taxa: string[]
  scale: string
  note: string
}

/**
 * Ready-made lists.
 *
 * The kingdom presets are each one name, because iNaturalist returns everything
 * below a taxon: asking for Fungi is asking for every fungus, not for a list of
 * fungi to then ask about. `scale` is what that costs, and it is on the preset
// rather than in a footnote because it is the whole difference between them.
 */
export const TAXON_PRESETS: TaxonPreset[] = [
  {
    key: 'frms',
    label: 'FRMS genera',
    taxa: FRMS_GENERA,
    scale: 'tens of thousands of records',
    note: 'The forty-two genera this project has been collecting. A good default.',
  },
  {
    key: 'fungi',
    label: 'All fungi',
    taxa: ['Fungi'],
    scale: 'hundreds of thousands of records',
    note: 'Every fungus, including the lichens, moulds and rusts nobody forages.',
  },
  {
    key: 'plants',
    label: 'All plants',
    taxa: ['Plantae'],
    scale: 'millions of records',
    note: 'Useful as host context. The commonest observations on iNaturalist.',
  },
  {
    key: 'animals',
    label: 'All animals',
    taxa: ['Animalia'],
    scale: 'millions of records',
    note: 'Everything from birds to beetles.',
  },
  {
    key: 'fungi-plants',
    label: 'Fungi and plants',
    taxa: ['Fungi', 'Plantae'],
    scale: 'millions of records',
    note: 'Fruiting bodies and the hosts they depend on, in one dataset.',
  },
]

/** Fewer than this is not a taxon name; more than this is not one either. */
const MIN_LENGTH = 3
const MAX_LENGTH = 80

/**
 * How many taxa one list may name.
 *
 * Each one is a separate fetch, so a list is a queue. Past about this many the
 * queue takes longer than anybody watches it for, and the answer is the
// pipeline rather than the browser.
 */
export const MAX_TAXA = 200

/**
 * Letters, spaces, hyphens, apostrophes, full stops and the hybrid sign.
 *
 * Deliberately narrow. A taxon name goes into a query to somebody else's API,
// and nothing that is not a name has any business there — a digit or a bracket
// is a typo or an attempt at something, and neither is worth a request.
 */
const NAME = /^[\p{L}][\p{L}\s'.×-]*$/u

/** Whether a string could be a taxon name. Not whether iNaturalist knows it. */
export function isTaxonName(value: any): boolean {
  const text = String(value ?? '').trim()
  if (text.length < MIN_LENGTH || text.length > MAX_LENGTH) return false
  return NAME.test(text)
}

/**
 * A list of taxon names from whatever the caller has.
 *
 * Accepts an array or a comma-separated string, because it reads both what the
// settings hold and what somebody pastes in from an environment variable.
 *
 * Duplicates go by case-insensitive comparison but the first spelling is kept:
// "amanita" and "Amanita" are one genus, and the one worth showing is the one
// written the way the rest of the world writes it.
 */
export function parseTaxa(value: string | string[] | null | undefined): string[] {
  const parts = Array.isArray(value) ? value : String(value ?? '').split(',')
  const out: string[] = []
  const seen = new Set<string>()
  for (const part of parts) {
    const text = String(part ?? '').trim().replace(/\s+/g, ' ')
    if (!isTaxonName(text)) continue
    const key = text.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(text)
    if (out.length >= MAX_TAXA) break
  }
  return out
}

/** The names that were thrown away, so the UI can say which rather than how many. */
export function rejectedTaxa(value: string | string[] | null | undefined): string[] {
  const parts = Array.isArray(value) ? value : String(value ?? '').split(',')
  return parts
    .map((p) => String(p ?? '').trim().replace(/\s+/g, ' '))
    .filter((p) => p && !isTaxonName(p))
}

/** The canonical comma-separated form, for storage and for the env var. */
export function formatTaxa(list: string | string[] | null | undefined): string {
  return parseTaxa(list).join(', ')
}

/**
 * The environment variable line the Python pipeline reads.
 *
 * Lower-cased, because that is what the pipeline's own parser does with it and
// a line that round-trips unchanged is a line nobody has to wonder about.
 */
export function asEnvLine(list: string | string[] | null | undefined): string {
  return `INAT_TAXON_NAME=${parseTaxa(list).map((t) => t.toLowerCase()).join(', ')}`
}

/** The preset a list exactly matches, or null when it is somebody's own list. */
export function matchingPreset(list: string | string[] | null | undefined): TaxonPreset | null {
  const mine = parseTaxa(list).map((t) => t.toLowerCase()).sort()
  return TAXON_PRESETS.find((p) => {
    const theirs = parseTaxa(p.taxa).map((t) => t.toLowerCase()).sort()
    return theirs.length === mine.length && theirs.every((t, i) => t === mine[i])
  }) || null
}

/**
 * Roughly what a list will cost to fetch.
 *
 * Not a count — nobody can know that without asking — but the difference
// between forty genera and every plant on Earth is four orders of magnitude,
// and a viewer about to queue the second one should be told before they do.
 */
export function scaleOf(list: string | string[] | null | undefined): string {
  const taxa = parseTaxa(list).map((t) => t.toLowerCase())
  const kingdoms = ['plantae', 'animalia', 'chromista', 'protozoa', 'bacteria']
  if (taxa.some((t) => kingdoms.includes(t))) return 'huge'
  if (taxa.includes('fungi')) return 'large'
  if (taxa.length > 60) return 'large'
  return 'ordinary'
}
