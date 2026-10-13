// Well-known choice edibles found in Colorado, for the foray planner's Edibles
// filter. Deliberately short and conservative: genera listed whole are ones
// whose members are widely eaten with no deadly look-alikes in the genus;
// otherwise only named species. It is a planning aid, never an identification.

export const EDIBLE_GENERA = new Set([
  'Morchella', 'Cantharellus', 'Craterellus', 'Hericium', 'Sparassis', 'Laetiporus', 'Calvatia',
  'Pleurotus', 'Grifola', 'Hydnum', 'Polyozellus', 'Sarcomyxa',
])

export const EDIBLE_SPECIES = new Set([
  'Boletus edulis', 'Boletus rubriceps', 'Boletus barrowsii', 'Boletus rex-veris', 'Boletus regineus',
  'Lactarius deliciosus', 'Lactarius rubrilacteus', 'Hypomyces lactifluorum',
  'Agaricus campestris', 'Agaricus augustus', 'Agaricus bitorquis', 'Coprinus comatus',
  'Lycoperdon perlatum', 'Lepista nuda', 'Cortinarius caperatus', 'Tricholoma murrillianum',
  'Gomphus clavatus', 'Leccinum insigne', 'Suillus brevipes', 'Suillus luteus', 'Lyophyllum decastes',
])

export const EDIBLE_DISCLAIMER = 'Edibles is a short list of well-known choice edibles. Never eat a mushroom on an app\'s word: confirm every find with an experienced identifier.'

/** Is a scientific name a listed edible? Matches the species, or its genus when the genus is listed. */
export function isEdible(name: string | null | undefined): boolean {
  const s = String(name || '').trim()
  if (!s) return false
  if (EDIBLE_SPECIES.has(s)) return true
  return EDIBLE_GENERA.has(s.split(/\s+/)[0])
}
