// Which taxa a loaded dataset actually contains.
//
// The job runner's taxon field was free text, and free text is how somebody
// asks for "Amanita" against a dataset that carries no genus column and gets
// told their area and date range are empty. The field should offer what is
// there, so the wrong answer is not typeable in the first place.
//
// Read from the dataset already in the browser rather than from a new endpoint:
// the map, the table and the charts all hold it, it is the dataset the picker
// is currently pointed at, and counting it is cheaper than asking a server
// which taxa it has.

/** Coarse to fine, which is the order a picker should list them in. */
export const TAXON_RANKS = [
  { key: 'kingdom', label: 'Kingdom' },
  { key: 'phylum', label: 'Phylum' },
  { key: 'class', label: 'Class' },
  { key: 'order', label: 'Order' },
  { key: 'family', label: 'Family' },
  { key: 'genus', label: 'Genus' },
  { key: 'species', label: 'Species' },
]

/**
 * The genus implied by a species binomial.
 *
 * Only used where there is no genus column, and for exactly the reason the
 * pipeline's matcher does the same: the committed baseline carries a binomial
 * and nothing else, and a picker that offered no genera against it would be
 * offering the reader nothing they could act on. A real genus column always
 * wins — see matchesTaxon in netlify/lib/job-source.mjs, which is the other
 * half of this and has to agree with it.
 */
export function impliedGenus(props = {}) {
  if (String(props.genus || '').trim()) return ''
  const species = String(props.species || '').trim()
  if (!species) return ''
  const first = species.split(/\s+/)[0]
  // A binomial's genus is capitalised and is not the whole name. "sp." and
  // bare epithets are not genera.
  if (first.length < 3 || first !== species.split(/\s+/)[0]) return ''
  return /^[A-Z][a-z-]+$/.test(first) ? first : ''
}

/**
 * Every taxon name in a feature collection, by rank, with counts.
 *
 * Sorted by count within a rank, because the useful ones are the well-recorded
 * ones and a picker of four hundred alphabetical names buries them.
 */
export function taxaInFeatures(features = [], { min = 1 } = {}) {
  const byRank = new Map(TAXON_RANKS.map((r) => [r.key, new Map()]))

  for (const f of features) {
    const props = f?.properties || {}
    for (const rank of TAXON_RANKS) {
      const value = String(props[rank.key] || '').trim()
      if (!value) continue
      const counts = byRank.get(rank.key)
      counts.set(value, (counts.get(value) || 0) + 1)
    }
    const implied = impliedGenus(props)
    if (implied) {
      const counts = byRank.get('genus')
      counts.set(implied, (counts.get(implied) || 0) + 1)
    }
  }

  return TAXON_RANKS
    .map((rank) => ({
      ...rank,
      taxa: [...byRank.get(rank.key).entries()]
        .filter(([, n]) => n >= min)
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .map(([name, count]) => ({ name, count })),
    }))
    .filter((rank) => rank.taxa.length)
}

/** How many records a named taxon covers, at whatever rank it sits. */
export function countForTaxon(ranks = [], taxon = '') {
  const needle = String(taxon || '').trim().toLowerCase()
  if (!needle) return 0
  for (const rank of ranks) {
    const hit = rank.taxa.find((t) => t.name.toLowerCase() === needle)
    if (hit) return hit.count
  }
  return 0
}

/** Is this name one the dataset carries? The question the old field could not ask. */
export function datasetHasTaxon(ranks = [], taxon = '') {
  return countForTaxon(ranks, taxon) > 0
}
