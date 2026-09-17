// USDA soil taxonomy: the twelve orders, and how to read a great group name.
//
// The OpenLandMap great-group raster carries about four hundred classes. That
// is far too many for a key, and a four-hundred-colour palette is not a key
// anyway — it is a lookup table you have to read pixel by pixel. So the map
// paints the twelve ORDERS, which is the level that has meaning at map scale,
// and the panel explains whichever great group you are actually over.
//
// Neither of those needs a table of four hundred descriptions, because soil
// taxonomy names are compositional and were designed to be read:
//
//   Dystrocryepts  =  dystro  +  cry  +  epts
//                     low base   cold     Inceptisol
//
// The order is the suffix, the modifiers are the prefixes, and both come from a
// fixed vocabulary. So a decoder of that vocabulary describes a class the
// dataset has and a class it does not, and cannot go stale against a table
// somebody has to maintain by hand.
//
// Shared between the tile function, which needs the order to build a palette,
// and the browser, which needs the description to put in a panel.

/**
 * The twelve orders, in the alphabetical order the literature always lists them.
 *
 * `suffix` is what the great group names of that order end in. It is not a
 * convention this app invented: it is the formative element the taxonomy
 * assigns to the order, and every name below it carries it.
 *
 * `wiki` is the article for the order. The link on a class goes here rather
 * than to the great group, because the great groups mostly have no article and
 * a link to nothing is worse than a link to the level above.
 */
export const SOIL_ORDERS = [
  {
    key: 'alfisols',
    name: 'Alfisols',
    suffix: 'alfs',
    color: '#6aa84f',
    summary: 'Clay has washed down into the subsoil, and the soil is still base-rich.',
    where: 'Under broadleaf forest in humid and subhumid climates. Fertile.',
    wiki: 'https://en.wikipedia.org/wiki/Alfisol',
  },
  {
    key: 'andisols',
    name: 'Andisols',
    suffix: 'ands',
    color: '#8e44ad',
    summary: 'Formed in volcanic ash. Light, very porous, and they hold a lot of water.',
    where: 'Volcanic regions. High in organic matter and they fix phosphorus strongly.',
    wiki: 'https://en.wikipedia.org/wiki/Andisol',
  },
  {
    key: 'aridisols',
    name: 'Aridisols',
    suffix: 'ids',
    color: '#f1c232',
    summary: 'Dry for most of the year, with salts or carbonate collected in the profile.',
    where: 'Deserts and steppe. Little leaching, so what comes in stays.',
    wiki: 'https://en.wikipedia.org/wiki/Aridisol',
  },
  {
    key: 'entisols',
    name: 'Entisols',
    suffix: 'ents',
    color: '#e0b080',
    summary: 'Almost no horizons. The soil is too young or too disturbed to have made any.',
    where: 'Floodplains, dunes, steep eroding slopes, fresh deposits.',
    wiki: 'https://en.wikipedia.org/wiki/Entisol',
  },
  {
    key: 'gelisols',
    name: 'Gelisols',
    suffix: 'els',
    color: '#76c7e0',
    summary: 'Permafrost within two metres of the surface.',
    where: 'Arctic and high alpine ground. Freezing and thawing churns the profile.',
    wiki: 'https://en.wikipedia.org/wiki/Gelisol',
  },
  {
    key: 'histosols',
    name: 'Histosols',
    suffix: 'ists',
    color: '#3d2b1f',
    summary: 'Made of organic material rather than mineral material. Peat and muck.',
    where: 'Bogs, fens and swamps, where water stops things decomposing.',
    wiki: 'https://en.wikipedia.org/wiki/Histosol',
  },
  {
    key: 'inceptisols',
    name: 'Inceptisols',
    suffix: 'epts',
    color: '#b5651d',
    summary: 'Horizons have started to form, but none of them is strong yet.',
    where: 'One step on from an Entisol. Common on young mountain slopes.',
    wiki: 'https://en.wikipedia.org/wiki/Inceptisol',
  },
  {
    key: 'mollisols',
    name: 'Mollisols',
    suffix: 'olls',
    color: '#6d4c2f',
    summary: 'A thick, dark, base-rich surface horizon built by grass roots.',
    where: 'Prairie and steppe. The most agriculturally productive order.',
    wiki: 'https://en.wikipedia.org/wiki/Mollisol',
  },
  {
    key: 'oxisols',
    name: 'Oxisols',
    suffix: 'ox',
    color: '#c0392b',
    summary: 'Weathered for so long that little is left but iron, aluminium and quartz.',
    where: 'Old tropical land surfaces. Low fertility and very stable structure.',
    wiki: 'https://en.wikipedia.org/wiki/Oxisol',
  },
  {
    key: 'spodosols',
    name: 'Spodosols',
    suffix: 'ods',
    color: '#4a6fa5',
    summary: 'Acid and sandy. Iron and organic matter wash down into a bright subsoil band.',
    where: 'Conifer forest on sandy parent material, in cool humid climates.',
    wiki: 'https://en.wikipedia.org/wiki/Spodosol',
  },
  {
    key: 'ultisols',
    name: 'Ultisols',
    suffix: 'ults',
    color: '#e8743b',
    summary: 'Clay in the subsoil like an Alfisol, but leached of its bases and acid.',
    where: 'Old, humid, warm landscapes. Needs lime and fertiliser to farm.',
    wiki: 'https://en.wikipedia.org/wiki/Ultisol',
  },
  {
    key: 'vertisols',
    name: 'Vertisols',
    suffix: 'erts',
    color: '#2e8b7a',
    summary: 'Swelling clay. The ground cracks open when dry and seals when wet.',
    where: 'Flat land on clay-rich parent material with a dry season.',
    wiki: 'https://en.wikipedia.org/wiki/Vertisol',
  },
]

/** The article on the classification itself, for the panel's header. */
export const SOIL_TAXONOMY_WIKI = 'https://en.wikipedia.org/wiki/USDA_soil_taxonomy'

/** The order suffixes, longest first, so 'olls' is tested before 'els'. */
const SUFFIXES = [...SOIL_ORDERS]
  .sort((a, b) => b.suffix.length - a.suffix.length)

/**
 * The formative elements that appear in front of the order suffix.
 *
 * Longest first, because several are prefixes of each other — 'quartzi' before
 * 'qu', 'dystr' before 'dur'. Matching the short one first would describe
 * Quartzipsamments as a wet soil.
 *
 * This is not the whole vocabulary. It is the part that appears in the great
 * groups of cool, humid, forested ground, which is the ground this app is
 * about, plus the common ones from everywhere else so a class from outside
 * that still says something.
 */
export const FORMATIVE_ELEMENTS = [
  { element: 'quartzi', meaning: 'almost pure quartz sand, with very little left to weather' },
  { element: 'fragi', meaning: 'a dense brittle pan in the subsoil that roots and water struggle through' },
  { element: 'plinth', meaning: 'iron-rich material that hardens irreversibly once it dries' },
  { element: 'sombri', meaning: 'a dark horizon of humus washed down from above' },
  { element: 'glossi', meaning: 'pale tongues reaching down into the clay horizon' },
  { element: 'dystro', meaning: 'low base saturation: acid, and short of calcium and magnesium' },
  { element: 'dystr', meaning: 'low base saturation: acid, and short of calcium and magnesium' },
  { element: 'eutro', meaning: 'high base saturation: well supplied with calcium and magnesium' },
  { element: 'eutr', meaning: 'high base saturation: well supplied with calcium and magnesium' },
  { element: 'psamm', meaning: 'sandy throughout' },
  { element: 'kandi', meaning: 'a clay horizon of low-activity clays' },
  { element: 'calci', meaning: 'a horizon where carbonate has collected' },
  { element: 'melan', meaning: 'a thick black surface horizon very high in organic matter' },
  { element: 'sapr', meaning: 'organic material decomposed past recognition' },
  { element: 'hemi', meaning: 'organic material about half decomposed' },
  { element: 'fibr', meaning: 'organic material still fibrous and barely decomposed' },
  { element: 'lithi', meaning: 'hard rock close to the surface' },
  { element: 'natr', meaning: 'a clay horizon high in sodium' },
  { element: 'rhod', meaning: 'dark red throughout' },
  { element: 'umbr', meaning: 'a dark surface horizon that is acid rather than base-rich' },
  { element: 'sulf', meaning: 'sulfur-bearing, and strongly acid when it is drained' },
  { element: 'fluv', meaning: 'on river deposits, with the layering still visible' },
  { element: 'verm', meaning: 'heavily mixed by worms' },
  { element: 'gloss', meaning: 'pale tongues reaching down into the clay horizon' },
  { element: 'torr', meaning: 'hot and dry: usually too dry for plants when it is warm enough for them' },
  { element: 'vitr', meaning: 'glassy volcanic material, coarse and free-draining' },
  { element: 'hapl', meaning: 'the simple case for its group: minimal horizon development' },
  { element: 'pale', meaning: 'an old, strongly developed profile' },
  { element: 'petr', meaning: 'a cemented, rock-hard horizon' },
  { element: 'argi', meaning: 'a clay-enriched subsoil horizon' },
  { element: 'endo', meaning: 'saturated from the bottom of the profile up' },
  { element: 'xero', meaning: 'a Mediterranean pattern: wet cool winters, dry warm summers' },
  { element: 'xer', meaning: 'a Mediterranean pattern: wet cool winters, dry warm summers' },
  { element: 'alb', meaning: 'a pale leached horizon below the surface' },
  { element: 'acr', meaning: 'extremely weathered, with almost no capacity to hold nutrients' },
  { element: 'bor', meaning: 'cold: the older name for the northern conifer zone' },
  { element: 'cry', meaning: 'cold: a mean soil temperature under about 8 °C' },
  { element: 'dur', meaning: 'a silica-cemented pan' },
  { element: 'epi', meaning: 'a perched wet layer sitting over drier soil' },
  { element: 'hum', meaning: 'high in organic matter' },
  { element: 'orth', meaning: 'the typical case for its group' },
  { element: 'aqu', meaning: 'wet: saturated long enough each year to lose its oxygen' },
  { element: 'usti', meaning: 'a monsoonal pattern: a dry season and a wet growing season' },
  { element: 'ust', meaning: 'a monsoonal pattern: a dry season and a wet growing season' },
  { element: 'udi', meaning: 'humid: rarely dry anywhere in the profile' },
  { element: 'ud', meaning: 'humid: rarely dry anywhere in the profile' },
]

const clean = (name) => String(name ?? '').trim().toLowerCase()

/**
 * The order a great group belongs to, read off the end of its name.
 *
 * Null when nothing matches, which is the honest answer for a label that is not
 * a great group at all — a nodata string, or a name from a revision that added
 * an order this table does not have.
 */
export function orderForGreatGroup(name) {
  const lower = clean(name)
  if (!lower) return null
  return SUFFIXES.find((o) => lower.endsWith(o.suffix)) || null
}

/**
 * The formative elements in a great group name, in the order they appear.
 *
 * Matched left to right against the part of the name in front of the order
 * suffix, so 'Dystrocryepts' yields dystro then cry rather than every element
 * that happens to be a substring of it.
 */
export function elementsOf(name) {
  const order = orderForGreatGroup(name)
  const lower = clean(name)
  if (!lower) return []
  let head = order ? lower.slice(0, lower.length - order.suffix.length) : lower
  const out = []
  let guard = 0
  while (head && guard < 8) {
    guard += 1
    const hit = FORMATIVE_ELEMENTS.find((e) => head.startsWith(e.element))
    if (!hit) {
      // An element this table does not know. Drop one letter and try again:
      // the connecting vowels between elements are not themselves elements, and
      // stopping at the first one would hide everything after it.
      head = head.slice(1)
      continue
    }
    if (!out.some((e) => e.meaning === hit.meaning)) out.push(hit)
    head = head.slice(hit.element.length)
  }
  return out
}

/**
 * Everything the panel says about one great group.
 *
 * `known` is false when the name does not end in an order suffix. The caller
 * shows the name alone in that case rather than a description assembled from
 * nothing, because a confident description of a class nobody can place is
 * exactly the kind of wrong this app tries not to be.
 */
export function describeGreatGroup(name) {
  const label = String(name ?? '').trim()
  const order = orderForGreatGroup(label)
  const elements = order ? elementsOf(label) : []
  return {
    name: label,
    known: !!order,
    order: order ? order.name : '',
    orderKey: order ? order.key : '',
    color: order ? order.color : '#888888',
    summary: order ? order.summary : '',
    where: order ? order.where : '',
    wiki: order ? order.wiki : SOIL_TAXONOMY_WIKI,
    elements: elements.map((e) => ({ element: e.element, meaning: e.meaning })),
  }
}

/**
 * Great groups whose name or whose order matches `query`.
 *
 * Matches the order too, so "spodosol" finds every podzol great group without
 * the reader having to know that none of them contains that word.
 */
export function searchGreatGroups(classes = [], query = '') {
  const q = clean(query)
  if (!q) return [...classes]
  return classes.filter((c) => {
    const name = clean(c.name)
    if (name.includes(q)) return true
    const order = orderForGreatGroup(c.name)
    return !!order && (order.key.includes(q) || clean(order.name).includes(q))
  })
}

/**
 * The palette index for a class, 1-based, matching SOIL_ORDERS.
 *
 * Zero for a class with no order, which the layer paints as nodata rather than
 * as the first order in the list.
 */
export function orderIndex(name) {
  const order = orderForGreatGroup(name)
  return order ? SOIL_ORDERS.indexOf(order) + 1 : 0
}

/**
 * Great groups that FRMS members have flagged as matsutake ground on the Front
 * Range, by name.
 *
 * By name rather than by the raster's class code on purpose. The names are the
 * taxonomy's and do not move; the codes belong to one version of one published
 * raster, and a version bump that renumbers them would silently repoint this
 * list at eighteen different soils. The codes are in the test, as a check that
 * the table being read is the one this list was built from.
 */
export const MATSUTAKE_GREAT_GROUPS = [
  'Hapludalfs', 'Cryoboralfs', 'Haplocryalfs',
  'Haplocryands', 'Hapludands', 'Vitricryands', 'Vitrixerands',
  'Cryopsamments', 'Quartzipsamments', 'Udipsamments', 'Xeropsamments',
  'Dystrocryepts', 'Dystrudepts', 'Haploxerepts', 'Cryumbrepts',
  'Fragiorthods', 'Haplohumods', 'Haplorthods',
]
