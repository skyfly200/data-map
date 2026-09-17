// The guide's table of contents.
//
// The guide was one page of about six hundred lines. That is one contents list
// with forty entries, one scroll position for nine unrelated subjects, and a
// reader who cannot tell how much is left. It is now one page for each subject.
//
// Splitting a document breaks its links, and this one is linked from everywhere:
// every control in the app has a ? that points into it, the home page points at
// two of its sections, and whatever anyone bookmarked points at the rest. So the
// old anchors are kept and redirected rather than dropped, and LEGACY_ANCHORS
// below is the record of where each one went.
//
// No Nuxt, no ?raw imports, nothing bundler-specific: the tests read this and
// then read the Markdown off disk to check the two agree.

/** The pages, in reading order. An empty slug is /guide itself. */
export const GUIDE_PAGES = [
  {
    slug: '',
    file: 'index.md',
    title: 'Start here',
    blurb: 'What Nexstrata is, and the first five minutes.',
  },
  {
    slug: 'map',
    file: 'map.md',
    title: 'The map',
    blurb: 'Points, heatmaps, layers, the record panel and offline use.',
  },
  {
    slug: 'data',
    file: 'data.md',
    title: 'Data and export',
    blurb: 'The species list, the table, the filters, and how to export.',
  },
  {
    slug: 'charts',
    file: 'charts.md',
    title: 'Charts',
    blurb: 'The gallery, the chart builder and the style controls.',
  },
  {
    slug: 'analysis',
    file: 'analysis.md',
    title: 'Analysis',
    blurb: 'Correlations, species statistics and fruiting timing.',
  },
  {
    slug: 'jobs',
    file: 'jobs.md',
    title: 'Pipeline jobs',
    blurb: 'Run a job, save the result, and chain jobs together.',
  },
  {
    slug: 'layers',
    file: 'layers.md',
    title: 'Your own layers',
    blurb: 'Compute a layer in Earth Engine and draw it on the map.',
  },
  {
    slug: 'learning',
    file: 'learning.md',
    title: 'Learning',
    blurb: 'Where to start with GIS, Earth Engine and machine learning.',
  },
  {
    slug: 'sources',
    file: 'sources.md',
    title: 'Data sources',
    blurb: 'Which product each column comes from, and the terms on it.',
  },
  {
    // Generated from the option registry rather than written, so it has no
    // Markdown file behind it.
    slug: 'reference',
    file: null,
    title: 'Option reference',
    blurb: 'Every control in the app, in one list.',
  },
]

/** The URL of a guide page. */
export function guidePath(slug) {
  return slug ? `/guide/${slug}` : '/guide'
}

/** Every guide URL, for the offline shell. */
export function guidePaths() {
  return GUIDE_PAGES.map((p) => guidePath(p.slug))
}

/** The page with this slug, or null. */
export function guidePage(slug) {
  const want = String(slug || '')
  return GUIDE_PAGES.find((p) => p.slug === want) || null
}

/**
 * Where an anchor from the single-page guide lives now.
 *
 * Written as `slug#anchor`, with an empty anchor meaning the top of that page.
 * A few of these are not renames but redirects of substance: #coverage was a
 * section about a page that no longer exists, and the nearest honest answer to
 * the question it was asking is the data-quality tab.
 */
export const LEGACY_ANCHORS = {
  'start-here': '#start-here',
  map: 'map#',
  'color-and-size': 'map#color-and-size',
  heatmaps: 'map#heatmaps',
  'cell-size-and-shape': 'map#cell-size-and-shape',
  'wind-and-aspect-vectors': 'map#wind-and-aspect-vectors',
  'basemaps-and-layers': 'map#basemaps-and-layers',
  'other-map-controls': 'map#other-map-controls',
  'the-record-panel': 'map#the-record-panel',
  'offline-use': 'map#offline-use',
  'managing-saved-areas': 'map#how-to-manage-saved-areas',
  taxonomy: 'map#taxonomy',
  data: 'data#',
  table: 'data#table',
  filters: 'data#filters',
  charts: 'charts#',
  gallery: 'charts#gallery',
  build: 'charts#build',
  style: 'charts#style',
  analysis: 'analysis#',
  'what-relates-to-what': 'analysis#what-relates-to-what',
  // "Species" was a heading twice in the old document, under Data and under
  // Analysis, and both claimed this id. The Data one came first, so that is
  // where the anchor actually landed.
  species: 'data#species',
  'fruiting-timing': 'analysis#fruiting-timing',
  'year-over-year': 'analysis#year-over-year',
  'data-quality': 'analysis#data-quality',
  coverage: 'analysis#data-quality',
  'pipeline-jobs': 'jobs#',
  'your-own-earth-engine-layers': 'layers#',
  'how-the-asset-and-the-token-fit-together': 'layers#how-the-asset-and-the-token-fit-together',
  'a-tree-cover-layer-end-to-end': 'layers#a-tree-cover-layer-from-start-to-end',
  'sharing-and-saving': 'sources#sharing-and-saving',
  'where-the-data-comes-from': 'sources#where-the-data-comes-from',
  reference: 'reference#reference',
}

/** The slug and anchor a `slug#anchor` string names. */
export function splitTarget(target) {
  const [slug = '', anchor = ''] = String(target || '').split('#')
  return { slug, anchor }
}
