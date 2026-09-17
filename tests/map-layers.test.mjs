import test from 'node:test'
import assert from 'node:assert/strict'

import {
  TILE_LAYERS, TIME_LAYERS,
  arcgisExportUrl, filterLayerGroups, gibs, gibsUrl, layerDataType, layerDate,
  layerGroups, layerSource, tileBounds,
} from '../composables/mapLayers.js'

test('tile bounds cover the whole world at zoom 0 and quarter it at zoom 1', () => {
  const R = 20037508.342789244
  const [xmin, ymin, xmax, ymax] = tileBounds(0, 0, 0)
  assert.ok(Math.abs(xmin + R) < 1e-6 && Math.abs(ymin + R) < 1e-6)
  assert.ok(Math.abs(xmax - R) < 1e-6 && Math.abs(ymax - R) < 1e-6)

  // y grows downward in XYZ, so tile (0,0) at z1 is the NORTH-west quadrant.
  const nw = tileBounds(0, 0, 1)
  assert.ok(Math.abs(nw[3] - R) < 1e-6, 'top edge at the north pole')
  assert.ok(Math.abs(nw[1]) < 1e-6, 'bottom edge at the equator')
  const sw = tileBounds(0, 1, 1)
  assert.ok(Math.abs(sw[3]) < 1e-6, 'the tile below starts at the equator')
})

test('an ArcGIS export request asks for the tile it was given', () => {
  const url = arcgisExportUrl('https://example.test/MapServer', 3, 5, 4)
  const q = new URL(url).searchParams
  assert.equal(new URL(url).pathname, '/MapServer/export')
  assert.deepEqual(q.get('bbox').split(',').map(Number), tileBounds(3, 5, 4))
  assert.equal(q.get('bboxSR'), '3857')
  assert.equal(q.get('imageSR'), '3857')
  assert.equal(q.get('size'), '256,256')
  assert.equal(q.get('transparent'), 'true')
  assert.equal(q.get('f'), 'image')
  // No sublayer selection unless one was asked for — omitting it draws the
  // whole service, which is what a single-purpose service should draw.
  assert.equal(q.get('layers'), null)
  assert.equal(new URL(arcgisExportUrl('https://e.test/M', 0, 0, 0, { layers: 'show:3' }))
    .searchParams.get('layers'), 'show:3')
})

test('a GIBS url carries the layer, matrix level and a date placeholder', () => {
  const url = gibsUrl('MODIS_Terra_NDVI_8Day', { level: 9 })
  assert.ok(url.includes('/MODIS_Terra_NDVI_8Day/default/{date}/'))
  assert.ok(url.includes('GoogleMapsCompatible_Level9'))
  assert.ok(url.endsWith('/{z}/{y}/{x}.png'))
  // GIBS is y-before-x, which is the opposite of the OSM-style templates
  // alongside it — getting it backwards silently draws the wrong hemisphere.
  assert.ok(url.indexOf('{y}') < url.indexOf('{x}'))
})

test('a GIBS layer takes its tile ceiling from its matrix set', () => {
  const layer = gibs('MODIS_Terra_NDVI_8Day', 9)
  assert.ok(layer.url.includes('GoogleMapsCompatible_Level9'))
  assert.equal(layer.maxZoom, 9)
})

test('no GIBS layer asks for tiles above its own matrix set', () => {
  // The regression. These were two independent numbers and every GIBS layer
  // had drifted two levels apart: the catalogue's maxZoom becomes Leaflet's
  // maxNativeZoom, so the map requested tiles the matrix does not contain and
  // GIBS answered each one with a 400 — a screenful of console errors and a
  // layer that stopped drawing once you zoomed past its ceiling.
  const gibsLayers = TILE_LAYERS.filter((l) => /gibs\.earthdata/.test(l.url || ''))
  // Rainfall, land surface temp and NDVI. SMAP soil moisture used to be a fourth
  // but GIBS only serves it in EPSG:4326, so it moved to Earth Engine.
  assert.ok(gibsLayers.length >= 3, 'expected the GIBS layers to still be here')

  for (const layer of gibsLayers) {
    const level = Number((layer.url.match(/GoogleMapsCompatible_Level(\d+)/) || [])[1])
    assert.ok(Number.isFinite(level), `${layer.name} has no matrix level in its url`)
    assert.equal(
      layer.maxZoom, level,
      `${layer.name} serves Level${level} but declares a ceiling of ${layer.maxZoom}. `
      + 'Build it with gibs() so the two come from one number.',
    )
  }
})

test('layer dates back off by the product lag and never reach tomorrow', () => {
  const now = new Date('2026-03-10T12:00:00Z')
  assert.equal(layerDate(0, now), '2026-03-10')
  assert.equal(layerDate(10, now), '2026-02-28')
  // Across a year boundary too.
  assert.equal(layerDate(10, new Date('2026-01-05T00:00:00Z')), '2025-12-26')
})

test('every layer has a name, an attribution and a way to fetch a tile', () => {
  const names = new Set()
  for (const l of TILE_LAYERS) {
    assert.ok(l.name, 'unnamed layer')
    assert.ok(!names.has(l.name), `duplicate layer name ${l.name}`)
    names.add(l.name)
    assert.ok(l.group, `${l.name} has no group`)
    assert.ok(l.attribution, `${l.name} is unattributed`)
    // Exactly one of the two fetch styles.
    assert.equal(Boolean(l.url) !== Boolean(l.arcgis), true, `${l.name}: url XOR arcgis`)
    if (l.url) {
      assert.ok(/\{z\}/.test(l.url) && /\{x\}/.test(l.url) && /\{y\}/.test(l.url),
        `${l.name} is missing a tile placeholder`)
      assert.ok(l.url.startsWith('https://'), `${l.name} is not over https`)
    } else {
      assert.ok(l.arcgis.startsWith('https://') && l.arcgis.endsWith('MapServer'),
        `${l.name} is not a MapServer endpoint`)
    }
    assert.ok(Number.isFinite(l.maxZoom), `${l.name} has no maxZoom`)
  }
})

test('a time-varying layer has a date placeholder and a lag; a fixed one has neither', () => {
  for (const l of TILE_LAYERS) {
    if (l.time) {
      assert.ok(l.url?.includes('{date}'), `${l.name} is time-varying but has no {date}`)
      assert.ok(Number.isInteger(l.lag) && l.lag >= 0, `${l.name} has no usable lag`)
    } else {
      assert.ok(!l.url?.includes('{date}'), `${l.name} has a {date} nothing will fill`)
    }
  }
  assert.deepEqual(TIME_LAYERS, TILE_LAYERS.filter((l) => l.time).map((l) => l.name))
  assert.ok(TIME_LAYERS.length >= 3)
})

test('every measured layer carries a key a reader can use', () => {
  // Imagery and place labels are pictures, not measurements, and are the only
  // things allowed through without one.
  const PICTURES = new Set(['USGS topo', 'USGS imagery', 'OpenTopoMap relief', 'Hillshade',
    'Place labels', 'Hiking trails'])
  for (const l of TILE_LAYERS) {
    if (PICTURES.has(l.name)) continue
    assert.ok(l.legend, `${l.name} has no key`)
    if (l.legend.type === 'ramp') {
      assert.ok(l.legend.stops.length >= 2, `${l.name}: a ramp needs two ends`)
      for (const c of l.legend.stops) assert.match(c, /^#[0-9a-f]{6}$/i, `${l.name}: ${c}`)
      assert.ok(l.legend.min && l.legend.max && l.legend.unit, `${l.name}: an unlabelled ramp`)
    } else {
      assert.equal(l.legend.type, 'classes')
      assert.ok(l.legend.items.length >= 2)
      for (const it of l.legend.items) {
        assert.match(it.color, /^#[0-9a-f]{6}$/i)
        assert.ok(it.label)
      }
    }
  }
})

test('every layer that could read as empty ground carries its caveat', () => {
  // A blank rainfall layer and a dry one look identical, and a blank ownership
  // layer reads as "no public land here". Anything modelled or partial says so.
  for (const l of TILE_LAYERS) {
    if (l.group === 'Weather' || l.group === 'Ground' || l.group === 'Vegetation') {
      assert.ok(l.note, `${l.name} has no caveat`)
    }
  }
})

test('no layer is served from the host that went down', () => {
  // ESA WorldCover came from services.terrascope.be until that host started
  // failing at the protocol level. It is rendered by Earth Engine now, and its
  // class colours are asserted alongside the other EE layers.
  for (const l of TILE_LAYERS) {
    assert.ok(!/terrascope/.test(l.url || ''), `${l.name} still points at Terrascope`)
  }
})

test('grouping lists every layer once, and names each group once', () => {
  const groups = layerGroups()
  const flat = groups.flatMap((g) => g.layers)
  assert.equal(flat.length, TILE_LAYERS.length)
  const names = groups.map((g) => g.name)
  assert.equal(new Set(names).size, names.length)
})

test('opacity defaults keep context layers under the data', () => {
  // A layer meant as context that ships opaque covers the observations it is
  // supposed to be context for.
  for (const l of TILE_LAYERS) {
    if (l.opacity === undefined) continue
    assert.ok(l.opacity > 0 && l.opacity <= 1, `${l.name}: opacity ${l.opacity}`)
  }
  const ownership = TILE_LAYERS.find((l) => l.name === 'Land ownership (US)')
  assert.ok(ownership.opacity < 0.6)
})

// ── Searching the layer manager ──────────────────────────────────────────────

/** The catalogue in the shape the layer manager receives it. */
const managerGroups = () => layerGroups().map((g) => ({ label: g.name, items: g.layers }))

test('a layer is found by any part of its name', () => {
  const hits = filterLayerGroups(managerGroups(), 'ndvi').flatMap((g) => g.items.map((i) => i.name))
  assert.deepEqual(hits, ['NDVI (greenness)'])
})

test('searching for a group by its start returns the whole group', () => {
  // Someone typing a heading means the section, not a layer that happens to
  // contain the word.
  const weather = layerGroups().find((g) => g.name === 'Weather')
  const got = filterLayerGroups(managerGroups(), 'weath')
  assert.equal(got.length, 1)
  assert.equal(got[0].items.length, weather.layers.length)
})

test('a query inside a group name does not drag the whole group in', () => {
  // The bug this function exists for: "Terrain" contains "rain", so matching
  // group names by substring returned Hillshade, USGS topo, USGS imagery and
  // OpenTopoMap relief for a search whose only real answers are the two
  // rainfall layers.
  const names = filterLayerGroups(managerGroups(), 'rain').flatMap((g) => g.items.map((i) => i.name))
  assert.ok(names.every((n) => n.toLowerCase().includes('rain')), `got ${JSON.stringify(names)}`)
  assert.ok(names.includes('Rain past 24h (US)'))
  assert.ok(names.includes('Rainfall (global)'))
  assert.ok(!names.includes('Hillshade'))
})

test('an empty query changes nothing, and a hopeless one returns nothing', () => {
  const groups = managerGroups()
  assert.equal(filterLayerGroups(groups, ''), groups)
  assert.equal(filterLayerGroups(groups, '   '), groups)
  assert.deepEqual(filterLayerGroups(groups, 'zzzz'), [])
})

test('searching ignores case and surrounding space', () => {
  const a = filterLayerGroups(managerGroups(), '  HILLSHADE ').flatMap((g) => g.items.map((i) => i.name))
  assert.deepEqual(a, ['Hillshade'])
})

test('a custom group searches like any other', () => {
  // The registered Earth Engine layers arrive as their own group, and must be
  // reachable the same way rather than being a special case.
  const groups = [{ label: 'Custom', items: [{ key: 'custom:tree-cover', name: 'Tree cover 2026' }] }]
  assert.equal(filterLayerGroups(groups, 'tree')[0].items.length, 1)
  assert.equal(filterLayerGroups(groups, 'cust')[0].items.length, 1)
  assert.deepEqual(filterLayerGroups(groups, 'fire'), [])
})

// ── Grouping by source and by data type ──────────────────────────────────────

test('a layer source is read from its attribution, most specific first', () => {
  // The sensor is preferred over the platform that serves it, and MTBS/GAP are
  // matched before the bare USGS they contain.
  assert.equal(layerSource('NASA MODIS MCD64A1 via Google Earth Engine'), 'NASA MODIS')
  assert.equal(layerSource('Copernicus Sentinel-2 via Google Earth Engine'), 'Copernicus Sentinel-2')
  assert.equal(layerSource('USFS / MTBS via Google Earth Engine'), 'USFS MTBS')
  assert.equal(layerSource('USFS TreeMap via Google Earth Engine'), 'USFS TreeMap')
  assert.equal(layerSource('Hansen / UMD / Google / USGS / NASA via Google Earth Engine'), 'Hansen GFC')
  assert.equal(layerSource('USGS GAP/LANDFIRE National Terrestrial Ecosystems 2011 via Google Earth Engine'), 'USGS GAP')
  assert.equal(layerSource('USGS The National Map'), 'USGS')
  assert.equal(layerSource('NASA GIBS / MODIS Terra'), 'NASA MODIS')
})

test('an unrecognised attribution keeps its first credited token', () => {
  assert.equal(layerSource('Some New Provider / else'), 'Some New Provider')
  assert.equal(layerSource(''), 'Other')
})

test('every catalogue layer resolves to a non-empty source', () => {
  for (const l of TILE_LAYERS) {
    const s = layerSource(l.attribution)
    assert.ok(s && s !== 'Other', `${l.name} has no recognisable source (${l.attribution})`)
  }
})

test('data type comes from the legend, and imagery has none', () => {
  assert.equal(layerDataType({ type: 'ramp' }), 'Continuous raster')
  assert.equal(layerDataType({ type: 'classes' }), 'Categorical raster')
  assert.equal(layerDataType(undefined), 'Basemap & imagery')
  // A layer with a legend of an unexpected shape is still a raster, not imagery.
  assert.equal(layerDataType({ type: 'mystery' }), 'Other raster')
})
