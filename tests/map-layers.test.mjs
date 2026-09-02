import test from 'node:test'
import assert from 'node:assert/strict'

import {
  TILE_LAYERS, TIME_LAYERS, WORLDCOVER_CLASSES,
  arcgisExportUrl, gibsUrl, layerDate, layerGroups, tileBounds,
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

test('WorldCover uses the product\'s own class colours', () => {
  assert.equal(WORLDCOVER_CLASSES.length, 11)
  assert.equal(WORLDCOVER_CLASSES[0].color, '#006400')  // tree cover
  assert.equal(WORLDCOVER_CLASSES[7].color, '#0064c8')  // permanent water
  assert.equal(new Set(WORLDCOVER_CLASSES.map((c) => c.color)).size, 11)
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
