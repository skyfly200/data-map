import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  compassPoint, detailSections, indexFraction, missingEnrichment, rainLeadUp,
} from '../composables/observationDetail.js'

test('a bearing becomes a compass point, wrapping at north', () => {
  assert.equal(compassPoint(0), 'N')
  assert.equal(compassPoint(90), 'E')
  assert.equal(compassPoint(180), 'S')
  assert.equal(compassPoint(270), 'W')
  assert.equal(compassPoint(45), 'NE')
  assert.equal(compassPoint(213), 'SSW')
  // The wrap is the part worth pinning: 350° is north, not north-north-west.
  assert.equal(compassPoint(350), 'N')
  assert.equal(compassPoint(359.9), 'N')
  assert.equal(compassPoint(360), 'N')
  // And it survives values outside 0–360 rather than indexing off the end.
  assert.equal(compassPoint(-90), 'W')
  assert.equal(compassPoint(720), 'N')
})

test('a non-numeric bearing yields nothing rather than a wrong direction', () => {
  assert.equal(compassPoint(null), '')
  assert.equal(compassPoint(undefined), '')
  assert.equal(compassPoint('north'), '')
  assert.equal(compassPoint(NaN), '')
})

test('the rain total says how many days it actually covers', () => {
  const full = Object.fromEntries([0, 1, 2, 3, 4, 5, 6].map((o) => [`prcp_d${o}`, 2]))
  assert.deepEqual(rainLeadUp(full), { total: 14, days: 7 })
  // A partial week must not masquerade as a whole one.
  assert.deepEqual(rainLeadUp({ prcp_d0: 5, prcp_d3: 1 }), { total: 6, days: 2 })
  assert.equal(rainLeadUp({}), null)
  assert.equal(rainLeadUp({ prcp_d0: null, prcp_d1: '' }), null)
  // Zero rain is a reading, not a gap.
  assert.deepEqual(rainLeadUp({ prcp_d0: 0 }), { total: 0, days: 1 })
})

test('an index maps onto its own scale, clamped', () => {
  assert.equal(indexFraction(0.5), 0.5)
  assert.equal(indexFraction(0, [-1, 1]), 0.5)
  assert.equal(indexFraction(-1, [-1, 1]), 0)
  assert.equal(indexFraction(1, [-1, 1]), 1)
  assert.equal(indexFraction(5), 1)
  assert.equal(indexFraction(-5), 0)
  assert.equal(indexFraction('x'), null)
})

const ctx = {
  elevLabel: (v) => `${Math.round(v)} m`,
  tempLabel: (v) => `${Math.round(v)}°C`,
  precisionLabel: (k) => ({ precise: 'Precise', obscured: 'Obscured (~20km)' }[k] || k),
}
const find = (sections, label) =>
  sections.flatMap((s) => s.rows).find((r) => r.label === label)

test('sections carry only what the record actually has', () => {
  const sections = detailSections({ species: 'x', date: '2023-09-01' }, ctx)
  assert.deepEqual(sections.map((s) => s.title), ['Record'])
  assert.equal(find(sections, 'Observed').value, '2023-09-01')
  // Nothing empty is invented.
  assert.equal(find(sections, 'Slope'), undefined)
  assert.equal(detailSections(null, ctx).length, 0)
})

test('a full record groups into record, terrain and weather', () => {
  const sections = detailSections({
    date: '2023-09-01', day_of_year: 244, location: 'Boulder, CO',
    lat: 40.01234, lon: -105.56789, num_identification_agreements: 3,
    elevation: 2650, slope: 12.4, aspect: 213, land_cover_label: 'Tree cover',
    ndvi: 0.62, soil_moisture: 0.31, water_retention: 0.4,
    tmax: 21, tmin: 7, prcp_d0: 3, prcp_d1: 2,
  }, ctx)
  assert.deepEqual(sections.map((s) => s.title), ['Record', 'Terrain', 'Weather'])
  assert.equal(find(sections, 'Observed').value, '2023-09-01 · day 244')
  assert.equal(find(sections, 'Coordinates').value, '40.0123, -105.5679')
  assert.equal(find(sections, 'Agreements').value, '3 identifiers')
  assert.equal(find(sections, 'Faces').value, 'SSW · 213°')
  assert.equal(find(sections, 'Elevation').value, '2650 m')
  assert.equal(find(sections, 'High that day').value, '21°C')
  // A two-day total must announce itself as one.
  assert.match(find(sections, 'Rain, 7 days before').hint, /Only 2 of the 7 days/)
})

test('an unconfirmed identification is called out', () => {
  const one = detailSections({ num_identification_agreements: 1 }, ctx)
  assert.equal(find(one, 'Agreements').value, '1 identifier')
  assert.equal(find(one, 'Agreements').hint, null)
  const none = detailSections({ num_identification_agreements: 0 }, ctx)
  assert.match(find(none, 'Agreements').hint, /Nobody has confirmed/)
})

test('an obscured location is flagged as changing how the terrain reads', () => {
  const obscured = detailSections({ location_precision: 'obscured', slope: 10 }, ctx)
  const rowObscured = find(obscured, 'Precision')
  assert.equal(rowObscured.value, 'Obscured (~20km)')
  assert.equal(rowObscured.warn, true)
  assert.match(rowObscured.hint, /may have moved/)

  const precise = detailSections({ location_precision: 'precise' }, ctx)
  assert.equal(find(precise, 'Precision').warn, false)
  assert.equal(find(precise, 'Precision').hint, null)
})

test('missing enrichment is reported rather than left as a silent gap', () => {
  assert.deepEqual(missingEnrichment({}), ['terrain', 'satellite', 'weather'])
  assert.deepEqual(missingEnrichment({ slope: 3, ndvi: 0.4, prcp_d0: 1 }), [])
  assert.deepEqual(missingEnrichment({ aspect: 90, soil_moisture: 0.2, prcp_d6: 0 }), [])
  assert.deepEqual(missingEnrichment(null), [])
})
