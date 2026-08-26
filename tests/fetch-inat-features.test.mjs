import test from 'node:test'
import assert from 'node:assert/strict'

import { fetchInatFeatures } from '../netlify/lib/observations.mjs'

test('fetchInatFeatures paginates through all iNaturalist result pages', async () => {
  const calls = []
  const fetchImpl = async (url) => {
    const u = new URL(url)
    const page = Number(u.searchParams.get('page') ?? '1')
    calls.push(page)

    if (page === 1) {
      return {
        ok: true,
        json: async () => ({
          results: Array.from({ length: 200 }, (_, idx) => ({
            id: idx + 1,
            uuid: `u-${idx + 1}`,
            observed_on: '2024-05-01',
            taxon: { name: 'Amanita muscaria' },
            place_guess: 'Boulder',
            num_identification_agreements: 1,
            geojson: { coordinates: [-105, 40] },
          })),
        }),
      }
    }

    if (page === 2) {
      return {
        ok: true,
        json: async () => ({
          results: Array.from({ length: 50 }, (_, idx) => ({
            id: 200 + idx + 1,
            uuid: `u-${200 + idx + 1}`,
            observed_on: '2024-05-02',
            taxon: { name: 'Amanita muscaria' },
            place_guess: 'Boulder',
            num_identification_agreements: 1,
            geojson: { coordinates: [-105, 40] },
          })),
        }),
      }
    }

    return {
      ok: true,
      json: async () => ({ results: [] }),
    }
  }

  const features = await fetchInatFeatures({ taxonName: 'Amanita muscaria', perPage: 200 }, fetchImpl)

  assert.equal(features.length, 250)
  assert.deepEqual(calls, [1, 2])
})
