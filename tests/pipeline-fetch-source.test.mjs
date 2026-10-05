// WANT-16: after a fetch, the source must become the saved dataset.
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const src = readFileSync(new URL('../pages/pipeline.vue', import.meta.url), 'utf8')

test('runFetch switches the source to the fetched dataset', () => {
  const body = src.slice(src.indexOf('async function runFetch'), src.indexOf('function advanceSource'))
  assert.match(body, /sourceForm\.datasetSlug = saved\.dataset\.slug\s*\n\s*sourceForm\.type = 'dataset'/)
})
