import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const src = readFileSync(new URL('../error.vue', import.meta.url), 'utf8')

test('error page is branded and handles 404', () => {
  assert.match(src, /Nexstrata/)
  assert.match(src, /Page not found/)
  assert.match(src, /--accent: #34c46a/)
})
