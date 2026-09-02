import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  OPTION_DOCS, docAnchor, docFor, docGroups, docHref, docSummary,
} from '../composables/optionDocs.js'

test('every entry has the parts a tooltip and the guide both need', () => {
  for (const doc of OPTION_DOCS) {
    assert.ok(doc.id, 'missing id')
    assert.match(doc.id, /^[a-z0-9-]+$/, `${doc.id} is not anchor-safe`)
    assert.ok(doc.group, `${doc.id} has no group`)
    assert.ok(doc.title, `${doc.id} has no title`)
    assert.ok(doc.summary, `${doc.id} has no summary`)
    assert.ok(Array.isArray(doc.detail) && doc.detail.length, `${doc.id} has no detail`)
  }
})

test('ids are unique — two entries cannot claim one anchor', () => {
  const ids = OPTION_DOCS.map((d) => d.id)
  assert.equal(new Set(ids).size, ids.length)
})

test('a summary is one sentence, short enough for a tooltip', () => {
  for (const doc of OPTION_DOCS) {
    assert.ok(doc.summary.length <= 130, `${doc.id}: summary is ${doc.summary.length} chars`)
    assert.ok(!doc.summary.includes('\n'), `${doc.id}: summary spans lines`)
  }
})

test('every cross-reference points at an entry that exists', () => {
  for (const doc of OPTION_DOCS) {
    for (const ref of doc.also || []) {
      assert.ok(docFor(ref), `${doc.id} links to unknown option "${ref}"`)
      assert.notEqual(ref, doc.id, `${doc.id} links to itself`)
    }
  }
})

test('lookups answer for known ids and stay quiet for unknown ones', () => {
  const first = OPTION_DOCS[0]
  assert.equal(docFor(first.id).title, first.title)
  assert.equal(docSummary(first.id), first.summary)
  assert.equal(docHref(first.id), `/guide#${docAnchor(first.id)}`)
  assert.equal(docFor('nope'), null)
  assert.equal(docSummary('nope'), '')
})

test('grouping keeps every entry, once, in declaration order', () => {
  const groups = docGroups()
  const flat = groups.flatMap((g) => g.items)
  assert.equal(flat.length, OPTION_DOCS.length)
  assert.deepEqual(flat.map((d) => d.id), OPTION_DOCS.map((d) => d.id))
  // Each group is named once rather than reappearing further down.
  const names = groups.map((g) => g.name)
  assert.equal(new Set(names).size, names.length)
})

test('reference anchors are namespaced away from the prose headings', () => {
  // The guide renders its prose and this reference on one page, and both anchor
  // their headings. "Coverage" is a section of the prose AND an option here, so
  // without a prefix the two claim the same #coverage — duplicate ids, and a
  // tooltip that scrolls to the wrong thing.
  for (const doc of OPTION_DOCS) {
    assert.equal(docAnchor(doc.id), `opt-${doc.id}`)
    assert.notEqual(docAnchor(doc.id), doc.id)
    assert.ok(docHref(doc.id).endsWith(`#opt-${doc.id}`))
  }
})
