/* Rearranging cards by dragging one onto another.
 *
 * The arithmetic is the whole risk here. "Drop it where that one is" means two
 * different splices depending on which way the drag came from, and the classic
 * symptom of getting it wrong is a card that can never reach the end of the
 * list — which reads as a dead drop target rather than as an off-by-one.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { previewOrder, reorder, useDragReorder } from '../composables/useDragReorder.js'

const L = ['a', 'b', 'c', 'd', 'e']

// ── Direction ────────────────────────────────────────────────────────────────

test('dragging forwards lands on the target, pushing it back', () => {
  assert.deepEqual(reorder(L, 'a', 'c'), ['b', 'c', 'a', 'd', 'e'])
  assert.deepEqual(reorder(L, 'b', 'd'), ['a', 'c', 'd', 'b', 'e'])
})

test('dragging backwards lands in front of the target', () => {
  assert.deepEqual(reorder(L, 'd', 'b'), ['a', 'd', 'b', 'c', 'e'])
  assert.deepEqual(reorder(L, 'e', 'a'), ['e', 'a', 'b', 'c', 'd'])
})

test('a card can reach either end', () => {
  // The off-by-one this exists to catch: dropping on the last card has to put
  // the dragged one LAST, not second-to-last.
  assert.deepEqual(reorder(L, 'a', 'e'), ['b', 'c', 'd', 'e', 'a'])
  assert.equal(reorder(L, 'a', 'e').at(-1), 'a')
  assert.deepEqual(reorder(L, 'e', 'a'), ['e', 'a', 'b', 'c', 'd'])
  assert.equal(reorder(L, 'e', 'a')[0], 'e')
})

test('neighbours swap', () => {
  assert.deepEqual(reorder(L, 'b', 'c'), ['a', 'c', 'b', 'd', 'e'])
  assert.deepEqual(reorder(L, 'c', 'b'), ['a', 'c', 'b', 'd', 'e'])
})

// ── Invariants ───────────────────────────────────────────────────────────────

test('every reorder is a permutation: nothing lost, nothing duplicated', () => {
  for (const from of L) {
    for (const to of L) {
      const out = reorder(L, from, to)
      assert.equal(out.length, L.length, `${from}->${to} changed the length`)
      assert.deepEqual([...out].sort(), [...L].sort(), `${from}->${to} lost or duplicated`)
    }
  }
})

test('dropping a card on itself changes nothing', () => {
  for (const id of L) assert.deepEqual(reorder(L, id, id), L)
})

test('the source list is never mutated', () => {
  const original = [...L]
  reorder(L, 'a', 'e')
  reorder(L, 'e', 'a')
  assert.deepEqual(L, original)
})

test('an unknown id moves nothing rather than inserting at the front', () => {
  // Returning early matters: splice(-1) or indexOf(-1) would silently move
  // whatever happened to be at that index.
  assert.deepEqual(reorder(L, 'zz', 'c'), L)
  assert.deepEqual(reorder(L, 'a', 'zz'), L)
  assert.deepEqual(reorder(L, 'zz', 'yy'), L)
})

test('degenerate lists do not throw', () => {
  assert.deepEqual(reorder([], 'a', 'b'), [])
  assert.deepEqual(reorder(['a'], 'a', 'a'), ['a'])
})

// ── Round trips ──────────────────────────────────────────────────────────────

test('dragging a card away and back restores the order', () => {
  // Not true of every reorder scheme, and worth pinning: it is what makes a
  // drag feel undoable rather than lossy.
  const away = reorder(L, 'b', 'e')
  assert.deepEqual(reorder(away, 'b', 'a'), ['b', 'a', 'c', 'd', 'e'])
  const back = reorder(reorder(L, 'a', 'b'), 'a', 'a')
  assert.deepEqual(back, ['b', 'a', 'c', 'd', 'e'])
})

test('repeated drags compose without drift', () => {
  let cur = [...L]
  for (const [f, t] of [['a', 'c'], ['e', 'b'], ['d', 'a'], ['c', 'e']]) {
    cur = reorder(cur, f, t)
    assert.deepEqual([...cur].sort(), [...L].sort())
  }
  assert.equal(cur.length, L.length)
})

// ── The preview ──────────────────────────────────────────────────────────────

test('a preview with nothing being dragged is the list itself', () => {
  assert.deepEqual(previewOrder(L, '', ''), L)
  assert.deepEqual(previewOrder(L, 'a', ''), L)
  assert.deepEqual(previewOrder(L, '', 'a'), L)
  assert.deepEqual(previewOrder(L, null, null), L)
})

test('a preview mid-drag shows where the card would land', () => {
  assert.deepEqual(previewOrder(L, 'a', 'd'), reorder(L, 'a', 'd'))
})

test('the preview never mutates the committed order', () => {
  // The drag is abandoned by dropping the preview, not by undoing a change, so
  // the committed list has to be untouched until the drop.
  const committed = [...L]
  previewOrder(committed, 'a', 'e')
  assert.deepEqual(committed, L)
})

// ── The drag controller ──────────────────────────────────────────────────────

/** Drive a controller the way the browser does, recording what it commits. */
function controller(list) {
  const state = { ids: [...list], committed: null, paused: 0 }
  const c = useDragReorder({
    key: `test-${Math.random()}`,
    ids: () => state.ids,
    onCommit: (next) => { state.committed = next; state.ids = next },
    onPause: () => { state.paused += 1 },
    onResume: () => { state.paused -= 1 },
  })
  return { c, state }
}

test('a plain drag commits the new order once', () => {
  const { c, state } = controller(L)
  c.start('a')
  c.enter('c')
  c.end()
  assert.deepEqual(state.committed, ['b', 'c', 'a', 'd', 'e'])
})

test('the preview moving the card under the pointer does not cancel the drop', () => {
  // The bug this exists for, as the browser actually reported it:
  //   dragstart card0 → dragenter card3 → dragenter card0 → drop
  // The preview slides the dragged card to position 3, which puts it under the
  // cursor, so the browser fires dragenter on the dragged card itself. Taken at
  // face value that makes source === target, and the drop commits nothing — a
  // drag that looks right and does nothing.
  const { c, state } = controller(L)
  c.start('a')
  c.enter('d')
  c.enter('a')          // the browser's self-enter
  c.end()
  assert.deepEqual(state.committed, ['b', 'c', 'd', 'a', 'e'],
    'the self-enter should have been ignored')
})

test('dropping without moving commits nothing', () => {
  const { c, state } = controller(L)
  c.start('a')
  c.end()
  assert.equal(state.committed, null)
})

test('cancelling commits nothing and releases the pause', () => {
  const { c, state } = controller(L)
  c.start('a')
  c.enter('e')
  c.cancel()
  assert.equal(state.committed, null)
  assert.equal(state.paused, 0)
  assert.equal(c.dragging.value, '')
})

test('rendering is paused for exactly the duration of the drag', () => {
  const { c, state } = controller(L)
  assert.equal(state.paused, 0)
  c.start('a')
  assert.equal(state.paused, 1, 'a drag should pause redraws')
  c.enter('c')
  assert.equal(state.paused, 1, 'moving should not pause again')
  c.end()
  assert.equal(state.paused, 0, 'the drop should release it')
})

test('drop and dragend both firing commits only once', () => {
  // The browser sends both, and both handlers call end().
  const { c, state } = controller(L)
  c.start('a')
  c.enter('c')
  c.end()
  const first = state.committed
  state.committed = null
  c.end()
  assert.deepEqual(first, ['b', 'c', 'a', 'd', 'e'])
  assert.equal(state.committed, null, 'the second end() should be a no-op')
  assert.equal(state.paused, 0)
})

test('the position shown mid-drag is the preview, and settles on commit', () => {
  const { c, state } = controller(L)
  c.start('a')
  c.enter('c')
  assert.deepEqual(c.shown.value, ['b', 'c', 'a', 'd', 'e'])
  assert.equal(c.orderOf('a'), 2)
  c.end()
  assert.deepEqual(c.shown.value, state.ids)
  assert.equal(c.orderOf('a'), 2)
})

test('two lists drag independently', () => {
  const one = controller(['a', 'b'])
  const two = controller(['x', 'y'])
  one.c.start('a')
  assert.equal(two.c.dragging.value, '', 'a drag in one list must not start one in the other')
  one.c.end()
})
