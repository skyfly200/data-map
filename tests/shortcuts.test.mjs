import { test } from 'node:test'
import assert from 'node:assert/strict'

import { isTyping, prettyKey } from '../composables/useShortcuts.js'

/** A stand-in for a focused element, close enough for isTyping. */
const el = (tagName, type) => ({
  tagName,
  type,
  getAttribute: (name) => (name === 'type' ? type ?? null : null),
})

test('text entry swallows keystrokes, so shortcuts stay out of it', () => {
  for (const type of ['text', 'search', 'email', 'password', 'number', 'url', 'tel', 'date', 'time']) {
    assert.equal(isTyping(el('INPUT', type)), true, `input[type=${type}]`)
  }
  assert.equal(isTyping(el('TEXTAREA')), true)
  assert.equal(isTyping(el('SELECT')), true)
  assert.equal(isTyping({ isContentEditable: true }), true)
})

test('an input with no type is a text field, per the HTML default', () => {
  assert.equal(isTyping({ tagName: 'INPUT', getAttribute: () => null }), true)
})

test('a range input is left alone — arrows and Home/End are its own controls', () => {
  assert.equal(isTyping(el('INPUT', 'range')), true)
})

test('a focused checkbox is not typing', () => {
  // Ticking a box in a settings panel used to disable every shortcut until you
  // clicked elsewhere, because the whole INPUT tag counted as text entry.
  for (const type of ['checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'color']) {
    assert.equal(isTyping(el('INPUT', type)), false, `input[type=${type}]`)
  }
})

test('ordinary elements never block a shortcut', () => {
  assert.equal(isTyping(el('DIV')), false)
  assert.equal(isTyping(el('BUTTON')), false)
  assert.equal(isTyping(el('A')), false)
  assert.equal(isTyping(null), false)
  assert.equal(isTyping(undefined), false)
})

test('keys are shown the way a keyboard shows them', () => {
  assert.equal(prettyKey('p'), 'P')
  assert.equal(prettyKey('?'), '?')
  assert.equal(prettyKey('escape'), 'Esc')
  assert.equal(prettyKey('arrowleft'), '←')
  assert.equal(prettyKey('ctrl+k'), 'ctrl + K')
})
