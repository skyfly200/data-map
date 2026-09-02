// Keyboard shortcuts, and the single source of truth for what they are.
//
// Registering a shortcut also documents it: the help overlay and the tooltips on
// the controls both read this registry, so a shortcut cannot exist without being
// discoverable, and a tooltip cannot claim a key that no longer works.
//
// Global shortcuts are registered once in app.vue. A page or component registers
// its own on mount and drops them on unmount, so pressing "o" on the Charts page
// does nothing rather than reaching for a map control that is not on screen.

import { computed, onUnmounted, ref, shallowRef } from 'vue'

// Module-level, deliberately not `useState`: entries carry `run` callbacks, and
// anything in `useState` is serialized into the SSR payload, where a function
// cannot be stringified. Shortcuts are client-only behaviour with nothing worth
// carrying across hydration, so a plain module ref is both correct and simpler.
const registry = shallowRef([])
const helpOpen = ref(false)

// Input types that consume keystrokes themselves: text entry, and the ones the
// browser drives with arrows and Home/End. A shortcut must not eat those.
const KEYBOARD_INPUTS = new Set([
  'text', 'search', 'url', 'tel', 'email', 'password', 'number',
  'date', 'datetime-local', 'month', 'week', 'time', 'range',
])

/**
 * Fields where a keystroke is text, not a command.
 *
 * Type matters, not just the tag: a focused checkbox is not typing, and treating
 * it as such means ticking a box in a panel silently disables every shortcut
 * until you click elsewhere.
 */
export function isTyping(target) {
  if (!target) return false
  if (target.isContentEditable) return true
  const tag = target.tagName
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true
  if (tag !== 'INPUT') return false
  // A missing or unknown type is a text field, per the HTML spec's default.
  const type = (target.getAttribute?.('type') || target.type || 'text').toLowerCase()
  return KEYBOARD_INPUTS.has(type) || !KNOWN_NON_TEXT.has(type)
}

// Everything else an <input> can be: none of these are text entry.
const KNOWN_NON_TEXT = new Set([
  'checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'color', 'image', 'hidden',
])

/** "shift+/" → "?" and friends, for display. */
export function prettyKey(keys) {
  return keys
    .split('+')
    .map((k) => ({
      arrowleft: '←', arrowright: '→', arrowup: '↑', arrowdown: '↓',
      escape: 'Esc', ' ': 'Space',
    }[k.toLowerCase()] || (k.length === 1 ? k.toUpperCase() : k)))
    .join(' + ')
}

/**
 * The combination an event represents, in the same form a registration uses.
 * Exported so the shift rules can be tested — they are subtle enough that a
 * dead shortcut sat in the map for weeks without anyone noticing.
 */
export function eventKeys(e) {
  const parts = []
  if (e.ctrlKey) parts.push('ctrl')
  if (e.metaKey) parts.push('meta')
  if (e.altKey) parts.push('alt')
  // Shift is recorded for letters and for named keys, but NOT for other
  // printable characters: the browser has already folded shift into those, so
  // "?" arrives as "?" and demanding "shift+?" would never match. A letter is
  // the opposite case — "O" and "o" read as the same registration, so without
  // recording shift here "shift+O" could never match either, which is exactly
  // what had happened to the map's "previous overlay" shortcut.
  const printable = e.key.length === 1
  const isLetter = printable && /[a-z]/i.test(e.key)
  if (e.shiftKey && (!printable || isLetter)) parts.push('shift')
  parts.push(e.key.toLowerCase())
  return parts.join('+')
}

export function useShortcuts() {
  /**
   * Add shortcuts, and remove them again when the caller goes away.
   *
   * `keys` is lower-case and may carry modifiers: "m", "?", "shift+p".
   * `scope` groups them in the help overlay.
   */
  function register(items) {
    const entries = items.map((i) => {
      const keys = String(i.keys).toLowerCase()
      return { ...i, keys, id: `${i.scope}:${keys}` }
    })
    const ids = new Set(entries.map((e) => e.id))
    registry.value = [...registry.value.filter((e) => !ids.has(e.id)), ...entries]

    const unregister = () => {
      registry.value = registry.value.filter((e) => !ids.has(e.id))
    }
    // Only auto-clean when called from a component; app-level registration has
    // no instance to hook onto.
    try { onUnmounted(unregister) } catch { /* not in a component */ }
    return unregister
  }

  function handle(e) {
    // Escape is the one key that must work while typing — it is how you get out.
    if (isTyping(e.target) && e.key !== 'Escape') return
    const combo = eventKeys(e)
    // Later registrations win, so a page shortcut can shadow a global one.
    const match = [...registry.value].reverse().find((s) => s.keys === combo)
    if (!match) return
    e.preventDefault()
    match.run(e)
  }

  const grouped = computed(() => {
    const byScope = new Map()
    for (const s of registry.value) {
      if (!byScope.has(s.scope)) byScope.set(s.scope, [])
      byScope.get(s.scope).push(s)
    }
    return [...byScope.entries()].map(([scope, items]) => ({ scope, items }))
  })

  /** The key for an action, for a tooltip — empty when it has none. */
  function keyFor(scope, label) {
    return registry.value.find((s) => s.scope === scope && s.label === label)?.keys || ''
  }

  /** "Show my location" → "Show my location (L)". */
  function withKey(text, keys) {
    return keys ? `${text}  (${prettyKey(keys)})` : text
  }

  return { registry, grouped, helpOpen, register, handle, keyFor, withKey, prettyKey }
}
