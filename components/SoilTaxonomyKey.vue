<template>
  <div class="stk">
    <p v-if="error" class="stk-msg error">{{ error }}</p>
    <p v-else-if="loading" class="stk-msg">Reading the soil classes…</p>

    <template v-else>
      <!-- The twelve orders are the key, because the twelve orders are what is
           painted. Each one is also a control: click it to see only its own
           great groups, which is the question "what are all the podzols here"
           and the one a search box cannot answer, because no podzol's name
           contains the word. -->
      <div class="stk-orders">
        <button v-for="o in orders" :key="o.key" type="button" class="stk-order"
                :class="{ on: orderKey === o.key }" :aria-pressed="String(orderKey === o.key)"
                :title="`${o.summary} ${o.where}`"
                @click="orderKey = orderKey === o.key ? '' : o.key">
          <span class="swatch" :style="{ background: o.color }"></span>{{ o.name }}
        </button>
      </div>

      <div class="stk-find">
        <input v-model="query" type="search" class="stk-search"
               placeholder="Search soil classes" aria-label="Search soil classes" />
        <a class="stk-wiki" :href="wiki" target="_blank" rel="noopener noreferrer"
           title="USDA soil taxonomy on Wikipedia">What is this?</a>
      </div>

      <p class="stk-count">
        {{ shown.length.toLocaleString() }} of {{ classes.length.toLocaleString() }} great groups
        <button v-if="query || orderKey" type="button" class="linkish" @click="clear">Clear</button>
      </p>

      <!-- Selecting: what is chosen, and the bulk actions on what is shown.
           "All" acts on the filtered list rather than on four hundred classes,
           which is what makes search and the order chips into a selection tool
           — filter to Spodosols, press All, and you have chosen the podzols. -->
      <div v-if="selectable" class="stk-sel">
        <span class="stk-sel-n" :class="{ none: !selected.length }">
          {{ selected.length ? `${selected.length} chosen` : 'None chosen' }}
        </span>
        <button type="button" class="linkish" :disabled="!shown.length" @click="pickShown">
          {{ shownAllChosen ? 'Unpick these' : `Pick these ${shown.length}` }}
        </button>
        <button v-if="selected.length" type="button" class="linkish" @click="emitCodes([])">None</button>
        <button v-if="flagged.length" type="button" class="linkish"
                :title="`A starting selection: ${flagged.length} great groups FRMS members flagged as matsutake ground`"
                @click="pickFlagged">FRMS set</button>
      </div>
      <p v-if="selectable && !selected.length" class="stk-msg">
        Nothing is chosen, so the layer draws nothing. Tick a class below.
      </p>
      <!-- Only reachable on a raster with thousands of classes, which this one
           is not. Kept because a selection silently missing its last few
           hundred draws a map that is wrong in a way nobody can see. -->
      <p v-if="limitHit" class="stk-msg error">
        At most {{ CODE_LIMIT.toLocaleString() }} classes at a time. None of the
        extra ones were added.
      </p>

      <ul class="stk-list">
        <li v-for="c in shown.slice(0, CAP)" :key="c.code" class="stk-item"
            :class="{ open: openName === c.name, chosen: chosenSet.has(c.code) }">
          <div class="stk-row-wrap">
            <input v-if="selectable" type="checkbox" class="stk-tick"
                   :checked="chosenSet.has(c.code)" :aria-label="`Draw ${c.name}`"
                   @change="toggleCode(c.code)" />
            <button type="button" class="stk-row" :aria-expanded="String(openName === c.name)"
                    :title="tipFor(c)"
                    @click="openName = openName === c.name ? '' : c.name">
              <span class="swatch" :style="{ background: c.color }"></span>
              <span class="stk-name">{{ c.name }}</span>
              <span v-if="flaggedSet.has(c.name)" class="stk-flag"
                    title="In the FRMS matsutake set">🍄</span>
              <span class="stk-order-name">{{ c.order || 'unplaced' }}</span>
            </button>
          </div>

          <!-- Hover gives the one-line version through the title; this is the
               whole of it, for a touch screen and for anyone who wants the
               parts of the name spelled out. -->
          <div v-if="openName === c.name" class="stk-detail">
            <p v-if="c.summary" class="stk-sum">{{ c.summary }}</p>
            <p v-if="c.where" class="stk-where">{{ c.where }}</p>
            <ul v-if="c.elements.length" class="stk-parts">
              <li v-for="e in c.elements" :key="e.element">
                <code>{{ e.element }}-</code> {{ e.meaning }}
              </li>
            </ul>
            <p v-if="!c.known" class="stk-sum">
              This class does not carry one of the twelve order endings, so the map
              leaves it blank rather than guessing where it belongs.
            </p>
            <a :href="c.wiki" target="_blank" rel="noopener noreferrer" class="stk-link">
              {{ c.order || 'USDA soil taxonomy' }} on Wikipedia ↗
            </a>
          </div>
        </li>
      </ul>

      <p v-if="shown.length > CAP" class="stk-more">
        {{ (shown.length - CAP).toLocaleString() }} more. Narrow the search to see them.
      </p>
      <p v-else-if="!shown.length" class="stk-msg">Nothing matches that.</p>
    </template>
  </div>
</template>

<script setup>
// A browser for the soil taxonomy classes, in place of a key.
//
// The great-group raster has about four hundred classes. A key with four
// hundred rows is not a key — it is a lookup table you have to read line by
// line, and it would be taller than the map. So what the map paints is the
// twelve orders, and this panel is how you find out what the class under your
// cursor actually is.
//
// The descriptions are assembled on the server, from the name, by the rules in
// netlify/lib/soil-taxonomy.mjs. Nothing here knows how to read a soil name,
// and that is on purpose: the same rules pick the colour the map painted with,
// and a second copy of them in the browser is a second copy to disagree.

import { computed, ref, watch } from 'vue'
import { CODE_LIMIT, codeList, normaliseCodes } from '~/netlify/lib/ee-tile-layers.mjs'

const props = defineProps({
  // The layer key, e.g. 'soil-taxonomy'. Also the cache key: two layers over
  // the same asset share one fetch.
  layer: { type: String, required: true },
  // Whether this layer draws a chosen subset rather than everything. When it
  // does, the list gains a checkbox per class and the bulk actions above it.
  selectable: { type: Boolean, default: false },
  // The current selection, as the canonical comma-separated string the layer's
  // `codes` parameter carries.
  codes: { type: String, default: '' },
})
const emit = defineEmits(['update:codes'])

/**
 * How many rows are rendered at once.
 *
 * Four hundred rows of a flex list inside a floating panel is a scroll nobody
 * finishes and a paint nobody asked for. The count above the list always says
 * how many matched, so the cap never hides the size of the answer.
 */
const CAP = 60

const loading = ref(true)
const error = ref('')
const classes = ref([])
const orders = ref([])
const flagged = ref([])
const wiki = ref('https://en.wikipedia.org/wiki/USDA_soil_taxonomy')

const query = ref('')
const orderKey = ref('')
const openName = ref('')

const flaggedSet = computed(() => new Set(flagged.value))

// The table is static — it is a property of a published asset — so useEeTiles
// caches it for the life of the page and every component that asks gets the
// same one.
const eeTiles = useEeTiles()

async function loadClasses(key) {
  if (!key) return
  loading.value = true
  error.value = ''
  try {
    apply(await eeTiles.classes(key))
  } catch (err) {
    // Said rather than swallowed: without it the panel is an empty box, and an
    // empty box below a painted map reads as "this ground has no soil classes".
    error.value = String(err?.message || err)
    loading.value = false
  }
}

function apply(body) {
  classes.value = body.classes || []
  orders.value = body.orders || []
  flagged.value = body.flagged || []
  if (body.wiki) wiki.value = body.wiki
  loading.value = false
}

watch(() => props.layer, (key) => loadClasses(key), { immediate: true })

const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  return classes.value.filter((c) => {
    if (orderKey.value && c.orderKey !== orderKey.value) return false
    if (!q) return true
    // Name or order, so "spodosol" finds the podzols. The server already
    // resolved each class's order, so this is a comparison rather than a second
    // implementation of the parsing.
    return c.name.toLowerCase().includes(q)
      || (c.order || '').toLowerCase().includes(q)
      || (c.orderKey || '').includes(q)
  })
})

// ── Selecting ────────────────────────────────────────────────────────────────

const limitHit = ref(false)

const selected = computed(() => codeList(props.codes))
const chosenSet = computed(() => new Set(selected.value))
const shownAllChosen = computed(() => shown.value.length > 0
  && shown.value.every((c) => chosenSet.value.has(c.code)))

/**
 * Hand a new selection up, normalised the same way the server will normalise it.
 *
 * Over the limit is reported rather than silently trimmed: a selection quietly
 * missing its last forty classes draws a map that is wrong in a way nobody can
 * see, which is the failure this app works hardest to avoid.
 */
function emitCodes(codes) {
  try {
    limitHit.value = false
    emit('update:codes', normaliseCodes(codes))
  } catch {
    limitHit.value = true
  }
}

function toggleCode(code) {
  const next = new Set(selected.value)
  if (next.has(code)) next.delete(code)
  else next.add(code)
  emitCodes([...next])
}

/** Every class the current search and order filter leaves, added or removed. */
function pickShown() {
  const next = new Set(selected.value)
  const visible = shown.value.map((c) => c.code)
  if (shownAllChosen.value) visible.forEach((c) => next.delete(c))
  else visible.forEach((c) => next.add(c))
  emitCodes([...next])
}

function pickFlagged() {
  const byName = new Map(classes.value.map((c) => [c.name, c.code]))
  emitCodes(flagged.value.map((n) => byName.get(n)).filter((c) => c !== undefined))
}

function tipFor(c) {
  const parts = [c.order && `${c.order}.`, c.summary]
  if (c.elements?.length) parts.push(c.elements.map((e) => `${e.element}- ${e.meaning}`).join('; '))
  return parts.filter(Boolean).join(' ')
}

function clear() {
  query.value = ''
  orderKey.value = ''
}
</script>

<style scoped>
.stk { display: grid; gap: 7px; margin-top: 4px; }

.stk-msg { margin: 0; font-size: 0.72rem; color: var(--muted); }
.stk-msg.error { color: #e0714f; }

.stk-orders { display: flex; flex-wrap: wrap; gap: 3px; }
.stk-order {
  display: inline-flex; align-items: center; gap: 4px;
  border: 1px solid transparent; background: none; color: var(--muted);
  border-radius: 4px; padding: 1px 5px 1px 2px; cursor: pointer;
  font: inherit; font-size: 0.68rem; line-height: 1.5;
}
.stk-order:hover { color: var(--text); background: var(--surface-2); }
.stk-order.on { color: var(--text-strong); border-color: var(--muted); background: var(--surface-2); }

.swatch {
  width: 10px; height: 10px; border-radius: 2px; flex: 0 0 auto;
  border: 1px solid rgba(127, 127, 127, 0.5);
}

.stk-find { display: flex; align-items: center; gap: 6px; }
.stk-search {
  flex: 1 1 auto; min-width: 0; box-sizing: border-box;
  background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 5px;
  padding: 3px 6px; font: inherit; font-size: 0.72rem;
}
.stk-wiki { flex: 0 0 auto; font-size: 0.68rem; color: var(--accent); text-decoration: none; }
.stk-wiki:hover { text-decoration: underline; }

.stk-count {
  margin: 0; font-size: 0.66rem; color: var(--muted);
  display: flex; align-items: center; gap: 6px;
}
.linkish {
  background: none; border: 0; padding: 0; cursor: pointer;
  font: inherit; font-size: 0.66rem; color: var(--accent); text-decoration: underline;
}

/* Its own scroll, so a long list does not push the rest of the key off the
   bottom of the map. */
.stk-list {
  list-style: none; margin: 0; padding: 0;
  max-height: 148px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 1px;
}
.stk-item.chosen .stk-name { color: var(--text-strong); font-weight: 600; }

.stk-sel {
  display: flex; align-items: center; flex-wrap: wrap; gap: 4px 8px;
  padding: 4px 0 1px; border-top: 1px solid var(--border-soft, var(--border));
}
/* Its own line. The panel is about 260px wide and the three actions beside it
   are most of that, so on one row the count and the buttons overlap rather than
   wrapping — flex only wraps whole items, and "18 chosen" is one item. */
.stk-sel-n {
  flex: 1 1 100%; font-size: 0.68rem; color: var(--text-strong); font-weight: 700;
}
.stk-sel .linkish { flex: 0 0 auto; }
.stk-sel-n.none { color: var(--muted); font-weight: 400; }
.linkish:disabled { opacity: 0.4; cursor: default; text-decoration: none; }

.stk-row-wrap { display: flex; align-items: center; gap: 5px; }
.stk-tick { flex: 0 0 auto; margin: 0; accent-color: var(--accent, #2b7a3d); cursor: pointer; }

.stk-row {
  display: flex; align-items: center; gap: 6px; width: 100%; text-align: left;
  background: none; border: 0; border-radius: 4px; padding: 2px 4px; cursor: pointer;
  font: inherit; font-size: 0.72rem; color: var(--text);
}
.stk-row:hover { background: var(--surface-2); }
.stk-item.open .stk-row { background: var(--surface-2); }
.stk-name { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stk-flag { flex: 0 0 auto; font-size: 0.7rem; }
.stk-order-name { flex: 0 0 auto; font-size: 0.64rem; color: var(--muted); }

.stk-detail {
  padding: 4px 6px 7px 22px; font-size: 0.7rem; line-height: 1.5; color: var(--muted);
  display: grid; gap: 4px;
}
.stk-sum { margin: 0; color: var(--text); }
.stk-where { margin: 0; }
.stk-parts { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.stk-parts code {
  font: 0.9em/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--text-strong);
}
.stk-link { color: var(--accent); text-decoration: none; }
.stk-link:hover { text-decoration: underline; }

.stk-more { margin: 0; font-size: 0.66rem; color: var(--muted); }
</style>
