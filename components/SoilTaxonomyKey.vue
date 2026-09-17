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

      <ul class="stk-list">
        <li v-for="c in shown.slice(0, CAP)" :key="c.code" class="stk-item"
            :class="{ open: openName === c.name, flagged: flaggedSet.has(c.name) }">
          <button type="button" class="stk-row" :aria-expanded="String(openName === c.name)"
                  :title="tipFor(c)"
                  @click="openName = openName === c.name ? '' : c.name">
            <span class="swatch" :style="{ background: c.color }"></span>
            <span class="stk-name">{{ c.name }}</span>
            <!-- The one mark worth carrying in the row: it is why somebody is
                 looking at this list at all. -->
            <span v-if="flaggedSet.has(c.name)" class="stk-flag" title="Flagged as matsutake ground">🍄</span>
            <span class="stk-order-name">{{ c.order || 'unplaced' }}</span>
          </button>

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

const props = defineProps({
  // The layer key, e.g. 'soil-taxonomy'. Also the cache key: two layers over
  // the same asset share one fetch.
  layer: { type: String, required: true },
})

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
.stk-item.flagged .stk-name { color: var(--text-strong); font-weight: 600; }

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
