<template>
  <div class="tx">
    <!-- Ready-made lists first. Most people want one of these and nothing
         else, and the difference between them is four orders of magnitude of
         records — so each says what it costs rather than leaving that to a
         footnote nobody reads. -->
    <div class="tx-presets">
      <button v-for="p in TAXON_PRESETS" :key="p.key" type="button"
              class="tx-preset" :class="{ on: preset?.key === p.key }"
              :aria-pressed="String(preset?.key === p.key)"
              :title="p.note" @click="setTaxa(p.taxa)">
        <strong>{{ p.label }}</strong>
        <em>{{ p.scale }}</em>
      </button>
    </div>

    <p v-if="scale !== 'ordinary'" class="tx-warn" :class="scale">
      <strong>{{ scale === 'huge' ? 'This is a very large request.' : 'This is a large request.' }}</strong>
      A fetch from the browser handles one taxon at a time and will take a long
      while. For a list this size, set it on the pipeline and run that instead.
    </p>

    <div class="tx-add">
      <input v-model="draft" type="text" :placeholder="placeholder"
             aria-label="Add a taxon" @keyup.enter="add" />
      <button class="btn small" :disabled="!draft.trim()" @click="add">Add</button>
    </div>
    <p v-if="addError" class="tx-err">{{ addError }}</p>

    <p class="tx-count">
      <strong>{{ taxa.length }}</strong>
      {{ taxa.length === 1 ? 'taxon' : 'taxa' }}
      <span v-if="preset" class="tx-is">· {{ preset.label }}</span>
      <button type="button" class="linkish" @click="reset">Back to the FRMS genera</button>
    </p>

    <ul v-if="taxa.length" class="tx-list">
      <li v-for="t in taxa" :key="t" class="tx-chip">
        <span>{{ t }}</span>
        <button type="button" :aria-label="`Remove ${t}`" :title="`Remove ${t}`"
                @click="removeTaxon(t)">×</button>
      </li>
    </ul>
    <p v-else class="tx-empty">
      Nothing in the list. Add a taxon above, or choose a ready-made list.
    </p>

    <!-- The list has to reach the Python pipeline to be run at scale, and the
         pipeline reads an environment variable. Rather than describe the
         format and let somebody transcribe forty genera by hand, write the
         line. -->
    <details class="tx-env">
      <summary>For the pipeline</summary>
      <p class="tx-note">
        The browser fetches one taxon at a time. To collect this list properly,
        put the line below in the pipeline's <code>.env</code> and run it.
      </p>
      <div class="tx-env-row">
        <code>{{ envLine }}</code>
        <button class="btn small" @click="copyEnv">{{ copied ? 'Copied' : 'Copy' }}</button>
      </div>
    </details>
  </div>
</template>

<script setup>
// Editing the list of taxa the app asks iNaturalist for.
//
// The list used to be a comma-separated line in a .env file, which meant two
// things: you had to find the file to change it, and the app could not tell you
// what it was set to. It is a setting now, and this is its editor.

import { computed, ref } from 'vue'

import { TAXON_PRESETS, isTaxonName, asEnvLine, matchingPreset, scaleOf } from '~/composables/taxonList'

const { taxa, setTaxa, addTaxon, removeTaxon, reset } = useTaxonList()

const draft = ref('')
const addError = ref('')
const copied = ref(false)

const preset = computed(() => matchingPreset(taxa.value))
const scale = computed(() => scaleOf(taxa.value))
const envLine = computed(() => asEnvLine(taxa.value))

const placeholder = 'Amanita, Amanitaceae, Fungi…'

function add() {
  const name = draft.value.trim()
  addError.value = ''
  if (!name) return
  if (!isTaxonName(name)) {
    // Said rather than swallowed. A name silently refused looks like a broken
    // button, and the usual cause is a stray character nobody can see.
    addError.value = `"${name}" does not look like a taxon name. Letters, spaces and hyphens only.`
    return
  }
  if (!addTaxon(name)) {
    addError.value = `${name} is already in the list.`
    return
  }
  draft.value = ''
}

async function copyEnv() {
  try {
    await navigator.clipboard.writeText(envLine.value)
    copied.value = true
    setTimeout(() => { copied.value = false }, 1500)
  } catch { /* the line is on screen to select by hand */ }
}
</script>

<style scoped>
.tx { display: grid; gap: 10px; }

.tx-presets { display: flex; flex-wrap: wrap; gap: 6px; }
.tx-preset {
  display: grid; gap: 1px; text-align: left; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 7px; padding: 6px 10px; font: inherit;
}
.tx-preset:hover { border-color: var(--muted); }
.tx-preset.on { border-color: var(--accent); background: var(--surface-3, var(--surface-2)); }
.tx-preset strong { font-size: 0.84rem; }
.tx-preset em { font-size: 0.7rem; color: var(--muted); font-style: normal; }

.tx-warn {
  margin: 0; padding: 8px 12px; border-radius: 6px; line-height: 1.55;
  font-size: 0.82rem; background: var(--surface-2);
  border-left: 3px solid #b3822f; color: var(--text);
}
.tx-warn.huge { border-left-color: #b3492f; }
.tx-warn strong { color: var(--text-strong); }

.tx-add { display: flex; gap: 6px; }
.tx-add input {
  flex: 1 1 auto; min-width: 0; box-sizing: border-box;
  background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 8px; font: inherit; font-size: 0.85rem;
}
.tx-err { margin: 0; font-size: 0.78rem; color: #e0714f; }

.tx-count {
  margin: 0; font-size: 0.78rem; color: var(--muted);
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.tx-count strong { color: var(--text-strong); }
.tx-is { color: var(--text); }
.linkish {
  background: none; border: 0; padding: 0; cursor: pointer;
  font: inherit; font-size: 0.78rem; color: var(--accent); text-decoration: underline;
}

.tx-list {
  list-style: none; margin: 0; padding: 0;
  display: flex; flex-wrap: wrap; gap: 4px;
  max-height: 168px; overflow-y: auto;
}
.tx-chip {
  display: inline-flex; align-items: center; gap: 5px;
  border: 1px solid var(--border); background: var(--surface-2);
  border-radius: 999px; padding: 2px 4px 2px 10px; font-size: 0.78rem;
}
.tx-chip button {
  border: 0; background: none; color: var(--muted); cursor: pointer;
  font-size: 0.85rem; line-height: 1; padding: 0 4px;
}
.tx-chip button:hover { color: #b3492f; }
.tx-empty { margin: 0; font-size: 0.8rem; color: var(--muted); }

.tx-env { font-size: 0.8rem; }
.tx-env summary { cursor: pointer; color: var(--accent); }
.tx-note { margin: 8px 0 6px; color: var(--muted); line-height: 1.55; }
.tx-env-row { display: flex; align-items: flex-start; gap: 8px; }
.tx-env-row code {
  flex: 1 1 auto; min-width: 0; overflow-x: auto; white-space: pre;
  font: 0.75rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--surface-2); border: 1px solid var(--border-soft, var(--border));
  border-radius: 5px; padding: 6px 8px;
}
</style>
