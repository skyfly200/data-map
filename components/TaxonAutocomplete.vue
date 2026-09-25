<template>
  <div class="ac-wrap" ref="wrap">
    <input
      v-bind="$attrs"
      :value="modelValue"
      type="text"
      autocomplete="off"
      @input="onInput"
      @focus="showDropdown = true"
      @keydown="onKey"
      @keyup.enter="onEnter"
    />
    <span v-if="loading" class="ac-spinner" aria-hidden="true"></span>
    <ul v-if="showDropdown && suggestions.length" class="ac-list" role="listbox">
      <li
        v-for="(s, i) in suggestions" :key="s.name"
        class="ac-item" :class="{ active: i === activeIndex }"
        role="option" :aria-selected="i === activeIndex"
        @mousedown.prevent="pick(s)"
        @mousemove="activeIndex = i"
      >
        <span class="ac-sci">{{ s.name }}</span>
        <span v-if="s.common" class="ac-common">{{ s.common }}</span>
        <span v-if="s.rank" class="ac-rank">{{ s.rank }}</span>
      </li>
    </ul>
  </div>
</template>

<script setup>
const props = defineProps({
  modelValue: { type: String, default: '' },
  source: { type: String, default: 'auto' }, // 'auto' | 'inat' | 'gbif'
})
const emit = defineEmits(['update:modelValue'])

const wrap = ref(null)
const suggestions = ref([])
const loading = ref(false)
const activeIndex = ref(-1)
const showDropdown = ref(false)
let timer = null

function onInput(e) {
  const val = e.target.value
  emit('update:modelValue', val)
  clearTimeout(timer)
  if (!val.trim()) { suggestions.value = []; return }
  timer = setTimeout(() => query(val.trim()), 250)
}

async function query(q) {
  if (q.length < 2) { suggestions.value = []; return }
  loading.value = true
  try {
    const useGbif = props.source === 'gbif'
    if (useGbif) {
      const res = await fetch(`https://api.gbif.org/v1/species/suggest?q=${encodeURIComponent(q)}&limit=8`)
      if (!res.ok) return
      const data = await res.json()
      suggestions.value = data.map((r) => ({
        name: r.canonicalName || r.scientificName,
        common: r.vernacularName || '',
        rank: r.rank ? r.rank.toLowerCase() : '',
      })).filter((r) => r.name)
    } else {
      const res = await fetch(`https://api.inaturalist.org/v1/taxa/autocomplete?q=${encodeURIComponent(q)}&per_page=8`)
      if (!res.ok) return
      const data = await res.json()
      suggestions.value = (data.results || []).map((r) => ({
        name: r.name,
        common: r.preferred_common_name || '',
        rank: r.rank || '',
      }))
    }
    activeIndex.value = -1
    showDropdown.value = true
  } catch { /* ignore */ }
  finally { loading.value = false }
}

watch(() => props.source, () => {
  if (props.modelValue?.trim().length >= 2) query(props.modelValue.trim())
})

function pick(s) {
  emit('update:modelValue', s.name)
  suggestions.value = []
  showDropdown.value = false
  activeIndex.value = -1
}

function onKey(e) {
  if (!suggestions.value.length) return
  if (e.key === 'ArrowDown') { e.preventDefault(); activeIndex.value = Math.min(activeIndex.value + 1, suggestions.value.length - 1) }
  else if (e.key === 'ArrowUp') { e.preventDefault(); activeIndex.value = Math.max(activeIndex.value - 1, -1) }
  else if (e.key === 'Escape') { showDropdown.value = false }
}

function onEnter() {
  if (activeIndex.value >= 0 && suggestions.value[activeIndex.value]) pick(suggestions.value[activeIndex.value])
}

function onOutside(e) {
  if (wrap.value && !wrap.value.contains(e.target)) showDropdown.value = false
}
onMounted(() => document.addEventListener('mousedown', onOutside))
onBeforeUnmount(() => { document.removeEventListener('mousedown', onOutside); clearTimeout(timer) })
</script>

<style scoped>
.ac-wrap { position: relative; display: block; flex: 1; min-width: 0; }
.ac-list {
  position: absolute; top: calc(100% + 2px); left: 0; right: 0; z-index: 200;
  background: var(--surface); border: 1px solid var(--border); border-radius: 7px;
  list-style: none; margin: 0; padding: 4px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  max-height: 240px; overflow-y: auto;
}
.ac-item {
  display: flex; align-items: baseline; gap: 8px;
  padding: 6px 12px; cursor: pointer; font-size: 0.85rem;
}
.ac-item:hover, .ac-item.active { background: var(--surface-2, var(--border)); }
.ac-sci { font-style: italic; }
.ac-common { color: var(--muted); font-size: 0.78rem; }
.ac-rank { margin-left: auto; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
.ac-spinner {
  position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
  width: 12px; height: 12px; border: 2px solid var(--border); border-top-color: var(--accent, #34c46a);
  border-radius: 50%; animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: translateY(-50%) rotate(360deg); } }
</style>
