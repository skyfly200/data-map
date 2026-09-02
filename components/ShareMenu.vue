<template>
  <div class="share">
    <button class="sh-btn" :class="{ on: open, compact, 'icon-only': iconOnly }"
            :title="compact ? `Share “${title}”` : 'Share this view'"
            :aria-label="compact || iconOnly ? `Share “${title}”` : null" @click="toggle">
      <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        <path fill="currentColor" d="M18 16.1a3 3 0 0 0-2 .8l-7.1-4.2a3 3 0 0 0 0-1.4L16 7.1a3 3 0 1 0-1-2.1l-7.1 4.2a3 3 0 1 0 0 5.6L15 19a3 3 0 1 0 3-2.9z" />
      </svg>
      <template v-if="!compact && !iconOnly">Share</template>
    </button>

    <div v-if="open" class="sh-panel" :class="{ right: compact }">
      <p class="sh-note">{{ note }}</p>

      <div class="sh-url">
        <input ref="urlInput" :value="url" readonly aria-label="Shareable link" @focus="$event.target.select()" />
        <button class="sh-copy" @click="copy(url, 'link')">{{ copied === 'link' ? '✓' : 'Copy' }}</button>
      </div>

      <div class="sh-actions">
        <a v-for="s in socials" :key="s.name" :href="s.href" target="_blank" rel="noopener"
           class="sh-act" :title="`Share on ${s.name}`">{{ s.name }}</a>
        <a :href="mailto" class="sh-act">Email</a>
        <a :href="sms" class="sh-act">Text</a>
        <button class="sh-act" @click="nativeShare" v-if="canNativeShare">More…</button>
      </div>

      <details class="sh-more" :open="showQr" @toggle="showQr = $event.target.open">
        <summary>QR code</summary>
        <div class="sh-qr">
          <div v-if="qrSvg" class="qr" v-html="qrSvg"></div>
          <p v-else class="sh-hint">{{ qrError || 'Generating…' }}</p>
          <p class="sh-hint">Point a phone camera at this to open the same view.</p>
        </div>
      </details>

      <details class="sh-more">
        <summary>Embed</summary>
        <textarea class="sh-embed" readonly rows="4" :value="embedCode"
                  aria-label="Embed code" @focus="$event.target.select()"></textarea>
        <button class="sh-copy wide" @click="copy(embedCode, 'embed')">
          {{ copied === 'embed' ? '✓ Copied' : 'Copy embed code' }}
        </button>
        <p class="sh-hint">Renders without the site header, so it sits cleanly in another page.</p>
      </details>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import qrcode from 'qrcode-generator'

const props = defineProps({
  // Live map centre/zoom and encodings, supplied by the host view. The map owns
  // these (Leaflet does), so they cannot be read out of shared state.
  mapView: { type: Object, default: null },
  colorBy: { type: String, default: '' },
  sizeBy: { type: String, default: '' },
  // What the link is of, used in the share text.
  title: { type: String, default: 'Mushroom observations' },
  // Extra query params the host view owns — a built chart's configuration, so
  // the link opens that chart rather than only the filters behind it.
  extra: { type: Object, default: null },
  // Where the link should land. Defaults to the current route.
  path: { type: String, default: '' },
  // Icon-only, for sitting in a chart card's toolbar rather than a control bar.
  compact: { type: Boolean, default: false },
  // Icon-only at full button size, for the map's control bar.
  iconOnly: { type: Boolean, default: false },
  note: { type: String, default: 'This link reproduces what you are looking at — filters, colouring and overlay included.' },
})

const share = useShareState()
const open = ref(false)
const showQr = ref(false)
const copied = ref('')
const qrError = ref('')

const state = computed(() => ({
  mapView: props.mapView, colorBy: props.colorBy, sizeBy: props.sizeBy, extra: props.extra,
}))
const path = computed(() => props.path || null)

// Recomputed while the panel is open so panning the map updates the link.
const url = computed(() => (open.value ? share.buildUrl(state.value, path.value) : ''))
const embedCode = computed(() => (open.value ? share.buildEmbedCode(state.value, path.value) : ''))

const text = computed(() => props.title)
const enc = encodeURIComponent

const socials = computed(() => {
  const u = enc(url.value)
  const t = enc(text.value)
  return [
    { name: 'X', href: `https://twitter.com/intent/tweet?url=${u}&text=${t}` },
    { name: 'Bluesky', href: `https://bsky.app/intent/compose?text=${t}%20${u}` },
    { name: 'Facebook', href: `https://www.facebook.com/sharer/sharer.php?u=${u}` },
    { name: 'Reddit', href: `https://www.reddit.com/submit?url=${u}&title=${t}` },
  ]
})
const mailto = computed(() =>
  `mailto:?subject=${enc(text.value)}&body=${enc(`${text.value}\n\n${url.value}`)}`)
// The ?body= form is what iOS and Android both accept.
const sms = computed(() => `sms:?&body=${enc(`${text.value} ${url.value}`)}`)

const canNativeShare = computed(() => import.meta.client && typeof navigator !== 'undefined' && !!navigator.share)

async function nativeShare() {
  try {
    await navigator.share({ title: text.value, text: text.value, url: url.value })
  } catch { /* the user dismissed the sheet */ }
}

// QR: generated locally rather than through a third-party image service, which
// would hand the shared URL — filters and all — to someone else's server.
const qrSvg = ref('')
watch([url, showQr, open], () => {
  if (!open.value || !showQr.value || !url.value) return
  qrError.value = ''
  try {
    // Type 0 lets the library pick the smallest version that fits; 'M' error
    // correction survives a phone camera at an angle.
    const qr = qrcode(0, 'M')
    qr.addData(url.value)
    qr.make()
    qrSvg.value = qr.createSvgTag({ cellSize: 4, margin: 8, scalable: true })
  } catch (e) {
    // A very long link (a big species filter) can exceed the largest QR version.
    qrSvg.value = ''
    qrError.value = 'This view is too detailed to fit in a QR code — use the link instead.'
  }
}, { immediate: true })

function toggle() {
  open.value = !open.value
  if (!open.value) copied.value = ''
}

async function copy(value, what) {
  try {
    await navigator.clipboard.writeText(value)
  } catch {
    return // clipboard blocked; the field is selectable as a fallback
  }
  copied.value = what
  setTimeout(() => { if (copied.value === what) copied.value = '' }, 1600)
}
</script>

<style scoped>
.share { position: relative; }

.sh-btn {
  display: inline-flex; align-items: center; gap: 5px;
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  border-radius: 6px; padding: 5px 10px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
}
.sh-btn:hover, .sh-btn.on { background: var(--surface-2); }
/* Square, for a control bar where every label costs a row. */
.sh-btn.icon-only {
  width: 34px; height: 34px; padding: 0; justify-content: center;
}

/* Sits in a chart card's toolbar, matching the buttons beside it. */
.sh-btn.compact {
  width: 22px; height: 22px; padding: 0; border-radius: 5px;
  justify-content: center; color: var(--muted);
}
.sh-btn.compact:hover { color: var(--text); }

.sh-panel {
  position: absolute; top: calc(100% + 6px); left: 0; z-index: 900; width: 300px;
  max-height: 70vh; overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 12px; font-size: 0.82rem;
}
/* A card's share button sits at its right edge, so the panel hangs inward. */
.sh-panel.right { left: auto; right: 0; }
.sh-note { margin: 0 0 8px; color: var(--muted); font-size: 0.76rem; line-height: 1.4; }

.sh-url { display: flex; gap: 5px; }
.sh-url input {
  flex: 1; min-width: 0; background: var(--input-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 5px; padding: 5px 7px; font-size: 0.76rem;
}
.sh-copy {
  border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink);
  border-radius: 5px; padding: 5px 10px; font-size: 0.78rem; font-weight: 600; cursor: pointer;
}
.sh-copy.wide { width: 100%; margin-top: 5px; }

.sh-actions { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
.sh-act {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 5px; padding: 4px 9px; font-size: 0.76rem; font-weight: 600;
  text-decoration: none; cursor: pointer;
}
.sh-act:hover { background: var(--surface-3); }

.sh-more { margin-top: 10px; border-top: 1px solid var(--border-soft); padding-top: 8px; }
.sh-more summary { cursor: pointer; font-weight: 600; color: var(--text); }
.sh-qr { display: grid; justify-items: center; gap: 6px; margin-top: 8px; }
/* The generated SVG is black-on-white and must stay that way to scan, so it
   gets its own white plate rather than inheriting the dark surface. */
.qr { background: #fff; border-radius: 6px; padding: 4px; line-height: 0; }
.qr :deep(svg) { width: 168px; height: 168px; display: block; }
.sh-hint { color: var(--muted); font-size: 0.72rem; margin: 0; text-align: center; line-height: 1.4; }

.sh-embed {
  width: 100%; margin-top: 8px; background: var(--input-bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 5px; padding: 6px;
  font: 0.72rem/1.4 ui-monospace, monospace; resize: vertical;
}
</style>
