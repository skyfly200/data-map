---
label: wayfinder:task
status: closed
assignee: claude
---

# ISSUE-1: Fix LayerManager state persistence on mobile Safari

## Question

LayerManager state occasionally fails to persist on mobile Safari. At a foray, members will be on phones and iPads in the field — this is a demo-breaking bug in exactly that setting.

Diagnose the root cause (likely `localStorage` or reactive state not surviving Safari's ITP or background-tab eviction), then fix it so layer selections survive page reload and navigation on iOS Safari.

Acceptance: open the app on mobile Safari, configure layers, navigate away and back, reload — state is restored.

## Blocked by

_(none — frontier ticket)_

## Resolution

Root cause: `activeOverlays`, `overlayOrder`, and `layerOpacity` in `MushroomMap.vue` had no localStorage persistence — entirely session-in-memory. On mobile Safari (aggressive background-tab eviction) and on any navigation, all layer selections were lost.

Fix applied in `components/MushroomMap.vue`:
- Added `OVERLAY_STATE_KEY = 'map-overlay-state'` and `saveOverlayState()` / `restoreOverlays()` functions.
- `watch([activeOverlays, overlayOrder, layerOpacity], saveOverlayState, { deep: true })` persists state on every change.
- `restoreOverlays()` called after `overlayLayers.value = tileOverlayList` in `onMounted` to restore static layers immediately.
- `addEeLayer()` extended to check saved state and re-enable matching EE layers as they arrive from the server.
- `soloKey` and `layerBlend` intentionally excluded (session gestures, not standing preferences).
