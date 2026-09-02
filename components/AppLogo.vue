<template>
  <svg class="logo" :class="{ 'with-wordmark': wordmark }" :viewBox="wordmark ? '0 0 512 512' : '0 0 512 400'"
       :width="size" :height="wordmark ? size : Math.round(size * 400 / 512)"
       role="img" :aria-label="title">
    <title>{{ title }}</title>
    <defs>
      <radialGradient :id="id('bg')" cx="50%" cy="40%" r="80%">
        <stop offset="0%" stop-color="#0f172a" />
        <stop offset="100%" stop-color="#020617" />
      </radialGradient>

      <!-- The ray and the geotag glow rather than sitting flat, which is what
           reads as "a beam through the strata" instead of "a blue line". -->
      <filter :id="id('ray-glow')" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>

      <!-- Each plane gets a lighter left face and a darker right face; the pair
           is what makes a flat polygon read as an isometric slab. Successive
           planes step darker so the stack has depth from top to bottom. -->
      <linearGradient v-for="g in faceGradients" :key="g.id" :id="id(g.id)"
                      x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" :stop-color="g.from" />
        <stop offset="100%" :stop-color="g.to" />
      </linearGradient>
    </defs>

    <rect width="512" :height="wordmark ? 512 : 400" :rx="wordmark ? 112 : 96" :fill="`url(#${id('bg')})`" />

    <g transform="translate(0, -10)">
      <!-- Orbital arc over the stack -->
      <path d="M 172 107 A 95 60 0 0 1 256 70 A 95 60 0 0 1 340 107"
            fill="none" stroke="#38bdf8" stroke-width="5" stroke-dasharray="9 7" stroke-linecap="round" />

      <!-- The planes are drawn back-half, then ray, then front-half, so the beam
           passes behind the near edge of each slab and in front of the far one.
           That interleaving is the only thing selling the pierce. -->
      <template v-for="p in planes" :key="p.k">
        <polygon :points="`${p.x0},${p.y} 256,${p.top} 256,${p.y}`" :fill="`url(#${id(p.k + '-left')})`" />
        <polygon :points="`256,${p.top} ${p.x1},${p.y} 256,${p.y}`" :fill="`url(#${id(p.k + '-right')})`" />

        <line x1="256" :y1="p.rayFrom" x2="256" :y2="p.y" stroke="#38bdf8" stroke-width="5"
              stroke-linecap="round" :filter="`url(#${id('ray-glow')})`" />

        <polygon :points="`${p.x0},${p.y} 256,${p.bottom} 256,${p.y}`" :fill="`url(#${id(p.k + '-left')})`" />
        <polygon :points="`256,${p.bottom} ${p.x1},${p.y} 256,${p.y}`" :fill="`url(#${id(p.k + '-right')})`" />
      </template>

      <!-- Out the bottom of the last plane and into the target -->
      <line x1="256" y1="322" x2="256" y2="350" stroke="#38bdf8" stroke-width="5"
            stroke-linecap="round" :filter="`url(#${id('ray-glow')})`" />

      <!-- Entry points where the beam breaks each surface -->
      <ellipse v-for="p in planes" :key="`e-${p.k}`" cx="256" :cy="p.y" :rx="p.rx" :ry="p.rx / 2"
               fill="#ffffff" :filter="`url(#${id('ray-glow')})`" />

      <!-- Geotag target at the base: where the column of data lands on the ground -->
      <g :filter="`url(#${id('ray-glow')})`">
        <circle cx="256" cy="350" r="16" fill="none" stroke="#38bdf8" stroke-width="3.5" />
        <circle cx="256" cy="350" r="7" fill="#38bdf8" />
        <circle cx="256" cy="350" r="3" fill="#ffffff" />
      </g>
    </g>

    <text v-if="wordmark" x="256" y="445" text-anchor="middle" fill="#f8fafc"
          font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
          font-size="24" font-weight="700" letter-spacing="6">NEXSTRATA</text>
  </svg>
</template>

<script setup>
const props = defineProps({
  // Rendered size in px. The mark is square with the wordmark and cropped
  // above it without, so the height follows from this rather than being set.
  size: { type: Number, default: 28 },
  // Include the NEXSTRATA wordmark. Off in the header, where the <h1> beside
  // it already says the name and a second copy would only repeat it.
  wordmark: { type: Boolean, default: false },
  title: { type: String, default: 'Nexstrata' },
})

// The gradient and filter ids are document-global, so they are namespaced to
// keep them clear of the other inline SVGs the app draws (chart marks, the
// control-bar icons). Two logos on one page share the same ids — the browser
// resolves each reference to the first match, and since the definitions are
// identical that renders correctly, whereas a per-instance id would differ
// between the server and client renders and break hydration.
const id = (name) => `nxs-${name}`

const RAMP = ['#e2e8f0', '#cbd5e1', '#94a3b8', '#64748b', '#475569', '#334155']
// Left face of plane n runs RAMP[n]→RAMP[n+1]; the right face is one step
// darker again, so the two faces of a slab differ by the same amount that
// consecutive slabs do.
const faceGradients = ['p1', 'p2', 'p3', 'p4'].flatMap((k, i) => [
  { id: `${k}-left`, from: RAMP[i], to: RAMP[i + 1] },
  { id: `${k}-right`, from: RAMP[i + 1], to: RAMP[i + 2] },
])

// y is the plane's midline (where the beam pierces), top/bottom its far and
// near tips, x0/x1 its left and right corners. They narrow going down, which
// is what gives the stack its perspective.
const planes = [
  { k: 'p1', x0: 176, x1: 336, y: 150, top: 120, bottom: 180, rayFrom: 70, rx: 4 },
  { k: 'p2', x0: 198, x1: 314, y: 215, top: 190, bottom: 240, rayFrom: 180, rx: 3.5 },
  { k: 'p3', x0: 220, x1: 292, y: 270, top: 250, bottom: 290, rayFrom: 240, rx: 3 },
  { k: 'p4', x0: 240, x1: 272, y: 310, top: 298, bottom: 322, rayFrom: 290, rx: 2.5 },
]
</script>

<style scoped>
.logo { display: block; flex: 0 0 auto; }
</style>
