import { watch } from 'vue'
import { MAP_MAX_ZOOM } from '~/composables/useMapTileDate'

export function setupElevBandLayer({ L, map, filters, activeTileNotes, tileLayers }) {
  const ElevBandGridLayer = L.GridLayer.extend({
    createTile(coords, done) {
      const sz = this.getTileSize()
      const canvas = document.createElement('canvas')
      canvas.width = sz.x
      canvas.height = sz.y
      const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${coords.z}/${coords.x}/${coords.y}.png`
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        try {
          const ctx = canvas.getContext('2d')
          ctx.drawImage(img, 0, 0, sz.x, sz.y)
          const src = ctx.getImageData(0, 0, sz.x, sz.y).data
          const out = ctx.createImageData(sz.x, sz.y)
          const loM = this.options.elevMin
          const hiM = this.options.elevMax
          const hasFilter = loM != null || hiM != null
          const lo = loM ?? -Infinity
          const hi = hiM ?? Infinity
          // Band gradient stops: deep blue (low) → teal → green (mid) → yellow → orange (high).
          const GRAD = [
            [30, 100, 200],
            [42, 161, 210],
            [80, 185, 120],
            [220, 185,  60],
            [210,  90,  30],
          ]
          function bandColor(t) {
            const seg = Math.min(3, Math.floor(t * 4))
            const s = t * 4 - seg
            const a = GRAD[seg], b = GRAD[seg + 1]
            return [
              Math.round(a[0] + s * (b[0] - a[0])),
              Math.round(a[1] + s * (b[1] - a[1])),
              Math.round(a[2] + s * (b[2] - a[2])),
            ]
          }
          for (let i = 0; i < src.length; i += 4) {
            const elev = src[i] * 256 + src[i + 1] + src[i + 2] / 256 - 32768
            if (hasFilter) {
              if (elev >= lo && elev <= hi) {
                const span = hi - lo
                const t = span > 0 ? (elev - lo) / span : 0.5
                const [r, g, bv] = bandColor(Math.max(0, Math.min(1, t)))
                out.data[i] = r; out.data[i + 1] = g; out.data[i + 2] = bv; out.data[i + 3] = 185
              } else {
                out.data[i] = 0; out.data[i + 1] = 0; out.data[i + 2] = 0; out.data[i + 3] = 55
              }
            } else {
              // No filter: hypsometric tint (green → tan → grey) across the full DEM range.
              const t = Math.max(0, Math.min(1, (elev + 50) / 4500))
              let r, g, bv
              if (t < 0.4) {
                const s = t / 0.4
                r = Math.round(132 + s * 56); g = Math.round(184 - s * 32); bv = Math.round(112 - s * 16)
              } else {
                const s = (t - 0.4) / 0.6
                r = Math.round(188 - s * 28); g = Math.round(152 + s * 6); bv = Math.round(96 + s * 62)
              }
              out.data[i] = r; out.data[i + 1] = g; out.data[i + 2] = bv; out.data[i + 3] = 130
            }
          }
          ctx.putImageData(out, 0, 0)
        } catch { /* silently ignore decode errors on bad tiles */ }
        done(null, canvas)
      }
      img.onerror = () => done(null, canvas)
      img.src = url
      return canvas
    },
  })

  const elevBandLayer = new ElevBandGridLayer({
    elevMin: null, elevMax: null,
    tileSize: 256, maxZoom: MAP_MAX_ZOOM, maxNativeZoom: 14,
    opacity: 0.75, attribution: 'Elevation: Tilezen / Amazon Web Services (CC BY)',
    updateWhenIdle: false, updateWhenZooming: true,
  })
  elevBandLayer._baseOpacity = 0.75
  tileLayers.push(elevBandLayer)

  watch([() => filters.value.elevMin, () => filters.value.elevMax], ([lo, hi]) => {
    elevBandLayer.options.elevMin = lo ?? null
    elevBandLayer.options.elevMax = hi ?? null
    if (map?.hasLayer(elevBandLayer)) elevBandLayer.redraw()
  })

  elevBandLayer.on('add', () => {
    if (!activeTileNotes.value.some((n) => n.name === 'Elevation band')) {
      activeTileNotes.value = [...activeTileNotes.value, {
        name: 'Elevation band',
        note: 'Decoded from Terrarium DEM tiles. With an elevation filter set (Map Filters), in-band terrain is highlighted; without one, a hypsometric tint shows relief.',
        legend: {
          type: 'ramp', unit: 'm',
          min: filters.value.elevMin != null ? String(filters.value.elevMin) : '0',
          max: filters.value.elevMax != null ? String(filters.value.elevMax) : '4 500+',
          stops: filters.value.elevMin != null || filters.value.elevMax != null
            ? ['#1e64c8', '#2aa1d2', '#50b978', '#dcb93c', '#d25a1e']
            : ['#84b870', '#c9a86c', '#a0a0a0'],
        },
        slug: 'elevation-band',
      }]
    }
  })
  elevBandLayer.on('remove', () => {
    activeTileNotes.value = activeTileNotes.value.filter((n) => n.name !== 'Elevation band')
  })

  return {
    key: 'Elevation band', name: 'Elevation band', group: 'Terrain',
    layer: elevBandLayer,
    note: 'Highlights terrain within the elevation filter. Hypsometric tint when no filter is set.',
    source: 'Tilezen/Amazon', type: 'Continuous raster',
  }
}
