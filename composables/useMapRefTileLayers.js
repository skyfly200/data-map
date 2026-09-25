import { watch } from 'vue'
import { TILE_LAYERS, arcgisExportUrl, layerSource, layerDataType } from '~/composables/mapLayers'
import { MAP_MAX_ZOOM } from '~/composables/useMapTileDate'

export function setupReferenceTileLayers({
  L, map,
  tileOpacity, tileDate,
  tileErrors, activeTileNotes, tileLayers, layerOpacity, heatmaps,
}) {
  // ArcGIS MapServer services render from a bbox rather than serving a cut
  // tile pyramid, so their tiles are asked for by extent.
  const ArcGISLayer = L.TileLayer.extend({
    getTileUrl(coords) {
      return arcgisExportUrl(this.options.service, coords.x, coords.y, coords.z,
        { size: 256, layers: this.options.serviceLayers })
    },
  })

  const tileOverlayList = []
  for (const o of TILE_LAYERS) {
    const opts = {
      attribution: o.attribution, maxZoom: MAP_MAX_ZOOM, maxNativeZoom: o.maxZoom,
      opacity: (o.opacity ?? 1) * tileOpacity.value,
      crossOrigin: 'anonymous',
      updateWhenIdle: false, updateWhenZooming: true,
    }
    const layer = o.arcgis
      ? new ArcGISLayer('', { ...opts, service: o.arcgis, serviceLayers: o.layers || '' })
      : L.tileLayer(o.url.replace('{date}', tileDate.value), opts)
    layer._baseOpacity = o.opacity ?? 1
    layer._spec = o
    tileLayers.push(layer)

    let loaded = 0
    let failed = 0
    layer.on('tileload', () => {
      loaded += 1
      if (loaded === 1) tileErrors.value = tileErrors.value.filter((n) => n !== o.name)
    })
    layer.on('tileerror', () => {
      failed += 1
      if (loaded === 0 && failed >= 3 && !tileErrors.value.includes(o.name)) {
        tileErrors.value = [...tileErrors.value, o.name]
      }
    })
    if (o.note || o.legend || o.time) {
      layer.on('add', () => {
        if (!activeTileNotes.value.some((n) => n.name === o.name)) {
          activeTileNotes.value = [...activeTileNotes.value, {
            name: o.name, note: o.note, legend: o.legend, time: !!o.time,
            native: o.maxZoom,
            slug: o.name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
          }]
        }
      })
    }
    layer.on('remove', () => {
      tileErrors.value = tileErrors.value.filter((n) => n !== o.name)
      activeTileNotes.value = activeTileNotes.value.filter((n) => n.name !== o.name)
      loaded = 0
      failed = 0
    })
    tileOverlayList.push({
      key: o.name, name: o.name, group: o.group, layer, note: o.note,
      source: layerSource(o.attribution), type: layerDataType(o.legend),
    })
  }

  // The global dimmer multiplies into each layer's own opacity rather than
  // replacing it, so proportional differences between layers survive.
  watch(tileOpacity, (v) => {
    for (const l of tileLayers) {
      const key = l._spec?.ee ? l._spec.key : l._spec?.name
      const own = layerOpacity.value[key] ?? l._baseOpacity ?? 1
      l.setOpacity(own * v)
    }
    heatmaps.persist()
  })

  // Moving the date re-points time-varying layers at another day's tiles.
  watch(tileDate, (d) => {
    if (!d) return
    for (const l of tileLayers) {
      if (l._spec?.time && l._spec.url) l.setUrl(l._spec.url.replace('{date}', d))
    }
  })

  return tileOverlayList
}
