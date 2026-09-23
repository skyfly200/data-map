// Active model to open on the map, handed from /jobs or /modeling/maxent.
// Stores only the model's identity — the map mints the tile surface on demand.

export interface OpenModel {
  configId: string
  label: string
}

export function useModelOverlay() {
  const pending = useState<OpenModel | null>('map-open-model', () => null)

  function open(data: OpenModel) {
    pending.value = data
  }

  function clear() {
    pending.value = null
  }

  return { pending, open, clear }
}
