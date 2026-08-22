// Elevation display units. Data is stored in metres (iNaturalist / DEM);
// the UI defaults to feet with a toggle. The choice is shared across views
// (Nuxt state) and persisted per viewer in localStorage (handled in app.vue).

const M_TO_FT = 3.28084

export function useUnits() {
  const unit = useState('elev-unit', () => 'ft') // 'ft' | 'm'

  function elevValue(metres) {
    if (metres === null || metres === undefined || metres === '') return null
    return unit.value === 'ft' ? metres * M_TO_FT : metres
  }

  function elevLabel(metres) {
    const v = elevValue(metres)
    return v === null ? '—' : `${Math.round(v).toLocaleString()} ${unit.value}`
  }

  return { unit, elevValue, elevLabel, M_TO_FT }
}
