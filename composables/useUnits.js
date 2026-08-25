// Display units. Data is stored in metres (elevation) and °C (temperature);
// the UI defaults to feet + Fahrenheit with toggles. The choices are shared
// across views (Nuxt state) and persisted per viewer in localStorage
// (handled in app.vue).

const M_TO_FT = 3.28084

export function useUnits() {
  const unit = useState('elev-unit', () => 'ft') // 'ft' | 'm'
  const tempUnit = useState('temp-unit', () => 'F') // 'F' | 'C'

  function has(v) { return v !== null && v !== undefined && v !== '' }

  function elevValue(metres) {
    if (!has(metres)) return null
    return unit.value === 'ft' ? metres * M_TO_FT : metres
  }

  function elevLabel(metres) {
    const v = elevValue(metres)
    return v === null ? '—' : `${Math.round(v).toLocaleString()} ${unit.value}`
  }

  function tempValue(celsius) {
    if (!has(celsius)) return null
    return tempUnit.value === 'F' ? celsius * 9 / 5 + 32 : Number(celsius)
  }

  function tempLabel(celsius) {
    const v = tempValue(celsius)
    return v === null ? '—' : `${Math.round(v)}°${tempUnit.value}`
  }

  return { unit, tempUnit, elevValue, elevLabel, tempValue, tempLabel, M_TO_FT }
}
