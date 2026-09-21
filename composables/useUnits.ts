// Display units. Data is stored in metres (elevation) and °C (temperature);
// the UI defaults to feet + Fahrenheit with toggles. The choices are shared
// across views (Nuxt state) and persisted per viewer in localStorage
// (handled in app.vue).

const M_TO_FT = 3.28084

export type ElevUnit = 'ft' | 'm'
export type TempUnit = 'F' | 'C'

export function useUnits() {
  const unit = useState<ElevUnit>('elev-unit', () => 'ft')
  const tempUnit = useState<TempUnit>('temp-unit', () => 'F')

  function has(v: any): v is (string | number | boolean) { 
    return v !== null && v !== undefined && v !== '' 
  }

  function elevValue(metres: number | string | null | undefined): number | null {
    if (!has(metres)) return null
    const m = Number(metres)
    if (isNaN(m)) return null
    return unit.value === 'ft' ? m * M_TO_FT : m
  }

  function elevLabel(metres: number | string | null | undefined): string {
    const v = elevValue(metres)
    return v === null ? '—' : `${Math.round(v).toLocaleString()} ${unit.value}`
  }

  function tempValue(celsius: number | string | null | undefined): number | null {
    if (!has(celsius)) return null
    const c = Number(celsius)
    if (isNaN(c) ) return null
    return tempUnit.value === 'F' ? c * 9 / 5 + 32 : c
  }

  function tempLabel(celsius: number | string | null | undefined): string {
    const v = tempValue(celsius)
    return v === null ? '—' : `${Math.round(v)}°${tempUnit.value}`
  }

  return { unit, tempUnit, elevValue, elevLabel, tempValue, tempLabel, M_TO_FT }
}
