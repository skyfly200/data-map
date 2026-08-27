// Field registry shared by the Explore builder and ChartRenderer.
// `unit` marks fields whose display follows the ft/m or °F/°C toggle.

export const ALL_NUMERIC = [
  { key: 'elevation', label: 'Elevation', unit: 'elev' },
  { key: 'day_of_year', label: 'Day of year' },
  { key: 'year', label: 'Year' },
  { key: 'month', label: 'Month (1–12)' },
  { key: 'tmax', label: 'High temp', unit: 'temp' },
  { key: 'tmin', label: 'Low temp', unit: 'temp' },
  { key: 'tavg', label: 'Avg temp', unit: 'temp' },
  { key: 'rain7', label: '7-day rain total (mm)' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'ndmi', label: 'NDMI' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'solar_exposure', label: 'Solar exposure' },
  { key: 'wind_exposure', label: 'Wind exposure' },
  { key: 'water_retention', label: 'Wetness index (TWI)' },
  { key: 'slope', label: 'Slope (°)' },
  { key: 'aspect', label: 'Aspect (°)' },
]

export const ALL_CATEGORY = [
  { key: 'species', label: 'Species' },
  { key: 'genus', label: 'Genus' },
  { key: 'land_cover_label', label: 'Land cover' },
  { key: 'cluster', label: 'Cluster' },
  { key: 'live_cluster', label: 'Live cluster' },
  { key: 'year', label: 'Year' },
  { key: 'month_name', label: 'Month' },
  { key: 'enrichment_level', label: 'Enrichment level' },
]
