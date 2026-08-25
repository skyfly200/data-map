// Field registry shared by the Explore builder and ChartRenderer.
// `unit` marks fields whose display follows the ft/m or °F/°C toggle.

export const ALL_NUMERIC = [
  { key: 'elevation', label: 'Elevation', unit: 'elev' },
  { key: 'day_of_year', label: 'Day of year' },
  { key: 'tmax', label: 'High temp', unit: 'temp' },
  { key: 'tmin', label: 'Low temp', unit: 'temp' },
  { key: 'tavg', label: 'Avg temp', unit: 'temp' },
  { key: 'rain7', label: '7-day rain total (mm)' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'solar_exposure', label: 'Solar exposure' },
  { key: 'wind_exposure', label: 'Wind exposure' },
  { key: 'water_retention', label: 'Water retention' },
  { key: 'slope', label: 'Slope (°)' },
  { key: 'aspect', label: 'Aspect (°)' },
  { key: 'num_identification_agreements', label: 'ID agreements' },
]

export const ALL_CATEGORY = [
  { key: 'species', label: 'Species' },
  { key: 'land_cover_label', label: 'Land cover' },
  { key: 'cluster', label: 'Cluster' },
]
