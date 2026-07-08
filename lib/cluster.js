import { kmeans } from 'ml-kmeans'

// Clusters observations by environmental similarity using k-means with
// z-score normalization. Mirrors the Python cluster.py (sklearn KMeans +
// StandardScaler). Returns an array of { id, cluster } for the rows that had
// all features present; rows with any missing feature are skipped.
export function clusterObservations(observations, features, k) {
  const valid = observations.filter(o =>
    features.every(f => o[f] !== null && o[f] !== undefined)
  )

  if (valid.length < k) return []

  // Build the feature matrix
  const matrix = valid.map(o => features.map(f => o[f]))

  // Z-score normalization (per feature)
  const means = features.map((_, j) =>
    matrix.reduce((s, r) => s + r[j], 0) / matrix.length
  )
  const stds = features.map((_, j) => {
    const variance = matrix.reduce((s, r) => s + (r[j] - means[j]) ** 2, 0) / matrix.length
    return Math.sqrt(variance) || 1
  })
  const normalized = matrix.map(row =>
    row.map((v, j) => (v - means[j]) / stds[j])
  )

  const result = kmeans(normalized, k, { initialization: 'kmeans++', seed: 42 })

  return valid.map((o, i) => ({ id: o.id, cluster: result.clusters[i] }))
}
