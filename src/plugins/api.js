// Thin fetch-based client for the local Express API.
// Vite proxies /api → http://localhost:3001 in dev (see vite.config.js).
const BASE = import.meta.env.VITE_API_URL || '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || `API error ${res.status}`)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  listDatasets: () => request('/datasets'),
  getDataset: (id) => request(`/datasets/${id}`),
  createDataset: (body) => request('/datasets', { method: 'POST', body: JSON.stringify(body) }),
  deleteDataset: (id) => request(`/datasets/${id}`, { method: 'DELETE' }),
  listObservations: (id) => request(`/datasets/${id}/observations`)
}
