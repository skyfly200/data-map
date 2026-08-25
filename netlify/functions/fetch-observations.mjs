import { getRuntimeConfig, fetchObservationsStage } from '../lib/pipeline.mjs'

export const config = { timeout: 300 }

export default async function handler(request) {
  const config = getRuntimeConfig({}, request)
  const result = await fetchObservationsStage(config)

  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })
}
