import { enrichObservationsStage, getRuntimeConfig } from '../lib/pipeline.mjs'

export const config = { timeout: 300 }

export default async function handler(request) {
  const config = getRuntimeConfig({}, request)
  const result = await enrichObservationsStage(config)

  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })
}
