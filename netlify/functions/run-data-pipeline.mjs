import { getRuntimeConfig, runPipelineStages } from '../lib/pipeline.mjs'
import { requireUser } from '../lib/auth.mjs'

export const config = { timeout: 300 }

export default async function handler(request) {
  const auth = await requireUser(request)
  if (!auth.ok) return auth.response

  const config = getRuntimeConfig({}, request)
  const result = await runPipelineStages(config)

  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })
}
