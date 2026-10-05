// Vercel adapter: runs a netlify/functions/<name>.mjs handler (Web-standard
// Request -> Response) behind a Nitro route. Only registered when building
// for Vercel (see nuxt.config.ts); the Netlify build never bundles this.
import { defineEventHandler, toWebRequest, sendWebResponse, createError } from 'h3'

const modules = import.meta.glob('../../netlify/functions/*.mjs')

export default defineEventHandler(async (event) => {
  const name = event.context.params?.name
  const load = modules[`../../netlify/functions/${name}.mjs`]
  if (!load) throw createError({ statusCode: 404, statusMessage: 'Unknown function' })
  const mod: any = await load()
  const res: Response = await mod.default(toWebRequest(event), {})
  return sendWebResponse(event, res)
})
