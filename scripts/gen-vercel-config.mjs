// Generates vercel.json from netlify/functions/*.mjs (schedules) so Netlify
// stays the single source of truth. Run: node scripts/gen-vercel-config.mjs
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { vercelSchedule } from './vercel-cron.mjs'

const dir = 'netlify/functions'
const crons = []
for (const f of readdirSync(dir).filter((x) => x.endsWith('.mjs'))) {
  const name = f.replace(/\.mjs$/, '')
  const src = readFileSync(`${dir}/${f}`, 'utf8')
  const cfg = src.match(/export const config = \{([^}]*)\}/)?.[1] ?? ''
  const schedule = cfg.match(/schedule:\s*'([^']+)'/)?.[1]
  if (schedule) crons.push({ path: `/api/fn/${name}`, schedule: vercelSchedule(schedule) })
}
const vercel = {
  buildCommand: 'npm run build',
  rewrites: [{ source: '/.netlify/functions/:name', destination: '/api/fn/:name' }],
  crons,
}
writeFileSync('vercel.json', JSON.stringify(vercel, null, 2) + '\n')
console.log(`vercel.json: ${crons.length} crons`)
