// Generates vercel.json from netlify/functions/*.mjs (schedules) so Netlify
// stays the single source of truth. Run: node scripts/gen-vercel-config.mjs
//
// Vercel's Hobby plan refuses the whole deployment if any cron runs more than
// once a day, so sub-daily schedules are relaxed to once a day unless
// VERCEL_CRON_PLAN=pro. That costs little: ee-worker's every-minute cron is
// only the backstop for the poke every job submission sends, and the
// observations refresh can run daily.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

/** A single whole number in a cron field, or null for a range, list or step. */
const fixed = (field) => (/^\d+$/.test(field) ? Number(field) : null)

/**
 * The schedule Vercel Hobby accepts: unchanged when it already runs at most
 * daily, otherwise once a day at the first minute and hour it named.
 */
export function hobbySchedule(schedule) {
  const [minute, hour, ...rest] = schedule.trim().split(/\s+/)
  if (fixed(minute) !== null && fixed(hour) !== null) return schedule
  const first = (field) => fixed(field.split(/[,/-]/)[0]) ?? 0
  return [first(minute), first(hour), ...rest].join(' ')
}

export function buildVercelConfig({ dir = 'netlify/functions', plan = 'hobby' } = {}) {
  const crons = []
  for (const f of readdirSync(dir).filter((x) => x.endsWith('.mjs')).sort()) {
    const name = f.replace(/\.mjs$/, '')
    const src = readFileSync(`${dir}/${f}`, 'utf8')
    const cfg = src.match(/export const config = \{([^}]*)\}/)?.[1] ?? ''
    const schedule = cfg.match(/schedule:\s*'([^']+)'/)?.[1]
    if (schedule) {
      crons.push({ path: `/api/fn/${name}`, schedule: plan === 'pro' ? schedule : hobbySchedule(schedule) })
    }
  }
  return {
    buildCommand: 'npm run build',
    rewrites: [{ source: '/.netlify/functions/:name', destination: '/api/fn/:name' }],
    crons,
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const vercel = buildVercelConfig({ plan: process.env.VERCEL_CRON_PLAN || 'hobby' })
  writeFileSync('vercel.json', JSON.stringify(vercel, null, 2) + '\n')
  console.log(`vercel.json: ${vercel.crons.length} crons`)
}
