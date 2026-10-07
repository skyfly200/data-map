// Vercel's Hobby plan rejects the whole deploy if any cron runs more than once
// a day. Netlify (the real host) keeps its own schedules; vercel.json gets each
// one cut down to a single daily run unless VERCEL_PLAN=pro says otherwise.
// ee-worker loses little: ee-jobs, gbif-fetch and fetch-species call it
// directly when they queue a job, so its cron is only a sweep for stragglers.

const NUM = /^\d+$/

/** '* * * * *' → '0 0 * * *', '0 *\/6 * * *' → '0 0 * * *', '30 4 * * *' unchanged. */
export function toDaily(schedule) {
  const [min, hour, ...rest] = schedule.trim().split(/\s+/)
  if (rest.length !== 3) return schedule
  return [NUM.test(min) ? min : '0', NUM.test(hour) ? hour : '0', ...rest].join(' ')
}

export function vercelSchedule(schedule, plan = process.env.VERCEL_PLAN) {
  return plan === 'pro' ? schedule : toDaily(schedule)
}
