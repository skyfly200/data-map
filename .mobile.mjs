import { chromium } from 'playwright-core'
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
// iPhone-ish
const ctx = await b.newContext({
  viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
})
const p = await ctx.newPage()
const errs = []
p.on('pageerror', e => errs.push(String(e).slice(0, 160)))

async function check(route, waitFor) {
  await p.goto('http://localhost:3000' + route, { waitUntil: 'domcontentloaded' })
  if (waitFor) await p.waitForFunction(waitFor, { timeout: 90000 }).catch(() => {})
  await p.waitForTimeout(2000)
  const m = await p.evaluate(() => {
    const doc = document.documentElement
    const overflowing = []
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.right > doc.clientWidth + 2) {
        overflowing.push(`${el.tagName.toLowerCase()}.${(el.className || '').toString().split(' ')[0]} → ${Math.round(r.right)}px`)
      }
    }
    // Tap targets below the 44px guideline.
    const small = []
    for (const el of document.querySelectorAll('button, a, select, input[type=checkbox]')) {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.height > 0 && (r.height < 32 || r.width < 24)) {
        small.push(`${el.tagName.toLowerCase()}.${(el.className || '').toString().split(' ')[0]} ${Math.round(r.width)}x${Math.round(r.height)}`)
      }
    }
    return {
      scrollW: doc.scrollWidth, clientW: doc.clientWidth,
      overflowing: [...new Set(overflowing)].slice(0, 6),
      small: [...new Set(small)].slice(0, 6),
    }
  })
  const bleeds = m.scrollW > m.clientW + 2
  console.log(`\n${route}`)
  console.log(`  horizontal scroll: ${bleeds ? `YES (${m.scrollW} > ${m.clientW})` : 'no'}`)
  if (m.overflowing.length) console.log(`  overflowing: ${m.overflowing.join(', ')}`)
  if (m.small.length) console.log(`  small tap targets: ${m.small.join(', ')}`)
}

await check('/', null)
await check('/map', () => document.querySelector('canvas.leaflet-zoom-animated'))
await check('/charts', () => document.querySelectorAll('path.dot').length > 0)
await check('/analysis', () => document.querySelector('.scope'))
await check('/data', () => document.querySelector('.filter-panel'))
await check('/guide', null)

console.log('\nerrors:', errs.length ? [...new Set(errs)] : 'none')
await b.close()
