import { chromium } from 'playwright'
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
for (const [name, w, h, touch] of [['mobile', 390, 844, true], ['desktop', 1300, 950, false]]) {
  const p = await (await b.newContext({ viewport:{width:w,height:h}, serviceWorkers:'block', hasTouch: touch, isMobile: touch })).newPage()
  const errs = []
  p.on('pageerror', e => errs.push(e.message))
  p.on('console', m => { if (m.type()==='error' && !/ERR_TUNNEL|Failed to load|net::/.test(m.text())) errs.push(m.text()) })
  await p.goto('http://localhost:3000/map', { waitUntil: 'domcontentloaded' })
  await p.waitForSelector('.controls', { timeout: 120000 })
  await p.waitForTimeout(3500)
  console.log(`\n=== ${name} ===`)
  const cb = await p.locator('.controls').boundingBox()
  console.log('control bar height:', Math.round(cb.height), '| items:', await p.locator('.controls > *').count())
  console.log('counter badge gone:', await p.locator('.chunk-status').count() === 0)
  console.log('pin button gone:', await p.locator('button[title*="Drop a point"]').count() === 0)
  console.log('leaflet layers control gone:', await p.locator('.leaflet-control-layers').count() === 0)
  console.log('locate control above zoom:', await p.locator('.leaflet-bottom.leaflet-left .locate-ctl').count())

  // Every button in the bar should now be the same height.
  const heights = await p.locator('.controls button').evaluateAll(els => [...new Set(els.map(e => Math.round(e.getBoundingClientRect().height)))])
  console.log('distinct button heights in bar:', heights)

  // Layers popover, in the bar.
  const layersBtn = p.locator('.controls .pop-btn').filter({ has: p.locator('text=≣') }).first()
  await p.locator('.controls .pop-btn').nth(1).click(); await p.waitForTimeout(500)
  console.log('layer rows in popover:', await p.locator('.lay-row').count())
  await p.keyboard.press('Escape'); await p.waitForTimeout(300)

  // Every popover must stay inside the viewport.
  let worst = 0
  for (let i = 0; i < await p.locator('.controls .pop-btn').count(); i++) {
    await p.locator('.controls .pop-btn').nth(i).click(); await p.waitForTimeout(450)
    const box = await p.locator('.pop-panel').boundingBox().catch(() => null)
    if (box) worst = Math.max(worst, Math.round(box.x + box.width - w), Math.round(-box.x))
    await p.keyboard.press('Escape'); await p.waitForTimeout(200)
  }
  console.log('worst panel overflow past a viewport edge (px, <=0 is good):', worst)
  console.log('errors:', errs.slice(0,3))
  await p.screenshot({ path: `/tmp/claude-0/bar-${name}.png` })
  await p.close()
}
await b.close()
