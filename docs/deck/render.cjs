// Render out/*.html to PNG at 2x (3200x1800). Local files only. Usage: node render.cjs [destDir]
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const out = path.join(__dirname, 'out'), dest = path.resolve(process.argv[2] || path.join(__dirname, 'preview'));
  fs.mkdirSync(dest, { recursive: true });
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 });
  await page.route('**/*', route => route.request().url().startsWith('file://') ? route.continue() : route.abort());
  for (const name of fs.readdirSync(out).filter(f => f.endsWith('.html')).sort()) {
    await page.goto('file://' + path.join(out, name));
    await page.evaluate(() => document.fonts.ready);
    // Write beside the target, then rename: the dashboard never serves a half-written slide.
    const target = path.join(dest, name.replace(/\.html$/, '.png')), temp = path.join(dest, '.' + path.basename(target) + '.tmp');
    await page.screenshot({ path: temp, type: 'png' });
    fs.renameSync(temp, target);
    console.log(target);
  }
  await browser.close();
})().catch(error => { console.error(error); process.exitCode = 1; });
