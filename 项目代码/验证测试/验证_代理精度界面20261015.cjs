const { chromium } = require("C:/Users/pc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");
const fs = require("fs");
const path = require("path");
const output = path.resolve(__dirname, "../运行结果/工作台代理精度验证_20261016");
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "msedge" });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    let requests = 0;
    page.on("request", r => { if (r.url().includes("/api/config")) requests++; });
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("http://127.0.0.1:8769/");
    await page.waitForFunction(() => document.querySelectorAll("#modelValidation tbody tr").length === 2);
    const options = await page.locator('[name="precision"] option').evaluateAll(items => items.map(i => i.value));
    const metrics = [];
    for (const value of options) {
      const start = Date.now();
      await page.selectOption('[name="precision"]', value);
      const text = await page.locator("#modelValidation").innerText();
      if (!text.includes("MAE") || !text.includes("R²") || text.includes("待评估") || text.includes("历史对照"))
        throw new Error(`${value}: ${text}`);
      metrics.push({ value, elapsedMs: Date.now() - start, text });
    }
    if (requests !== 1) throw new Error(`Repeated configuration requests: ${requests}`);
    await page.selectOption('[name="precision"]', "activegp");
    const panel = page.locator(".config-column .panel").nth(1);
    await panel.screenshot({ path: path.join(output, "气动板块桌面.png") });
    const positions = await page.evaluate(() => {
      const training = document.querySelector('[name="precision"]').getBoundingClientRect();
      const metrics = document.querySelector("#modelValidation").getBoundingClientRect();
      const mesh = document.querySelector('[name="vsp_mesh_level"]').getBoundingClientRect();
      const fidelity = document.querySelector('[name="fidelity"]').getBoundingClientRect();
      return { metricsRight: metrics.left > training.left, fidelityRight: fidelity.left > mesh.left };
    });
    if (!positions.metricsRight || !positions.fidelityRight) throw new Error("Unexpected layout");
    await page.setViewportSize({ width: 390, height: 844 });
    await panel.screenshot({ path: path.join(output, "气动板块手机.png") });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    const gallery = await browser.newPage();
    await gallery.goto("http://127.0.0.1:8769/research-figures-latest/index.html");
    const images = await gallery.locator("img").evaluateAll(async items => {
      await Promise.all(items.map(i => i.decode()));
      return items.map(i => ({ width: i.naturalWidth, height: i.naturalHeight }));
    });
    if (images.length !== 4 || images.some(i => i.width < 100)) throw new Error("Gallery image failure");
    const result = { requests, models: options.length, metrics, positions, overflow, errors, galleryImages: images.length };
    fs.writeFileSync(path.join(output, "界面验证.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
    if (overflow || errors.length) throw new Error("Responsive or script error");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
