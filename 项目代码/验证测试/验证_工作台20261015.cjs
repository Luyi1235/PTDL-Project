const { chromium } = require("C:/Users/pc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const output = path.join(root, "运行结果", "工作台同步验证_20261016");
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "msedge" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("http://127.0.0.1:8769/");
  await page.waitForFunction(() => document.querySelector("#runtimeStatus").textContent.includes("就绪"));
  await page.selectOption('[name="precision"]', "activegp");
  await page.waitForFunction(() => document.querySelector("#modelValidation").textContent.includes("MAE"));
  for (const checkbox of await page.locator('[name="algorithm"]').all()) await checkbox.uncheck();
  await page.locator('[name="algorithm"][value="PTDL-耦合引导"]').check();
  await page.locator('[name="algorithm"][value="PTDL-去重贡献学习"]').check();
  await page.locator('[name="resolution"]').fill("1");
  await page.locator('[name="generations"]').fill("1");
  await page.selectOption('[name="controller"]', "rule");
  await page.selectOption('[name="fidelity"]', "surrogate");
  await page.selectOption('[name="response_definition"]', "recovery-integral");
  const invalid = await page.evaluate(() => [...document.querySelectorAll("input,select")]
    .filter(x => !x.validity.valid).map(x => `${x.name} ${x.validationMessage}`));
  if (invalid.length) throw new Error(invalid.join("\n"));
  await page.locator("#runButton").click();
  await page.waitForFunction(() => ["completed", "failed"].some(
    state => document.querySelector("#runStatusPill").classList.contains(state)),
    null, { timeout: 180000 });
  const state = await page.locator("#runStatusPill").textContent();
  const error = await page.locator("#formError").textContent();
  if (state.includes("失败") || error) throw new Error(`${state} ${error}`);
  await page.locator("#results").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, "桌面结果.png"), fullPage: true });
  const canvas = await page.locator("#paretoChart").evaluate(c => {
    const pixels = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let colored = 0;
    for (let i = 0; i < pixels.length; i += 4)
      if (pixels[i+3] > 0 && Math.max(...pixels.slice(i, i+3))-Math.min(...pixels.slice(i, i+3)) > 20) colored++;
    return { width: c.width, height: c.height, coloredPixels: colored };
  });
  const runId = await page.locator("#runId").textContent();
  const response = await page.request.get(`http://127.0.0.1:8769/api/runs/${encodeURIComponent(runId.trim())}`);
  const run = await response.json();
  if (run.status !== "completed" || !run.result?.pareto || canvas.coloredPixels < 20)
    throw new Error("Workbench result or Pareto canvas is missing");
  const estimateLegacy = await page.locator("#runtimeEstimate").textContent();
  await page.selectOption('[name="fidelity"]', "all");
  await page.selectOption('[name="vsp_mesh_level"]', "legacy");
  const estimateCoarse = await page.locator("#runtimeEstimate").textContent();
  await page.selectOption('[name="vsp_mesh_level"]', "fine");
  const estimateFine = await page.locator("#runtimeEstimate").textContent();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#results").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, "手机结果.png"), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth+1);
  const gallery = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  await gallery.goto("http://127.0.0.1:8769/research-figures/index.html");
  const images = await gallery.locator("img").evaluateAll(async elements => {
    await Promise.all(elements.map(image => { image.loading = "eager"; return image.decode(); }));
    return elements.map(image => ({ width: image.naturalWidth, height: image.naturalHeight }));
  });
  if (images.length !== 10 || images.some(image => image.width < 100 || image.height < 100))
    throw new Error("Scientific gallery images are missing");
  await gallery.screenshot({ path: path.join(output, "科研图集网页.png") });
  const result = { state, runId, errors, canvas, overflow, estimateLegacy, estimateCoarse, estimateFine,
    galleryImages: images.length,
    algorithm: run.config.algorithms, model: run.config.precision,
    responseDefinition: run.result.summary.response_disturbance.objective_definition };
  fs.writeFileSync(path.join(output, "验证结果.json"), JSON.stringify(result, null, 2));
  await browser.close();
  if (errors.length || overflow) throw new Error(JSON.stringify(result));
  console.log(JSON.stringify(result, null, 2));
})().catch(error => { console.error(error); process.exit(1); });
