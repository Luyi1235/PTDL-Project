const { chromium } = require("C:/Users/pc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const output = path.resolve(__dirname, "../运行结果/工作台模型接入_20261020");
const base = process.env.PTDL_WORKBENCH_URL || "http://127.0.0.1:8770";
const models = ["fine6144", "selectedmesh6144"];

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: "msedge" });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    let configRequests = 0;
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => {
      if (request.url().includes("/api/config")) configRequests++;
    });
    await page.goto(base);
    await page.waitForFunction(() => document.querySelector("#runtimeStatus").textContent.includes("就绪"));
    assert.deepEqual(await page.locator('[name="precision"] option').evaluateAll(
      options => options.map(option => option.value)), models);
    assert.equal(await page.locator('[name="precision"]').inputValue(), "fine6144");
    assert.equal(await page.locator('[name="algorithm"]').count(), 5);
    const initialRequests = configRequests;
    const validations = {};
    const switches = [];
    for (const model of models) {
      const start = Date.now();
      await page.selectOption('[name="precision"]', model);
      switches.push({ model, elapsed_ms: Date.now() - start });
      validations[model] = await page.locator("#modelValidation").textContent();
      assert.match(validations[model], /MAE.*R²/s);
      assert.match(validations[model], model === "fine6144" ? /0\.960.*0\.933/s : /0\.960.*0\.939/s);
      assert.doesNotMatch(validations[model], /构型|升力有效|尾迹有效|困难区域/);
      assert.equal(await page.locator("#modelValidation > *").count(), 1);
      assert.equal(await page.locator("#modelValidation > table").count(), 1);
      assert.doesNotMatch(validations[model], /待评估|尚未生成/);
    }
    assert.equal(configRequests, initialRequests, "Switching models must use cached metrics");
    const requests = [];
    const runs = [];
    for (const model of models) {
      await page.selectOption('[name="precision"]', model);
      for (const checkbox of await page.locator('[name="algorithm"]').all()) {
        if ((await checkbox.getAttribute("value")) === "NSGA-II") await checkbox.check();
        else await checkbox.uncheck();
      }
      await page.locator('[name="resolution"]').fill("1");
      await page.locator('[name="generations"]').fill("1");
      await page.selectOption('[name="controller"]', "rule");
      await page.selectOption('[name="fidelity"]', "surrogate");
      await page.selectOption('[name="sixdof_mode"]', "optimizer-loop");
      const response = page.waitForResponse(r =>
        r.url().endsWith("/api/runs") && r.request().method() === "POST");
      await page.locator("#runButton").click();
      const submitted = await response;
      const submission = await submitted.json();
      assert.equal(submitted.status(), 201, JSON.stringify(submission));
      requests.push({ model, id: submission.id });
      await page.waitForFunction(() => ["completed", "failed"].some(
        state => document.querySelector("#runStatusPill").classList.contains(state)),
        null, { timeout: 360000 });
      const runId = (await page.locator("#runId").textContent()).trim();
      const run = await (await page.request.get(`${base}/api/runs/${encodeURIComponent(runId)}`)).json();
      assert.equal(run.status, "completed", run.log_tail);
      assert.equal(run.result.summary.aero_training_samples, 6144);
      assert.equal(run.config.precision, model);
      assert.equal(run.result.algorithms.length, 1);
      const modelPath = run.command[run.command.indexOf("--surrogate") + 1];
      assert.match(modelPath, model === "fine6144" ? /精细气动候选代理\.json$/ : /筛选网格候选\.joblib$/);
      const domain = run.command[run.command.indexOf("--aero-domain-data") + 1];
      assert.match(domain, /覆盖扩样6144代理_20261019.*完整训练标签\.csv$/);
      runs.push({ id: runId, model, output: run.output, command: run.command, summary: run.result.summary });
      console.log(JSON.stringify({ model, runId, status: run.status }));
    }
    await page.selectOption('[name="precision"]', "fine6144");
    await page.locator(".surrogate-grid").screenshot({ path: path.join(output, "几何参数模型精度.png") });
    await page.selectOption('[name="precision"]', "selectedmesh6144");
    await page.locator(".surrogate-grid").screenshot({ path: path.join(output, "网格融合模型精度.png") });
    await page.screenshot({ path: path.join(output, "桌面.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, "手机.png"), fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, "验证结果.json"),
      JSON.stringify({ models, validations, switches, configRequests, errors, overflow, requests, runs }, null, 2));
    console.log("Both deployed models completed joint optimization with cached validation metrics.");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
