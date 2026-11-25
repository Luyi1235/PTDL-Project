const { chromium } = require("C:/Users/pc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const output = path.resolve(__dirname, "../运行结果/工作台精选更新_20261018");
const base = process.env.PTDL_WORKBENCH_URL || "http://127.0.0.1:8770";
const algorithms = ["NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO", "PTDL-耦合引导"];
const models = ["fine3072", "original3072"];

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
    assert.deepEqual(await page.locator('[name="algorithm"]').evaluateAll(
      inputs => inputs.map(input => input.value)), algorithms);
    assert.equal(await page.locator('[name="precision"]').inputValue(), "fine3072");
    const initialRequests = configRequests;
    const validations = {};
    for (const model of models) {
      await page.selectOption('[name="precision"]', model);
      validations[model] = await page.locator("#modelValidation").textContent();
      assert.match(validations[model], /MAE.*R²/s);
      assert.match(validations[model], /256.*248.*235/s);
      assert.doesNotMatch(validations[model], /待评估|尚未生成/);
    }
    assert.equal(configRequests, initialRequests, "Model switching must use cached metrics");

    const runs = [];
    for (const model of models) {
      await page.selectOption('[name="precision"]', model);
      for (const checkbox of await page.locator('[name="algorithm"]').all()) {
        if (model === "fine3072") await checkbox.check();
        else await checkbox.uncheck();
      }
      if (model === "original3072")
        await page.locator('[name="algorithm"][value="PTDL-耦合引导"]').check();
      await page.locator('[name="resolution"]').fill("1");
      await page.locator('[name="generations"]').fill("1");
      await page.selectOption('[name="controller"]', "rule");
      await page.selectOption('[name="fidelity"]', "surrogate");
      await page.locator("#runButton").click();
      await page.waitForFunction(() => ["completed", "failed"].some(
        state => document.querySelector("#runStatusPill").classList.contains(state)),
        null, { timeout: 240000 });
      const runId = (await page.locator("#runId").textContent()).trim();
      const response = await page.request.get(`${base}/api/runs/${encodeURIComponent(runId)}`);
      const run = await response.json();
      assert.equal(run.status, "completed", run.log_tail);
      assert.equal(run.result.summary.aero_training_samples, 3072);
      assert.equal(run.config.precision, model);
      assert.equal(run.result.algorithms.length, model === "fine3072" ? 5 : 1);
      assert.match(run.command[1], /运行_工作台联合优化\.py$/);
      const modelPath = run.command[run.command.indexOf("--surrogate") + 1];
      assert.match(modelPath, model === "fine3072" ? /精细气动候选代理\.json$/ : /均匀覆盖候选代理\.json$/);
      runs.push({ id: runId, model, algorithms: run.config.algorithms, summary: run.result.summary });
      console.log(JSON.stringify({ model, runId, status: run.status }));
    }
    const staleAlgorithm = await page.request.post(`${base}/api/runs`, {
      data: { algorithms: ["PTDL-去重贡献学习"], precision: "fine3072" },
    });
    assert.equal(staleAlgorithm.status(), 400);
    const staleModel = await page.request.post(`${base}/api/runs`, {
      data: { algorithms: ["NSGA-II"], precision: "activegp" },
    });
    assert.equal(staleModel.status(), 400);

    await page.selectOption('[name="precision"]', "fine3072");
    await page.screenshot({ path: path.join(output, "桌面.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, "手机.png"), fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, "验证结果.json"),
      JSON.stringify({ models, algorithms, validations, configRequests, errors, overflow, runs }, null, 2));
    console.log("Workbench selections, cached metrics, both model runs and five optimizers passed.");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
