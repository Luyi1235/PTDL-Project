const { chromium } = require("C:/Users/pc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const output = path.resolve(__dirname, "../运行结果/动力系统更新_20261023");
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:8770/", { waitUntil: "networkidle" });
    await page.waitForFunction(() =>
      document.querySelector("#powertrainReference").textContent.includes("1005-20261023"));
    const values = await page.evaluate(() => {
      const f = document.querySelector("#configForm");
      return { engine: f.elements.engine_speed.value, resistance: f.elements.battery_resistance.value };
    });
    if (values.engine !== "5000" || values.resistance !== "1") throw Error(JSON.stringify(values));
    await page.locator("details").evaluateAll((nodes) => nodes.forEach((node) => { node.open = true; }));
    await page.screenshot({ path: path.join(output, "工作台桌面.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, "工作台手机.png"), fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    if (overflow > 2) throw Error(`Horizontal overflow ${overflow}`);
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.locator('input[name="resolution"]').fill("1");
    await page.locator('input[name="generations"]').fill("1");
    await page.locator('select[name="controller"]').selectOption("source-ems");
    await page.locator('select[name="fidelity"]').selectOption("surrogate");
    await page.locator('select[name="sixdof_mode"]').selectOption("full-mission");
    const boxes = page.locator('input[type="checkbox"][name="algorithm"]');
    for (let i = 0; i < await boxes.count(); i++) {
      const box = boxes.nth(i);
      await box.setChecked(await box.getAttribute("value") === "NSGA-II");
    }
    let record = { id: process.env.PTDL_CHECK_RUN_ID };
    if (!record.id) {
      const submitted = page.waitForResponse((response) =>
        response.url().endsWith("/api/runs") && response.request().method() === "POST");
      await page.locator("#runButton").click();
      const response = await submitted;
      record = await response.json();
      if (!response.ok()) throw Error(JSON.stringify(record));
    }
    let result;
    for (let attempt = 0; attempt < 600; attempt++) {
      result = await (await fetch(`http://127.0.0.1:8770/api/runs/${record.id}`)).json();
      if (["completed", "failed"].includes(result.status)) break;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    fs.writeFileSync(path.join(output, "工作台端到端结果.json"),
      JSON.stringify({ values, overflow, errors, result }, null, 2), "utf8");
    if (result.status !== "completed") throw Error(result.log_tail);
    if (result.result.summary.powertrain_source.source_version !== "1005-20261023") {
      throw Error("Unexpected powertrain version");
    }
    await page.locator("#runHistory").selectOption(record.id);
    await page.waitForFunction(() => !document.querySelector("#runButton").disabled);
    await page.screenshot({ path: path.join(output, "工作台运行结果.png"), fullPage: true });
    if (errors.length) throw Error(errors.join("\n"));
    console.log(JSON.stringify({
      id: record.id, status: result.status, output: result.output,
      outcome: result.result.summary.outcome || "feasible",
      version: result.result.summary.powertrain_source.source_version, overflow, errors,
    }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
