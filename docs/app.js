/* Web demo: load model.onnx + meta.json + samples.json, draw the ECG, run the model locally. */
(async function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const status = (t) => ($("status").textContent = t);
  let meta, samples, session;

  try {
    [meta, samples] = await Promise.all([fetch("meta.json").then((r) => r.json()), fetch("samples.json").then((r) => r.json())]);
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
  } catch (e) {
    status("Could not load the model files (model.onnx / meta.json / samples.json). Run the Kaggle notebook and copy its docs/ output here.");
    console.error(e);
    return;
  }

  samples.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `#${s.ecg_id} — true: ${s.true.join(", ")}`;
    $("sample").appendChild(o);
  });
  renderAbout();

  async function predict(leads, fs, truth) {
    const scale = ECG.guessScale(leads);
    const x = ECG.toModelInput(leads, fs, meta.fs, meta.n_samples, scale);
    status("Running…");
    const t0 = performance.now();
    const out = await session.run({ ecg: new ort.Tensor("float32", x, [1, 12, meta.n_samples]) });
    const p = out.probabilities.data;
    status(`Done in ${(performance.now() - t0).toFixed(0)} ms on this device.`);
    renderBars(p);
    drawECG(x, meta.n_samples, meta.fs);
    $("truth").innerHTML = truth ? `Cardiologist labels (PTB-XL): <b>${truth.join(", ")}</b>` : "";
  }

  function renderBars(p) {
    $("bars").innerHTML = meta.classes.map((c, i) => {
      const on = p[i] >= meta.thresholds[i];
      return `<div class="bar"><div><b>${c}</b> · ${meta.names[i]}</div>
        <div class="track"><div class="fill ${on ? "on" : ""}" style="width:${(100 * p[i]).toFixed(1)}%"></div>
        <div class="tick" style="left:${(100 * meta.thresholds[i]).toFixed(1)}%"></div></div>
        <div class="pct">${(100 * p[i]).toFixed(0)}%</div></div>`;
    }).join("");
  }

  function drawECG(x, n, fs) {
    const cv = $("ecg"), g = cv.getContext("2d");
    const W = cv.width, H = cv.height, rowH = H / 12, left = 46;
    g.fillStyle = "#fff"; g.fillRect(0, 0, W, H);
    const pxPerSec = (W - left) / (n / fs);
    for (let s = 0; s <= n / fs; s += 0.2) {                       // ECG paper: 0.2 s major, 0.04 s minor
      g.strokeStyle = "#eeb7b7"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(left + s * pxPerSec, 0); g.lineTo(left + s * pxPerSec, H); g.stroke();
    }
    for (let y = 0; y <= H; y += rowH / 4) {
      g.strokeStyle = "#f6d5d5"; g.beginPath(); g.moveTo(left, y); g.lineTo(W, y); g.stroke();
    }
    const mvPx = rowH / 3;                                           // 1 mV = a third of a row
    g.strokeStyle = "#111"; g.lineWidth = 1.2; g.font = "13px system-ui"; g.fillStyle = "#333";
    for (let c = 0; c < 12; c++) {
      const y0 = rowH * (c + 0.55);
      g.fillText(meta.leads[c], 6, y0 + 4);
      g.beginPath();
      for (let t = 0; t < n; t++) {
        const px = left + (t / fs) * pxPerSec, py = y0 - x[c * n + t] * mvPx;
        t ? g.lineTo(px, py) : g.moveTo(px, py);
      }
      g.stroke();
    }
  }

  function renderAbout() {
    const rows = meta.test.per_class.map((r) =>
      `<tr><td>${r.cls}</td><td>${r.name}</td><td>${r.auroc.toFixed(3)}</td><td>${r.sensitivity.toFixed(2)}</td><td>${r.specificity.toFixed(2)}</td></tr>`).join("");
    $("about").innerHTML = `<p>${meta.model.params.toLocaleString()} parameters · ${meta.model.onnx_kb} KB · input 12 × ${meta.n_samples}
      (${meta.fs} Hz, 10 s) · test-fold macro AUROC <b>${meta.test.macro_auroc.toFixed(3)}</b>
      (95% CI ${meta.test.macro_auroc_ci[0].toFixed(3)}–${meta.test.macro_auroc_ci[1].toFixed(3)})</p>
      <table><tr><th>Class</th><th>Meaning</th><th>AUROC</th><th>Sensitivity</th><th>Specificity</th></tr>${rows}</table>
      <p class="hint">${meta.source}</p>`;
  }

  $("sample").addEventListener("change", () => {
    const s = samples[+$("sample").value];
    predict(s.signal.map((l) => Float32Array.from(l)), meta.fs, s.true);
  });
  $("file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    try {
      const leads = ECG.parseCSV(await f.text());
      await predict(leads, +$("fs").value, null);
    } catch (e) {
      status(e.message);
    }
  });
  $("sample").dispatchEvent(new Event("change"));
})();
