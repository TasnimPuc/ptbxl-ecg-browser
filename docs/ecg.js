/* Pure helpers shared by the web page and the Node test (no DOM, no network). */
(function (root) {
  "use strict";

  /** Parse a CSV/TSV of an ECG: one row per sample, one column per lead (12 columns), optional header row.
   *  Returns an array of 12 Float32Arrays (lead-major). Throws a readable error for wrong shapes. */
  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length);
    if (!lines.length) throw new Error("The file is empty.");
    const split = (l) => l.split(/[,;\t ]+/).filter((v) => v.length);
    let rows = lines.map(split);
    if (rows[0].some((v) => isNaN(parseFloat(v)))) rows = rows.slice(1);          // header
    const n = rows.length;
    const nc = rows[0].length;
    if (nc < 12) throw new Error(`Expected 12 lead columns, found ${nc}.`);
    const leads = Array.from({ length: 12 }, () => new Float32Array(n));
    for (let t = 0; t < n; t++) {
      for (let c = 0; c < 12; c++) {
        const v = parseFloat(rows[t][nc - 12 + c]);                              // last 12 columns (time column allowed)
        leads[c][t] = Number.isFinite(v) ? v : 0;
      }
    }
    return leads;
  }

  /** Linear resampling of one lead from fsIn to fsOut. */
  function resample(x, fsIn, fsOut) {
    if (fsIn === fsOut) return Float32Array.from(x);
    const n = Math.max(1, Math.round((x.length * fsOut) / fsIn));
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = (i * fsIn) / fsOut;
      const j = Math.floor(p);
      const f = p - j;
      const a = x[Math.min(j, x.length - 1)];
      const b = x[Math.min(j + 1, x.length - 1)];
      y[i] = a + f * (b - a);
    }
    return y;
  }

  /** 12 leads (any length, sampling rate fsIn, values in mV; use scale for µV input) -> model input
   *  Float32Array of shape [1, 12, nOut] (lead-major), centre-cropped or zero-padded to nOut samples. */
  function toModelInput(leads, fsIn, fsModel, nOut, scale) {
    const s = scale || 1;
    const out = new Float32Array(12 * nOut);
    for (let c = 0; c < 12; c++) {
      const r = resample(leads[c], fsIn, fsModel);
      const start = Math.max(0, Math.floor((r.length - nOut) / 2));
      const pad = Math.max(0, Math.floor((nOut - r.length) / 2));
      for (let t = 0; t < nOut; t++) {
        const k = start + t - pad;
        out[c * nOut + t] = k >= 0 && k < r.length ? r[k] * s : 0;
      }
    }
    return out;
  }

  /** Guess the unit: values in µV (|max| > 50) are scaled to mV. */
  function guessScale(leads) {
    let m = 0;
    for (const l of leads) for (const v of l) m = Math.max(m, Math.abs(v));
    return m > 50 ? 0.001 : 1;
  }

  const api = { parseCSV, resample, toModelInput, guessScale };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ECG = api;
})(typeof self !== "undefined" ? self : this);
