// Runs docs/demo/model.js under Node for tests/test_demo.py: prints predictions and query Jacobians as JSON.
// usage: node tests/demo_check.js weights.json weights.bin input.json
const fs = require("fs");
const path = require("path");
const M = require(path.join(__dirname, "..", "docs", "demo", "model.js"));

const [, , manifestPath, binPath, inputPath] = process.argv;
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const raw = fs.readFileSync(binPath);
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const model = M.buildModel(manifest, buf);
const input = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const tokens = Float64Array.from(input.tokens);
const { preds, cache } = M.forwardAll(model, tokens, input.T);
const jac = input.jac_at.map((t) => M.jacobianAt(model, cache, tokens, t).map((row) => Array.from(row)));
process.stdout.write(JSON.stringify({ preds: preds.map((p) => Array.from(p)), jac }));
