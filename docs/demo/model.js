/* SUSS demo: the experiment-A transformer (4 blocks, d_model 128, 4 heads, 0.8M parameters) in plain
   JavaScript. An incremental forward pass with a key-value cache, the same computation as
   suss.model.GPT.step in PyTorch; tests/test_demo.py checks the two agree. Works as a browser script
   (window.SussModel) and as a CommonJS module for Node. */
(function (root) {
  "use strict";

  function buildModel(manifest, buffer) {
    var all = new Float32Array(buffer);
    var W = {};
    manifest.tensors.forEach(function (t) {
      var n = t.shape.reduce(function (a, b) { return a * b; }, 1);
      W[t.name] = all.subarray(t.offset, t.offset + n);
    });
    return { cfg: manifest.cfg, W: W };
  }

  function linear(w, b, x, out, nOut, nIn) {
    for (var o = 0; o < nOut; o++) {
      var s = b[o], row = o * nIn;
      for (var i = 0; i < nIn; i++) s += w[row + i] * x[i];
      out[o] = s;
    }
    return out;
  }

  function layerNorm(x, w, b, out, n) {
    var mean = 0, v = 0, i;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (i = 0; i < n; i++) { var d = x[i] - mean; v += d * d; }
    var inv = 1 / Math.sqrt(v / n + 1e-5);
    for (i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
    return out;
  }

  function erf(x) {  // Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7
    var s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }

  function newCache(model) {
    var c = model.cfg, K = [], V = [];
    for (var l = 0; l < c.n_layer; l++) {
      K.push(new Float64Array(c.T * c.d_model));
      V.push(new Float64Array(c.T * c.d_model));
    }
    return { K: K, V: V };
  }

  /* Forward one token `tok` (length d_in) at position t, attending to cached keys and values of positions
     0..t-1 and to itself. With store = true its keys and values are written to row t of the cache. */
  function step(model, cache, tok, t, store) {
    var c = model.cfg, W = model.W;
    var D = c.d_model, H = c.n_head, dh = D / H, M = c.d_mlp, din = c.d_in;
    var scale = 1 / Math.sqrt(dh);
    var h = new Float64Array(D), x = new Float64Array(D), qkv = new Float64Array(3 * D);
    var o = new Float64Array(D), tmp = new Float64Array(D), m1 = new Float64Array(M), sc = new Float64Array(t + 1);
    var ew = W["embed.weight"], eb = W["embed.bias"], pos = W["pos"];
    for (var i = 0; i < D; i++) {
      var s = eb[i] + pos[t * D + i];
      for (var j = 0; j < din; j++) s += ew[i * din + j] * tok[j];
      h[i] = s;
    }
    for (var l = 0; l < c.n_layer; l++) {
      var p = "blocks." + l + ".";
      layerNorm(h, W[p + "ln1.weight"], W[p + "ln1.bias"], x, D);
      linear(W[p + "qkv.weight"], W[p + "qkv.bias"], x, qkv, 3 * D, D);
      var K = cache.K[l], V = cache.V[l];
      if (store) {
        for (i = 0; i < D; i++) { K[t * D + i] = qkv[D + i]; V[t * D + i] = qkv[2 * D + i]; }
      }
      for (var hh = 0; hh < H; hh++) {
        var off = hh * dh, mx = -Infinity, r;
        for (r = 0; r <= t; r++) {
          var dot = 0;
          if (r < t) { for (j = 0; j < dh; j++) dot += qkv[off + j] * K[r * D + off + j]; }
          else { for (j = 0; j < dh; j++) dot += qkv[off + j] * qkv[D + off + j]; }
          dot *= scale;
          sc[r] = dot;
          if (dot > mx) mx = dot;
        }
        var z = 0;
        for (r = 0; r <= t; r++) { sc[r] = Math.exp(sc[r] - mx); z += sc[r]; }
        for (j = 0; j < dh; j++) {
          var acc = sc[t] * qkv[2 * D + off + j];
          for (r = 0; r < t; r++) acc += sc[r] * V[r * D + off + j];
          o[off + j] = acc / z;
        }
      }
      linear(W[p + "proj.weight"], W[p + "proj.bias"], o, tmp, D, D);
      for (i = 0; i < D; i++) h[i] += tmp[i];
      layerNorm(h, W[p + "ln2.weight"], W[p + "ln2.bias"], x, D);
      linear(W[p + "mlp.0.weight"], W[p + "mlp.0.bias"], x, m1, M, D);
      for (i = 0; i < M; i++) m1[i] = gelu(m1[i]);
      linear(W[p + "mlp.2.weight"], W[p + "mlp.2.bias"], m1, tmp, D, M);
      for (i = 0; i < D; i++) h[i] += tmp[i];
    }
    layerNorm(h, W["ln_f.weight"], W["ln_f.bias"], x, D);
    return linear(W["head.weight"], W["head.bias"], x, new Float64Array(c.d_out), c.d_out, D);
  }

  /* Predictions for every position of a token sequence (flat, length T * d_in), and the filled cache. */
  function forwardAll(model, tokens, T) {
    var din = model.cfg.d_in, cache = newCache(model), preds = [];
    for (var t = 0; t < T; t++) preds.push(step(model, cache, tokens.subarray(t * din, (t + 1) * din), t, true));
    return { preds: preds, cache: cache };
  }

  /* d prediction_t / d token_t (d_out x d_in), central differences, context held fixed. For this model it
     is the parameter estimate [A B] that the network applies to the query. */
  function jacobianAt(model, cache, tokens, t, eps) {
    eps = eps || 1e-3;
    var din = model.cfg.d_in, dout = model.cfg.d_out, J = [], i, j;
    for (i = 0; i < dout; i++) J.push(new Float64Array(din));
    for (j = 0; j < din; j++) {
      var a = Float64Array.from(tokens.subarray(t * din, (t + 1) * din)), b = Float64Array.from(a);
      a[j] += eps; b[j] -= eps;
      var fa = step(model, cache, a, t, false), fb = step(model, cache, b, t, false);
      for (i = 0; i < dout; i++) J[i][j] = (fa[i] - fb[i]) / (2 * eps);
    }
    return J;
  }

  var api = { buildModel: buildModel, step: step, forwardAll: forwardAll, jacobianAt: jacobianAt, newCache: newCache };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SussModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
