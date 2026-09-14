/* ============================================================
   ANN Course - a real (tiny) neural network, trained in the browser

     Net1D.mount(container, { target, xRange, yRange })

   Architecture: 1 input -> N hidden units -> 1 output.
   Trained by actual gradient descent (Adam) on mean squared error,
   in JavaScript, live. Nothing here is faked or pre-computed.

   It exists to make one idea physical: switch the activation to
   "none" and the network collapses to a straight line no matter how
   many hidden units you give it. Switch to ReLU and each unit buys
   you one more bend.
   ============================================================ */

(function (global) {
  'use strict';

  var ACTS = {
    none: { f: function (z) { return z; },        d: function () { return 1; } },
    relu: { f: function (z) { return z > 0 ? z : 0; }, d: function (z) { return z > 0 ? 1 : 0; } },
    tanh: { f: function (z) { return Math.tanh(z); },
            d: function (z) { var t = Math.tanh(z); return 1 - t * t; } }
  };

  function Net(n, act, xRange) {
    this.n = n;
    this.act = act;
    // Spread the initial "kink" positions (-b/w) across the input range so
    // every hidden unit starts somewhere useful.
    this.w1 = []; this.b1 = []; this.w2 = []; this.b2 = 0;
    for (var i = 0; i < n; i++) {
      var sign = Math.random() < 0.5 ? -1 : 1;
      var w = sign * (0.6 + Math.random() * 1.4);
      var kink = xRange[0] + (xRange[1] - xRange[0]) * (i + 0.5) / n;
      this.w1.push(w);
      this.b1.push(-w * kink);
      this.w2.push((Math.random() - 0.5) * 0.4);
    }
    // Adam moment buffers, one per parameter vector.
    this.m = { w1: z(n), b1: z(n), w2: z(n), b2: 0 };
    this.v = { w1: z(n), b1: z(n), w2: z(n), b2: 0 };
    this.t = 0;
    function z(k) { var a = []; for (var i = 0; i < k; i++) a.push(0); return a; }
  }

  // `mask` is an optional per-hidden-unit multiplier, used for inverted
  // dropout during training. Inference passes nothing and sees the full net.
  Net.prototype.forward = function (x, mask) {
    var a = ACTS[this.act], y = this.b2, h = [], zs = [];
    for (var i = 0; i < this.n; i++) {
      var zi = this.w1[i] * x + this.b1[i];
      var hi = a.f(zi);
      if (mask) hi *= mask[i];
      zs.push(zi); h.push(hi);
      y += this.w2[i] * hi;
    }
    return { y: y, h: h, z: zs };
  };

  Net.prototype.predict = function (x) { return this.forward(x).y; };

  Net.prototype.mse = function (xs, ys) {
    var s = 0;
    for (var i = 0; i < xs.length; i++) {
      var e = this.predict(xs[i]) - ys[i];
      s += e * e;
    }
    return s / xs.length;
  };

  // One full-batch Adam step over the sample points. Returns MSE.
  // opts: { l2, dropout } - both optional, both off by default.
  Net.prototype.step = function (xs, ys, lr, opts) {
    var n = this.n, a = ACTS[this.act];
    opts = opts || {};
    var l2 = opts.l2 || 0;
    var pDrop = opts.dropout || 0;
    var gw1 = [], gb1 = [], gw2 = [], gb2 = 0, loss = 0;
    for (var i = 0; i < n; i++) { gw1.push(0); gb1.push(0); gw2.push(0); }

    // One dropout mask per step (inverted dropout: surviving units are
    // scaled up by 1/(1-p) so the expected activation is unchanged).
    var mask = null;
    if (pDrop > 0) {
      mask = [];
      for (var m = 0; m < n; m++) {
        mask.push(Math.random() < pDrop ? 0 : 1 / (1 - pDrop));
      }
    }

    for (var k = 0; k < xs.length; k++) {
      var out = this.forward(xs[k], mask);
      var err = out.y - ys[k];
      loss += err * err;
      var g = 2 * err / xs.length;
      gb2 += g;
      for (var j = 0; j < n; j++) {
        gw2[j] += g * out.h[j];
        var scale = mask ? mask[j] : 1;
        var dh = g * this.w2[j] * scale * a.d(out.z[j]);
        gw1[j] += dh * xs[k];
        gb1[j] += dh;
      }
    }

    // L2 / weight decay: penalise large weights, leave biases alone.
    if (l2 > 0) {
      for (var q = 0; q < n; q++) {
        gw1[q] += 2 * l2 * this.w1[q];
        gw2[q] += 2 * l2 * this.w2[q];
      }
    }

    this.t++;
    var b1c = 1 - Math.pow(0.9, this.t), b2c = 1 - Math.pow(0.999, this.t);
    var self = this;

    function adam(key, idx, grad) {
      var m, v;
      if (idx === null) {
        self.m[key] = 0.9 * self.m[key] + 0.1 * grad;
        self.v[key] = 0.999 * self.v[key] + 0.001 * grad * grad;
        m = self.m[key] / b1c; v = self.v[key] / b2c;
        return lr * m / (Math.sqrt(v) + 1e-8);
      }
      self.m[key][idx] = 0.9 * self.m[key][idx] + 0.1 * grad;
      self.v[key][idx] = 0.999 * self.v[key][idx] + 0.001 * grad * grad;
      m = self.m[key][idx] / b1c; v = self.v[key][idx] / b2c;
      return lr * m / (Math.sqrt(v) + 1e-8);
    }

    for (var p = 0; p < n; p++) {
      this.w1[p] -= adam('w1', p, gw1[p]);
      this.b1[p] -= adam('b1', p, gb1[p]);
      this.w2[p] -= adam('w2', p, gw2[p]);
    }
    this.b2 -= adam('b2', null, gb2);

    return loss / xs.length;
  };

  /* --------------------------- the widget --------------------------- */

  var Net1D = {
    mount: function (selector, cfg) {
      var root = typeof selector === 'string' ? document.querySelector(selector) : selector;
      if (!root) return;

      var xRange = cfg.xRange || [-3, 3];
      var yRange = cfg.yRange || [-1.6, 1.6];
      var target = cfg.target;

      // Fixed training set: 60 points sampled from the target curve.
      var xs = [], ys = [];
      for (var i = 0; i < 60; i++) {
        var x = xRange[0] + (xRange[1] - xRange[0]) * i / 59;
        xs.push(x); ys.push(target(x));
      }

      root.innerHTML =
        '<h4>Live network &mdash; 1 input, N hidden units, 1 output</h4>' +
        '<canvas></canvas>' +
        '<div class="controls">' +
          '<label>activation' +
            '<select class="n1-act">' +
              '<option value="relu">ReLU</option>' +
              '<option value="none">none (linear)</option>' +
              '<option value="tanh">tanh</option>' +
            '</select>' +
          '</label>' +
          '<label>hidden units <input class="n1-n" type="range" min="1" max="24" value="8">' +
            '<span class="readout n1-nval">8</span></label>' +
          '<button class="btn n1-run" type="button">Train</button>' +
          '<button class="btn n1-reset" type="button">Reset</button>' +
          '<span class="readout n1-stat"></span>' +
        '</div>';

      var canvas = root.querySelector('canvas');
      var selAct = root.querySelector('.n1-act');
      var inpN   = root.querySelector('.n1-n');
      var lblN   = root.querySelector('.n1-nval');
      var btnRun = root.querySelector('.n1-run');
      var btnRst = root.querySelector('.n1-reset');
      var stat   = root.querySelector('.n1-stat');

      var net, loss = 0, running = false, raf = null;

      function rebuild() {
        net = new Net(parseInt(inpN.value, 10), selAct.value, xRange);
        loss = 0;
        stop();
        draw();
      }

      function stop() {
        running = false;
        if (raf) cancelAnimationFrame(raf);
        raf = null;
        btnRun.textContent = 'Train';
        btnRun.classList.remove('on');
      }

      function loop() {
        if (!running) return;
        for (var s = 0; s < 12; s++) loss = net.step(xs, ys, 0.05);
        draw();
        raf = requestAnimationFrame(loop);
      }

      function draw() {
        var t = Plot.theme();
        var f = Plot.fit(canvas, 2.0);
        var ctx = f.ctx, W = f.w, H = f.h;
        var pad = 8;

        var px = function (x) {
          return pad + (x - xRange[0]) / (xRange[1] - xRange[0]) * (W - pad * 2);
        };
        var py = function (y) {
          return H - pad - (y - yRange[0]) / (yRange[1] - yRange[0]) * (H - pad * 2);
        };

        ctx.clearRect(0, 0, W, H);

        // zero line
        ctx.strokeStyle = t.rule; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad, py(0)); ctx.lineTo(W - pad, py(0)); ctx.stroke();

        // target curve (what we are trying to learn)
        ctx.strokeStyle = t.faint;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (var i = 0; i <= 220; i++) {
          var x = xRange[0] + (xRange[1] - xRange[0]) * i / 220;
          var cx = px(x), cy = py(target(x));
          i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // what the network currently computes
        ctx.strokeStyle = cssAccent();
        ctx.lineWidth = 2.4;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (var j = 0; j <= 400; j++) {
          var xv = xRange[0] + (xRange[1] - xRange[0]) * j / 400;
          var yv = net.predict(xv);
          yv = Math.max(yRange[0] - 1, Math.min(yRange[1] + 1, yv));
          var qx = px(xv), qy = py(yv);
          j ? ctx.lineTo(qx, qy) : ctx.moveTo(qx, qy);
        }
        ctx.stroke();

        ctx.strokeStyle = t.rule;
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

        // legend
        ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillStyle = t.faint;
        ctx.fillText('- - target', pad + 4, pad + 2);
        ctx.fillStyle = cssAccent();
        ctx.fillText('— network', pad + 62, pad + 2);

        stat.textContent = 'epoch ' + net.t + '   loss ' + loss.toFixed(4);
      }

      function cssAccent() {
        return getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#1c5f8c';
      }

      btnRun.addEventListener('click', function () {
        if (running) { stop(); return; }
        running = true;
        btnRun.textContent = 'Pause';
        btnRun.classList.add('on');
        loop();
      });
      btnRst.addEventListener('click', rebuild);
      selAct.addEventListener('change', rebuild);
      inpN.addEventListener('input', function () {
        lblN.textContent = inpN.value;
        rebuild();
      });

      rebuild();
      Plot.live(canvas, draw);
    }
  };

  // Exposed so other lessons can build their own trainers on the same
  // network (see assets/overfit.js) instead of duplicating the maths.
  Net1D.Net = Net;

  global.Net1D = Net1D;
})(window);
