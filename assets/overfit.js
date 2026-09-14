/* ============================================================
   ANN Course - overfitting, and the three tools against it

     Overfit.mount(container)

   A genuinely over-parameterised network (32 hidden units) fitted to
   only 14 noisy points, trained live with Adam. You watch the training
   loss keep falling while the validation loss turns around and climbs -
   the actual signature of overfitting, not a drawing of one.

   Then you switch on L2, dropout and early stopping and watch the gap
   close. Built on the same network as assets/net1d.js.
   ============================================================ */

(function (global) {
  'use strict';

  var COL = { train: '#6a8ee8', val: '#d1495b' };

  function truth(x) { return 0.8 * Math.sin(1.7 * x); }

  // Deterministic pseudo-noise so Reset reproduces the same dataset and
  // any difference you see really is the regularizer, not luck.
  function rng(seed) {
    var s = seed;
    return function () {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
  }

  function gauss(rand) {
    var u = Math.max(1e-9, rand()), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function makeData() {
    var r = rng(20260914);
    var tr = { x: [], y: [] }, va = { x: [], y: [] };
    var i;
    for (i = 0; i < 14; i++) {
      var x = -2.5 + 5 * i / 13;
      tr.x.push(x);
      tr.y.push(truth(x) + gauss(r) * 0.18);
    }
    for (i = 0; i < 60; i++) {
      var xv = -2.5 + 5 * (i + 0.5) / 60;
      va.x.push(xv);
      va.y.push(truth(xv) + gauss(r) * 0.18);
    }
    return { tr: tr, va: va };
  }

  var Overfit = {
    mount: function (selector) {
      var root = typeof selector === 'string' ? document.querySelector(selector) : selector;
      if (!root) return;

      var D = makeData();
      var opts = { l2: false, dropout: false, early: false };
      // Tuned against this exact dataset (see the sweep in the session notes):
      // L2 at 0.002 is the genuine optimum here; dropout at 0.15 is deliberately
      // a touch strong, so you can see a regularizer over-correct.
      var L2_STRENGTH = 0.002, DROP_RATE = 0.15, PATIENCE = 60, MAX_EPOCH = 900;

      var net, hist, best, stopped, playing = false, raf = null;

      root.innerHTML =
        '<h4>32 hidden units, 14 noisy training points &mdash; trained live</h4>' +
        '<div class="of-grid">' +
          '<div><canvas class="of-fit"></canvas>' +
            '<p class="of-cap">the fit &mdash; <span style="color:var(--ink-faint)">- - true function</span>, ' +
            '&#9679; training data, <span style="color:var(--accent)">&mdash; network</span></p></div>' +
          '<div><canvas class="of-loss"></canvas>' +
            '<p class="of-cap"><span style="color:' + COL.train + '">&mdash; training loss</span> &nbsp; ' +
            '<span style="color:' + COL.val + '">&mdash; validation loss</span> (log scale)</p></div>' +
        '</div>' +
        '<div class="controls">' +
          '<button class="btn of-t" data-k="l2" type="button">L2 penalty</button>' +
          '<button class="btn of-t" data-k="dropout" type="button">Dropout</button>' +
          '<button class="btn of-t" data-k="early" type="button">Early stopping</button>' +
          '<button class="btn of-run" type="button">Train</button>' +
          '<button class="btn of-reset" type="button">Reset</button>' +
        '</div>' +
        '<p class="of-stat"></p>';

      if (!document.getElementById('of-style')) {
        var st = document.createElement('style');
        st.id = 'of-style';
        st.textContent =
          '.of-grid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}' +
          '@media(max-width:600px){.of-grid{grid-template-columns:1fr}}' +
          '.of-cap{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.68rem;' +
            'color:var(--ink-faint);margin:.4rem 0 0;text-align:center;line-height:1.5}' +
          '.of-stat{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.76rem;' +
            'background:var(--paper);border:1px solid var(--rule);border-radius:5px;' +
            'padding:.6rem .75rem;margin:.9rem 0 0;line-height:1.7;color:var(--ink-soft)}' +
          '.of-stat .flag{color:var(--bad);font-weight:600}' +
          '.of-stat .ok{color:var(--good);font-weight:600}';
        document.head.appendChild(st);
      }

      var cvFit  = root.querySelector('.of-fit');
      var cvLoss = root.querySelector('.of-loss');
      var btnRun = root.querySelector('.of-run');
      var btnRst = root.querySelector('.of-reset');
      var stat   = root.querySelector('.of-stat');

      function snapshot(n) {
        return { w1: n.w1.slice(), b1: n.b1.slice(), w2: n.w2.slice(), b2: n.b2 };
      }
      function restore(n, s) {
        n.w1 = s.w1.slice(); n.b1 = s.b1.slice(); n.w2 = s.w2.slice(); n.b2 = s.b2;
      }

      function build() {
        net = new Net1D.Net(32, 'tanh', [-2.6, 2.6]);
        hist = [];
        best = { val: Infinity, epoch: 0, weights: snapshot(net) };
        stopped = null;
        stop();
        draw();
      }

      function stop() {
        playing = false;
        if (raf) cancelAnimationFrame(raf);
        raf = null;
        btnRun.textContent = 'Train';
        btnRun.classList.remove('on');
      }

      function epoch() {
        net.step(D.tr.x, D.tr.y, 0.02, {
          l2: opts.l2 ? L2_STRENGTH : 0,
          dropout: opts.dropout ? DROP_RATE : 0
        });
        var trL = net.mse(D.tr.x, D.tr.y);
        var vaL = net.mse(D.va.x, D.va.y);
        hist.push({ t: trL, v: vaL });

        if (vaL < best.val - 1e-5) {
          best = { val: vaL, epoch: hist.length, weights: snapshot(net) };
        }
        if (opts.early && hist.length - best.epoch > PATIENCE && !stopped) {
          stopped = hist.length;
          restore(net, best.weights);   // restore_best_weights=True
          return false;
        }
        return hist.length < MAX_EPOCH;
      }

      function loop() {
        if (!playing) return;
        var alive = true;
        for (var i = 0; i < 6 && alive; i++) alive = epoch();
        draw();
        if (!alive) { stop(); return; }
        raf = requestAnimationFrame(loop);
      }

      function drawFit() {
        var t = Plot.theme();
        var f = Plot.fit(cvFit, 1.25);
        var ctx = f.ctx, W = f.w, H = f.h, pad = 10;
        var XR = [-2.8, 2.8], YR = [-1.5, 1.5];
        var px = function (x) { return pad + (x - XR[0]) / (XR[1] - XR[0]) * (W - 2 * pad); };
        var py = function (y) { return H - pad - (y - YR[0]) / (YR[1] - YR[0]) * (H - 2 * pad); };

        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = t.rule; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad, py(0)); ctx.lineTo(W - pad, py(0)); ctx.stroke();

        // the underlying truth
        ctx.strokeStyle = t.faint; ctx.lineWidth = 1.8;
        ctx.setLineDash([5, 4]); ctx.beginPath();
        for (var i = 0; i <= 200; i++) {
          var x = XR[0] + (XR[1] - XR[0]) * i / 200;
          i ? ctx.lineTo(px(x), py(truth(x))) : ctx.moveTo(px(x), py(truth(x)));
        }
        ctx.stroke(); ctx.setLineDash([]);

        // what the network learned
        ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent').trim();
        ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
        for (var j = 0; j <= 320; j++) {
          var xv = XR[0] + (XR[1] - XR[0]) * j / 320;
          var yv = Math.max(YR[0] - 1, Math.min(YR[1] + 1, net.predict(xv)));
          j ? ctx.lineTo(px(xv), py(yv)) : ctx.moveTo(px(xv), py(yv));
        }
        ctx.stroke();

        // the 14 noisy points it is being asked to explain
        ctx.fillStyle = t.ink;
        for (var k = 0; k < D.tr.x.length; k++) {
          ctx.beginPath();
          ctx.arc(px(D.tr.x[k]), py(D.tr.y[k]), 3.4, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.strokeStyle = t.rule; ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
      }

      function drawLoss() {
        var t = Plot.theme();
        var f = Plot.fit(cvLoss, 1.25);
        var ctx = f.ctx, W = f.w, H = f.h;
        var pad = { l: 34, r: 8, t: 8, b: 20 };
        var lo = -3, hi = 0.3;                       // log10 loss range
        var px = function (i) {
          return pad.l + (i / MAX_EPOCH) * (W - pad.l - pad.r);
        };
        var py = function (v) {
          var L = Math.log(Math.max(v, 1e-4)) / Math.LN10;
          L = Math.max(lo, Math.min(hi, L));
          return H - pad.b - (L - lo) / (hi - lo) * (H - pad.t - pad.b);
        };

        ctx.clearRect(0, 0, W, H);
        ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = t.faint; ctx.strokeStyle = t.rule; ctx.lineWidth = 1;
        [1, 0.1, 0.01, 0.001].forEach(function (v) {
          ctx.beginPath(); ctx.moveTo(pad.l, py(v)); ctx.lineTo(W - pad.r, py(v)); ctx.stroke();
          ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
          ctx.fillText(String(v), pad.l - 4, py(v));
        });

        ['t', 'v'].forEach(function (key) {
          if (!hist.length) return;
          ctx.strokeStyle = key === 't' ? COL.train : COL.val;
          ctx.lineWidth = 1.9; ctx.beginPath();
          hist.forEach(function (h, i) {
            var cx = px(i), cy = py(h[key]);
            i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
          });
          ctx.stroke();
        });

        // where validation loss bottomed out - the epoch you actually wanted
        if (hist.length && isFinite(best.val)) {
          ctx.strokeStyle = t.faint;
          ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px(best.epoch), pad.t); ctx.lineTo(px(best.epoch), H - pad.b);
          ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = t.faint; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
          ctx.fillText('best', px(best.epoch) + 3, pad.t + 1);
        }

        if (stopped) {
          ctx.strokeStyle = COL.val; ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(px(stopped), pad.t); ctx.lineTo(px(stopped), H - pad.b);
          ctx.stroke();
          ctx.fillStyle = COL.val; ctx.textAlign = 'right';
          ctx.fillText('stopped', px(stopped) - 3, pad.t + 1);
        }

        ctx.strokeStyle = t.rule; ctx.lineWidth = 1;
        ctx.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
      }

      function draw() {
        drawFit();
        drawLoss();

        var last = hist[hist.length - 1];
        if (!last) { stat.textContent = 'epoch 0 - press Train'; return; }
        var gap = last.v / Math.max(last.t, 1e-6);
        var verdict = stopped
          ? '<span class="ok">early stopping fired at epoch ' + stopped +
            ', weights rolled back to epoch ' + best.epoch + '</span>'
          : (gap > 6
            ? '<span class="flag">validation loss is ' + gap.toFixed(0) +
              'x the training loss &mdash; it is memorising the noise</span>'
            : 'train and validation still tracking together');

        stat.innerHTML =
          'epoch ' + hist.length +
          '   train ' + last.t.toFixed(4) +
          '   val ' + last.v.toFixed(4) +
          '   best val ' + (isFinite(best.val) ? best.val.toFixed(4) : '-') +
          ' @ epoch ' + best.epoch + '<br>' + verdict;
      }

      root.querySelectorAll('.of-t').forEach(function (b) {
        b.addEventListener('click', function () {
          var k = b.dataset.k;
          opts[k] = !opts[k];
          b.classList.toggle('on', opts[k]);
          build();
        });
      });

      btnRun.addEventListener('click', function () {
        if (playing) { stop(); return; }
        if (hist.length >= MAX_EPOCH || stopped) build();
        playing = true;
        btnRun.textContent = 'Pause';
        btnRun.classList.add('on');
        loop();
      });
      btnRst.addEventListener('click', build);

      build();
      Plot.live(cvFit, draw);
    }
  };

  global.Overfit = Overfit;
})(window);
