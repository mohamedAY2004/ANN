/* ============================================================
   ANN Course - the optimizer race

     Descent.mount(container)

   Four real optimizers walking down the same loss surface, from the
   same starting point, with the same learning rate. The surface is a
   deliberately ill-conditioned bowl - a "ravine", steep across and
   shallow along - because that is the situation every optimizer after
   plain SGD was invented to survive.

   The update rules below are the actual published ones, not
   approximations. Compare against Ruder (2016), sections 4.1-4.6.
   ============================================================ */

(function (global) {
  'use strict';

  var K = 12;                         // ravine steepness ratio
  var START = { x: -2.55, y: 1.05 };

  function loss(x, y) { return 0.5 * (x * x + K * y * y); }
  function grad(x, y) { return { x: x, y: K * y }; }

  var COLORS = {
    sgd:      '#d1495b',
    momentum: '#e0973c',
    rmsprop:  '#3fa06d',
    adam:     '#6a8ee8'
  };

  var LABELS = {
    sgd:      'SGD',
    momentum: 'Momentum',
    rmsprop:  'RMSProp',
    adam:     'Adam'
  };

  function makeRunner(kind, lr) {
    var p = { x: START.x, y: START.y };
    var v = { x: 0, y: 0 };           // momentum / first moment
    var s = { x: 0, y: 0 };           // squared-gradient accumulator
    var t = 0;
    var eps = 1e-8;

    return {
      kind: kind,
      path: [{ x: p.x, y: p.y }],
      dead: false,
      step: function () {
        if (this.dead) return;
        var g = grad(p.x, p.y);
        t++;

        if (kind === 'sgd') {
          p.x -= lr * g.x;
          p.y -= lr * g.y;

        } else if (kind === 'momentum') {
          // v = beta*v + grad ; theta -= lr*v
          v.x = 0.9 * v.x + g.x;
          v.y = 0.9 * v.y + g.y;
          p.x -= lr * v.x;
          p.y -= lr * v.y;

        } else if (kind === 'rmsprop') {
          // s = rho*s + (1-rho)*g^2 ; theta -= lr*g/sqrt(s)
          s.x = 0.9 * s.x + 0.1 * g.x * g.x;
          s.y = 0.9 * s.y + 0.1 * g.y * g.y;
          p.x -= lr * g.x / (Math.sqrt(s.x) + eps);
          p.y -= lr * g.y / (Math.sqrt(s.y) + eps);

        } else if (kind === 'adam') {
          // momentum AND per-parameter scaling, both bias-corrected
          v.x = 0.9 * v.x + 0.1 * g.x;
          v.y = 0.9 * v.y + 0.1 * g.y;
          s.x = 0.999 * s.x + 0.001 * g.x * g.x;
          s.y = 0.999 * s.y + 0.001 * g.y * g.y;
          var b1 = 1 - Math.pow(0.9, t), b2 = 1 - Math.pow(0.999, t);
          p.x -= lr * (v.x / b1) / (Math.sqrt(s.x / b2) + eps);
          p.y -= lr * (v.y / b1) / (Math.sqrt(s.y / b2) + eps);
        }

        if (!isFinite(p.x) || !isFinite(p.y) || Math.abs(p.x) > 12 || Math.abs(p.y) > 12) {
          this.dead = true;
          return;
        }
        this.path.push({ x: p.x, y: p.y });
      },
      loss: function () { return this.dead ? Infinity : loss(p.x, p.y); }
    };
  }

  var Descent = {
    mount: function (selector) {
      var root = typeof selector === 'string' ? document.querySelector(selector) : selector;
      if (!root) return;

      var enabled = { sgd: true, momentum: true, rmsprop: true, adam: true };
      var lr = 0.08;
      var runners = [];
      var playing = false, raf = null;

      var legend = Object.keys(LABELS).map(function (k) {
        return '<button type="button" class="btn on dc-tog" data-k="' + k + '">' +
               '<span class="swatch" style="background:' + COLORS[k] + '"></span>' +
               LABELS[k] + '</button>';
      }).join('');

      root.innerHTML =
        '<h4>Same surface, same start, same learning rate</h4>' +
        '<div class="dc-stage"><canvas class="dc-bg"></canvas><canvas class="dc-fg"></canvas></div>' +
        '<div class="controls" style="margin-top:.7rem">' + legend + '</div>' +
        '<div class="controls">' +
          '<label>learning rate <input class="dc-lr" type="range" min="1" max="40" value="8">' +
            '<span class="readout dc-lrv">0.080</span></label>' +
          '<button class="btn dc-play" type="button">Run</button>' +
          '<button class="btn dc-reset" type="button">Reset</button>' +
        '</div>' +
        '<div class="dc-table"></div>';

      if (!document.getElementById('dc-style')) {
        var st = document.createElement('style');
        st.id = 'dc-style';
        st.textContent =
          '.dc-stage{position:relative}' +
          '.dc-stage canvas{position:absolute;inset:0;width:100%}' +
          '.dc-stage .dc-bg{position:relative}' +
          '.dc-table{margin-top:.8rem;font-family:ui-monospace,Menlo,Consolas,monospace;' +
            'font-size:.74rem;line-height:1.75;color:var(--ink-soft)}' +
          '.dc-table .dead{color:var(--bad);font-weight:600}';
        document.head.appendChild(st);
      }

      var bg    = root.querySelector('.dc-bg');
      var fg    = root.querySelector('.dc-fg');
      var lrIn  = root.querySelector('.dc-lr');
      var lrOut = root.querySelector('.dc-lrv');
      var play  = root.querySelector('.dc-play');
      var reset = root.querySelector('.dc-reset');
      var table = root.querySelector('.dc-table');

      var XR = [-3, 3], YR = [-1.25, 1.25];

      Plot.contour(bg, {
        xRange: XR, yRange: YR, aspect: 2.4,
        f: function (x, y) { return loss(x, y); }
      });

      function build() {
        runners = Object.keys(LABELS).map(function (k) { return makeRunner(k, lr); });
        stop();
        drawPaths();
        drawTable();
      }

      function stop() {
        playing = false;
        if (raf) cancelAnimationFrame(raf);
        raf = null;
        play.textContent = 'Run';
        play.classList.remove('on');
      }

      function tick() {
        if (!playing) return;
        var done = true;
        runners.forEach(function (r) {
          if (r.path.length < 220 && !r.dead) { r.step(); done = false; }
        });
        drawPaths();
        drawTable();
        if (done) { stop(); return; }
        raf = requestAnimationFrame(tick);
      }

      function drawPaths() {
        var t = Plot.theme();
        var dpr = window.devicePixelRatio || 1;
        var w = bg.clientWidth, h = bg.clientHeight;
        fg.width = Math.round(w * dpr);
        fg.height = Math.round(h * dpr);
        fg.style.height = h + 'px';
        var ctx = fg.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        var px = function (x) { return (x - XR[0]) / (XR[1] - XR[0]) * w; };
        var py = function (y) { return (YR[1] - y) / (YR[1] - YR[0]) * h; };

        // the minimum we are all trying to reach
        ctx.strokeStyle = t.faint;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(px(0) - 5, py(0)); ctx.lineTo(px(0) + 5, py(0));
        ctx.moveTo(px(0), py(0) - 5); ctx.lineTo(px(0), py(0) + 5);
        ctx.stroke();

        runners.forEach(function (r) {
          if (!enabled[r.kind]) return;
          ctx.strokeStyle = COLORS[r.kind];
          ctx.lineWidth = 1.8;
          ctx.lineJoin = 'round';
          ctx.beginPath();
          r.path.forEach(function (p, i) {
            var cx = px(p.x), cy = py(p.y);
            i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
          });
          ctx.stroke();

          var last = r.path[r.path.length - 1];
          ctx.fillStyle = COLORS[r.kind];
          ctx.beginPath();
          ctx.arc(px(last.x), py(last.y), 3.2, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      function drawTable() {
        table.innerHTML = runners.map(function (r) {
          if (!enabled[r.kind]) return '';
          var name = (LABELS[r.kind] + '          ').slice(0, 10);
          if (r.dead) {
            return '<div class="dead">' + name + ' diverged &mdash; step size too large for the steep direction</div>';
          }
          var l = r.loss();
          return '<div>' + name + ' steps ' + String(r.path.length - 1) +
                 '   loss ' + (l < 0.001 ? l.toExponential(1) : l.toFixed(4)) + '</div>';
        }).join('');
      }

      root.querySelectorAll('.dc-tog').forEach(function (b) {
        b.addEventListener('click', function () {
          var k = b.dataset.k;
          enabled[k] = !enabled[k];
          b.classList.toggle('on', enabled[k]);
          drawPaths(); drawTable();
        });
      });

      lrIn.addEventListener('input', function () {
        lr = parseInt(lrIn.value, 10) / 100;
        lrOut.textContent = lr.toFixed(3);
        build();
      });

      play.addEventListener('click', function () {
        if (playing) { stop(); return; }
        playing = true;
        play.textContent = 'Pause';
        play.classList.add('on');
        tick();
      });

      reset.addEventListener('click', build);

      build();
      Plot.live(bg, function () { setTimeout(drawPaths, 0); });
    }
  };

  global.Descent = Descent;
})(window);
