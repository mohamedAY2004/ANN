/* ============================================================
   ANN Course - activation function explorer

     ActViz.mount(container)

   Draws f(x) next to f'(x) for each activation, because the second
   plot is the one that decides whether a deep network can learn.
   The depth slider turns "vanishing gradient" from a phrase into a
   number you can watch fall off a cliff.
   ============================================================ */

(function (global) {
  'use strict';

  function sig(x) { return 1 / (1 + Math.exp(-x)); }

  var FNS = {
    sigmoid: {
      label: 'Sigmoid',
      f: sig,
      d: function (x) { var s = sig(x); return s * (1 - s); },
      maxSlope: 0.25,
      range: '0 to 1',
      note: 'Saturates at both ends. Its steepest slope anywhere is 0.25, so every layer it passes through shrinks the gradient by at least 4x.'
    },
    tanh: {
      label: 'Tanh',
      f: function (x) { return Math.tanh(x); },
      d: function (x) { var t = Math.tanh(x); return 1 - t * t; },
      maxSlope: 1,
      range: '-1 to 1',
      note: 'Zero-centred, which helps, and its peak slope is 1 rather than 0.25. But it still saturates: push the input past about +/-3 and the gradient is gone.'
    },
    relu: {
      label: 'ReLU',
      f: function (x) { return x > 0 ? x : 0; },
      d: function (x) { return x > 0 ? 1 : 0; },
      maxSlope: 1,
      range: '0 to infinity',
      note: 'No saturation on the positive side at all: the gradient is exactly 1, forever. That is why deep networks became trainable. The cost is the flat left half - a unit stuck there gets zero gradient and may never recover.'
    },
    leaky: {
      label: 'Leaky ReLU',
      f: function (x) { return x > 0 ? x : 0.1 * x; },
      d: function (x) { return x > 0 ? 1 : 0.1; },
      maxSlope: 1,
      range: '-infinity to infinity',
      note: 'ReLU with the flat half tilted slightly. A dead unit still receives a small gradient, so it has a route back to life. Keras calls the tilt negative_slope.'
    }
  };

  var ActViz = {
    mount: function (selector) {
      var root = typeof selector === 'string' ? document.querySelector(selector) : selector;
      if (!root) return;

      var current = 'sigmoid';

      root.innerHTML =
        '<h4>The function, and the slope that has to travel back through it</h4>' +
        '<div class="controls av-pick" style="margin:0 0 .85rem"></div>' +
        '<div class="av-grid">' +
          '<div><canvas class="av-f"></canvas>' +
            '<p class="av-cap">f(x) &mdash; what the neuron outputs</p></div>' +
          '<div><canvas class="av-d"></canvas>' +
            '<p class="av-cap">f&prime;(x) &mdash; how much gradient survives</p></div>' +
        '</div>' +
        '<div class="controls">' +
          '<label>network depth <input class="av-depth" type="range" min="1" max="30" value="10">' +
          '<span class="readout av-dval">10</span> layers</label>' +
        '</div>' +
        '<p class="av-verdict"></p>' +
        '<p class="av-note"></p>';

      // scoped layout for the twin plots
      if (!document.getElementById('av-style')) {
        var st = document.createElement('style');
        st.id = 'av-style';
        st.textContent =
          '.av-grid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}' +
          '@media(max-width:560px){.av-grid{grid-template-columns:1fr}}' +
          '.av-cap{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.7rem;' +
            'color:var(--ink-faint);margin:.4rem 0 0;text-align:center}' +
          '.av-verdict{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.78rem;' +
            'background:var(--paper);border:1px solid var(--rule);border-radius:5px;' +
            'padding:.6rem .75rem;margin:.9rem 0 .6rem;line-height:1.6}' +
          '.av-note{font-size:.87rem;color:var(--ink-soft);margin:0;line-height:1.55}';
        document.head.appendChild(st);
      }

      var pick    = root.querySelector('.av-pick');
      var cvF     = root.querySelector('.av-f');
      var cvD     = root.querySelector('.av-d');
      var depth   = root.querySelector('.av-depth');
      var dval    = root.querySelector('.av-dval');
      var verdict = root.querySelector('.av-verdict');
      var note    = root.querySelector('.av-note');

      Object.keys(FNS).forEach(function (key) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn' + (key === current ? ' on' : '');
        b.textContent = FNS[key].label;
        b.dataset.key = key;
        b.addEventListener('click', function () {
          current = key;
          pick.querySelectorAll('.btn').forEach(function (x) {
            x.classList.toggle('on', x.dataset.key === key);
          });
          redraw();
        });
        pick.appendChild(b);
      });

      var plotF = Plot.curve(cvF, {
        xRange: [-5, 5], yRange: [-1.3, 2.4],
        xTicks: [-4, -2, 0, 2, 4], yTicks: [0, 1, 2],
        aspect: 1.35,
        series: [{ fn: function (x) { return FNS[current].f(x); },
                   color: accent(), width: 2.4 }]
      });

      var plotD = Plot.curve(cvD, {
        xRange: [-5, 5], yRange: [-0.1, 1.15],
        xTicks: [-4, -2, 0, 2, 4], yTicks: [0, 0.25, 0.5, 1],
        aspect: 1.35,
        series: [{ fn: function (x) { return FNS[current].d(x); },
                   color: bad(), width: 2.4 }]
      });

      function accent() {
        return getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#1c5f8c';
      }
      function bad() {
        return getComputedStyle(document.body).getPropertyValue('--bad').trim() || '#a32b2b';
      }

      function fmt(v) {
        if (v >= 0.01) return v.toFixed(3);
        return v.toExponential(1);
      }

      function redraw() {
        var a = FNS[current];
        var n = parseInt(depth.value, 10);
        dval.textContent = n;

        plotF.redraw();
        plotD.redraw();

        // Backprop multiplies by f'(z) once per layer. Using the best case
        // (the steepest the slope ever gets) gives an upper bound on what
        // can survive the trip back.
        var best = Math.pow(a.maxSlope, n);
        var line1 = 'steepest slope of f&prime; anywhere ...... ' + a.maxSlope;
        var line2 = 'best case after ' + n + ' layers ......... &times;' + fmt(best);
        var line3 = best < 1e-4
          ? '&rarr; the early layers receive essentially nothing. They stop learning.'
          : (best >= 0.999
            ? '&rarr; the signal arrives intact. Depth stays trainable.'
            : '&rarr; weakened but alive.');
        verdict.innerHTML = line1 + '<br>' + line2 + '<br><strong>' + line3 + '</strong>';
        note.innerHTML = '<strong>' + a.label + '</strong> &mdash; output range ' +
          a.range + '. ' + a.note;
      }

      depth.addEventListener('input', redraw);
      redraw();
    }
  };

  global.ActViz = ActViz;
})(window);
