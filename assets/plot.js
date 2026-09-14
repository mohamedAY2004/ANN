/* ============================================================
   ANN Course - reusable canvas plotting
   Shared by every lesson that draws a curve or a surface.

     Plot.curve(canvas, spec)    - one or more y = f(x) curves on axes
     Plot.contour(canvas, spec)  - filled contour map of z = f(x, y)
     Plot.theme()                - current palette pulled from course.css

   Everything is redrawn on resize and on light/dark theme change, so
   a plot authored once looks right in both themes and at phone width.
   ============================================================ */

(function (global) {
  'use strict';

  function cssVar(name, fallback) {
    var v = getComputedStyle(document.body).getPropertyValue(name).trim();
    return v || fallback;
  }

  function theme() {
    return {
      ink:   cssVar('--ink', '#1c1b19'),
      soft:  cssVar('--ink-soft', '#57534a'),
      faint: cssVar('--ink-faint', '#8a857a'),
      rule:  cssVar('--rule', '#e2ddd1'),
      paper: cssVar('--paper', '#fffdf8')
    };
  }

  // Size the backing store for the device pixel ratio so lines stay crisp.
  function fit(canvas, aspect) {
    var dpr = global.devicePixelRatio || 1;
    var w = canvas.clientWidth || canvas.parentNode.clientWidth || 560;
    var h = Math.round(w / (aspect || 1.9));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + 'px';
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  // Re-run `draw` whenever the canvas resizes or the theme flips.
  function live(canvas, draw) {
    draw();
    if (global.ResizeObserver) {
      var ro = new ResizeObserver(function () { draw(); });
      ro.observe(canvas.parentNode || canvas);
    } else {
      global.addEventListener('resize', draw);
    }
    var mq = global.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', draw);
  }

  /* ---------------------------- curves ---------------------------- */

  var Plot = {
    theme: theme,
    fit: fit,
    live: live,

    /*  spec = {
          xRange: [lo, hi], yRange: [lo, hi],
          xTicks: [..], yTicks: [..],
          aspect: 1.9,
          series: [{ fn, color, width, dash, label }],
          xLabel, yLabel
        }                                                            */
    curve: function (canvas, spec) {
      function draw() {
        var t = theme();
        var f = fit(canvas, spec.aspect);
        var ctx = f.ctx, W = f.w, H = f.h;

        var pad = { l: 38, r: 12, t: 12, b: 26 };
        var x0 = spec.xRange[0], x1 = spec.xRange[1];
        var y0 = spec.yRange[0], y1 = spec.yRange[1];

        var px = function (x) { return pad.l + (x - x0) / (x1 - x0) * (W - pad.l - pad.r); };
        var py = function (y) { return H - pad.b - (y - y0) / (y1 - y0) * (H - pad.t - pad.b); };

        ctx.clearRect(0, 0, W, H);

        // grid
        ctx.strokeStyle = t.rule;
        ctx.lineWidth = 1;
        ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillStyle = t.faint;

        (spec.xTicks || []).forEach(function (x) {
          ctx.beginPath();
          ctx.moveTo(px(x), pad.t); ctx.lineTo(px(x), H - pad.b);
          ctx.stroke();
          ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillText(String(x), px(x), H - pad.b + 5);
        });

        (spec.yTicks || []).forEach(function (y) {
          ctx.beginPath();
          ctx.moveTo(pad.l, py(y)); ctx.lineTo(W - pad.r, py(y));
          ctx.stroke();
          ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
          ctx.fillText(String(y), pad.l - 6, py(y));
        });

        // axes through the origin where visible
        ctx.strokeStyle = t.faint;
        ctx.lineWidth = 1.2;
        if (y0 < 0 && y1 > 0) {
          ctx.beginPath(); ctx.moveTo(pad.l, py(0)); ctx.lineTo(W - pad.r, py(0)); ctx.stroke();
        }
        if (x0 < 0 && x1 > 0) {
          ctx.beginPath(); ctx.moveTo(px(0), pad.t); ctx.lineTo(px(0), H - pad.b); ctx.stroke();
        }

        // series
        (spec.series || []).forEach(function (s) {
          if (s.hidden) return;
          ctx.save();
          ctx.beginPath();
          ctx.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
          ctx.clip();

          ctx.strokeStyle = s.color;
          ctx.lineWidth = s.width || 2;
          ctx.setLineDash(s.dash || []);
          ctx.lineJoin = 'round';
          ctx.beginPath();

          var N = 400, started = false;
          for (var i = 0; i <= N; i++) {
            var x = x0 + (x1 - x0) * i / N;
            var y = s.fn(x);
            if (!isFinite(y)) { started = false; continue; }
            var cx = px(x), cy = py(y);
            if (!started) { ctx.moveTo(cx, cy); started = true; }
            else ctx.lineTo(cx, cy);
          }
          ctx.stroke();
          ctx.restore();
        });

        // frame
        ctx.setLineDash([]);
        ctx.strokeStyle = t.rule;
        ctx.lineWidth = 1;
        ctx.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);

        if (spec.xLabel) {
          ctx.fillStyle = t.faint;
          ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
          ctx.fillText(spec.xLabel, W - pad.r - 4, H - pad.b - 4);
        }
      }

      live(canvas, draw);
      return { redraw: draw };
    },

    /*  spec = { xRange, yRange, f(x,y), levels, aspect, palette }   */
    contour: function (canvas, spec) {
      function draw() {
        var t = theme();
        var f = fit(canvas, spec.aspect);
        var ctx = f.ctx, W = f.w, H = f.h;

        var x0 = spec.xRange[0], x1 = spec.xRange[1];
        var y0 = spec.yRange[0], y1 = spec.yRange[1];

        var img = ctx.createImageData(W, H);
        var d = img.data;
        var dark = t.paper.toLowerCase() !== '#fffdf8';

        // Find the value range once so shading uses the full ramp.
        var vmax = 0;
        for (var sy = 0; sy < H; sy += 4) {
          for (var sx = 0; sx < W; sx += 4) {
            var v = spec.f(x0 + (x1 - x0) * sx / W, y1 - (y1 - y0) * sy / H);
            if (v > vmax) vmax = v;
          }
        }

        for (var j = 0; j < H; j++) {
          for (var i = 0; i < W; i++) {
            var xv = x0 + (x1 - x0) * i / W;
            var yv = y1 - (y1 - y0) * j / H;
            var z = spec.f(xv, yv);
            // sqrt compresses the huge outer values so inner rings stay visible
            var u = Math.sqrt(Math.max(0, z) / vmax);
            // banding makes the "contour lines" read without tracing them
            var band = (Math.sin(u * Math.PI * 14) * 0.5 + 0.5) * 0.13;
            var s = Math.min(1, u + band);
            var p = (j * W + i) * 4;
            if (dark) {
              d[p] = 20 + s * 46; d[p + 1] = 24 + s * 52; d[p + 2] = 32 + s * 62;
            } else {
              d[p] = 255 - s * 56; d[p + 1] = 253 - s * 66; d[p + 2] = 246 - s * 60;
            }
            d[p + 3] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);

        ctx.strokeStyle = t.rule;
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

        if (spec.onDraw) {
          spec.onDraw(ctx, {
            W: W, H: H,
            px: function (x) { return (x - x0) / (x1 - x0) * W; },
            py: function (y) { return (y1 - y) / (y1 - y0) * H; },
            theme: t
          });
        }
      }

      live(canvas, draw);
      return { redraw: draw };
    }
  };

  global.Plot = Plot;
})(window);
