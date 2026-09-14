/* ============================================================
   ANN Course - reusable retrieval-practice widgets
   Shared by every lesson. Two components:

     Quiz.mount(el, { title, questions: [{ q, options, answer, why }] })
     Recall.mount(el, { title, items: [{ prompt, answer }] })

   Quiz gives immediate feedback (tight loop) and shuffles options
   on every load so repeat visits are genuine retrieval, not
   position memory. Recall is free-recall-then-reveal: answer out
   loud BEFORE clicking, which is where storage strength comes from.
   ============================================================ */

(function (global) {
  'use strict';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.innerHTML = text;
    return n;
  }

  function shuffle(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  /* ---------------------------- Quiz ---------------------------- */

  var Quiz = {
    mount: function (selector, config) {
      var root = typeof selector === 'string' ? document.querySelector(selector) : selector;
      if (!root) return;

      var questions = config.questions || [];
      var title = config.title || 'Check yourself';
      var idx = 0, score = 0;

      root.className = 'quiz';

      function render() {
        root.innerHTML = '';

        if (idx >= questions.length) { renderDone(); return; }

        var q = questions[idx];

        var head = el('div', 'quiz-head');
        head.appendChild(el('span', null, title));
        head.appendChild(el('span', null, (idx + 1) + ' / ' + questions.length));
        root.appendChild(head);

        var qp = el('p', 'quiz-q',
          '<span class="qnum">' + (idx + 1) + '.</span>' + q.q);
        root.appendChild(qp);

        // Pair each option with whether it is the right one, then shuffle.
        var pairs = q.options.map(function (text, i) {
          return { text: text, right: i === q.answer };
        });
        pairs = shuffle(pairs);

        var opts = el('div', 'quiz-opts');
        var buttons = [];

        pairs.forEach(function (p) {
          var b = el('button', 'quiz-opt', p.text);
          b.type = 'button';
          b.addEventListener('click', function () { choose(p, b); });
          buttons.push(b);
          opts.appendChild(b);
        });
        root.appendChild(opts);

        function choose(picked, button) {
          buttons.forEach(function (b) { b.disabled = true; });
          buttons.forEach(function (b, i) {
            if (pairs[i].right) b.classList.add('correct');
          });
          if (!picked.right) button.classList.add('wrong');
          else score++;

          var why = el('div', 'quiz-why ' + (picked.right ? 'ok' : 'no'));
          why.innerHTML =
            '<span class="verdict">' + (picked.right ? 'Correct.' : 'Not quite.') +
            '</span> ' + q.why;
          root.appendChild(why);

          var nav = el('div', 'quiz-nav');
          nav.appendChild(el('span', 'readout', 'score ' + score + '/' + (idx + 1)));
          var next = el('button', 'btn',
            idx + 1 < questions.length ? 'Next question' : 'See result');
          next.type = 'button';
          next.addEventListener('click', function () { idx++; render(); });
          nav.appendChild(next);
          root.appendChild(nav);
          next.focus();
        }
      }

      function renderDone() {
        var pct = Math.round((score / questions.length) * 100);
        var msg;
        if (pct === 100) msg = 'Clean sweep. This one has landed.';
        else if (pct >= 60) msg = 'Solid. Re-read the sections behind the misses, then run it again.';
        else msg = 'Worth another pass through the lesson before moving on - that is the point of the test, not a bad sign.';

        var done = el('div', 'quiz-done');
        done.appendChild(el('div', 'quiz-score', score + ' / ' + questions.length));
        done.appendChild(el('p', 'quiz-msg', msg));

        var again = el('button', 'btn', 'Run it again (options reshuffle)');
        again.type = 'button';
        again.addEventListener('click', function () { idx = 0; score = 0; render(); });
        done.appendChild(again);

        root.appendChild(done);
      }

      render();
    }
  };

  /* --------------------------- Recall --------------------------- */

  var Recall = {
    mount: function (selector, config) {
      var root = typeof selector === 'string' ? document.querySelector(selector) : selector;
      if (!root) return;

      root.className = 'recall';
      root.appendChild(el('span', 'label',
        config.title || 'Say it out loud first, then reveal'));

      (config.items || []).forEach(function (item) {
        var wrap = el('div', 'recall-item');
        wrap.appendChild(el('p', 'recall-prompt', item.prompt));

        var btn = el('button', 'btn', 'Reveal');
        btn.type = 'button';
        var ans = el('div', 'recall-ans', item.answer);
        ans.hidden = true;

        btn.addEventListener('click', function () {
          ans.hidden = !ans.hidden;
          btn.textContent = ans.hidden ? 'Reveal' : 'Hide';
        });

        wrap.appendChild(btn);
        wrap.appendChild(ans);
        root.appendChild(wrap);
      });
    }
  };

  global.Quiz = Quiz;
  global.Recall = Recall;
})(window);
