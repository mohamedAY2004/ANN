# Working notes

## How the user wants to be taught
- Asked explicitly: **fundamentals, not depth.** Intuition first; skip heavy derivations.
- Four topics requested in session 1: ANN intuition, activation functions, optimizers, TensorFlow/Keras features (callbacks + regularization).
- Working from the Samsung Innovation Campus AI Course deck, "Understanding Artificial Neural Networks" (165 slides).

## Teaching decisions made
- **The spine of the whole course:** a network is a stack of linear maps separated by non-linearities, trained by walking downhill. Every later design choice (activation, optimizer, regularizer, callback) is an answer to a specific failure mode of that process. Every lesson ties back to this.
- Lessons are kept to one idea each. The deck spreads one idea over 10+ slides; we compress.
- The deck contains at least one outright error (see `learning-records/0002`). Flag deck errors in a `.note.correction` box rather than silently contradicting them - the user is still being examined on that deck.

## Open questions for the user
- Is there a graded assessment / exam attached to this course? Changes how much we drill deck-specific phrasing.
- Do they have a dataset or project in mind? A real project would sharpen the mission a lot.

## Verification log (session 1)

Every interactive was tested headlessly in Node before shipping. Three findings
changed what the lessons say:

1. **Lesson 1's core claim is exact, not approximate.** With `activation='none'`, the
   trained network's output has max curvature ~1e-16 (floating-point zero) at N = 1, 8
   and 24 hidden units, and the final loss is identical to 5 decimal places (0.33967 /
   0.33973 / 0.33967). The collapse is not "roughly a line" - it is a line.

2. **Lesson 3 originally claimed SGD and Momentum both diverge around lr 0.17. Wrong.**
   Measured: SGD diverges at 0.17, Momentum survives to 0.32 (theory agrees:
   `lr < 2(1+beta)/k` = 3.8/12 = 0.317). Adam and RMSProp never diverge in range.
   Fixed by extending the slider to 0.40 and rewriting the passage as a three-stage
   ordering, which is a better demonstration anyway.
   Also confirmed: Adam's first step is *exactly* +/- the learning rate on both axes
   despite gradients of -2.55 and +12.6. Tightened "almost exactly" to "exactly".

3. **Dropout does not help on the Lesson 4 dataset, and the lesson now says so.**
   Sweep over 6 runs, final validation loss: none 0.0804, dropout 0.05 -> 0.0753,
   0.10 -> 0.0787, 0.15 -> 0.0875, 0.20 -> 0.1319, 0.30 -> 0.1094. L2 sweep: 0.002 is
   the optimum at 0.0565 vs 0.0818 unregularized.
   14 data points is simply too small for dropout to pay off - it swings the model
   straight from overfitting to underfitting. Rather than fake it, the demo keeps
   dropout at 0.15 and the lesson teaches the real result: regularizers have
   preconditions and are not interchangeable. Constants set to L2 = 0.002,
   dropout = 0.15, patience = 60.

**Standing rule from this: do not ship an interactive whose behaviour has not been
measured.** Two of the four lessons made a claim that the code did not support.

## Verification log (session 2)

**Link check, 2026-09-14.** The learner reported the 3Blue1Brown URL was dead. Checked all 25
external URLs across every `.md` and `.html` in the workspace with curl. Three genuine breaks:

- `www.3blue1brown.com/lessons/neural-networks` and `.../topics/neural-networks` - connection
  times out entirely (the apex `3blue1brown.com` 301s to `www.`, which then hangs). Not a wrong
  path; the host is unreachable. Replaced with the YouTube originals, video IDs confirmed by
  page title.
- `jmlr.org/papers/v15/srivastava14a/srivastava14a.pdf` (the dropout paper) - 404. The `v15/`
  short path serves the abstract page only; the PDF lives under `volume15/`. Switched to the
  abstract page `jmlr.org/papers/v15/srivastava14a.html`, which links the PDF itself.
- `stats.stackexchange.com` returns 403 to curl even with a browser user-agent. This is
  Cloudflare refusing datacenter IPs, not a broken link - it loads fine in a real browser.
  Left as is. **Any future link check will show this same 403; do not "fix" it.**

**Standing rule, extended:** do not ship an interactive whose behaviour has not been measured -
and do not ship a citation whose URL has not been fetched. A dead primary source is worse than
no primary source: it costs the learner trust in every other claim on the page.
