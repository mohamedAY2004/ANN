# The course deck mis-states the Universal Approximation Theorem

Slide 39 of the Samsung deck reads: *"Zhou Lu and later on Boris Hanin proved
mathematically that Neural Networks can approximate any **convex** continuous function."*

The word **convex** does not belong there and materially weakens the claim. The theorem
says a sufficiently large network can approximate *any* continuous function on a compact
set to arbitrary accuracy - convexity is not required, and if it were, neural networks
would be near-useless (almost no interesting real function is convex). Lu et al. (2017)
and Hanin & Sellke (2018) proved the *arbitrary-depth, bounded-width* variant; the
original arbitrary-width results are Cybenko (1989) and Hornik (1991), with Leshno et al.
(1993) establishing that the activation just has to be non-polynomial.

**Implications.** (1) Correct this explicitly in Lesson 1 - it is load-bearing for the
whole "why do we need non-linearity" argument. (2) Treat the deck as a reliable guide to
*what will be examined* but not as a citable source of fact; verify its claims against
`RESOURCES.md` before building a lesson on them. (3) Also worth teaching the theorem's
real limitation, which the deck omits: it is an **existence** result. It promises a good
network exists; it promises nothing about gradient descent finding it. That distinction
is what motivates Lessons 3 and 4.
