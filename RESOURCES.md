# Artificial Neural Networks - Resources

Curated, high-trust sources. Lesson claims should be traceable to something here.

## Knowledge

### Core intuition

- [3Blue1Brown: Neural Networks (video series)](https://www.3blue1brown.com/topics/neural-networks)
  Four short films building from "what is a neuron" to backpropagation calculus. The single best
  visual explanation of gradient descent and backprop that exists. Use for: the geometric feel of
  weights, layers, cost surfaces and the chain rule.
- [Michael Nielsen, _Neural Networks and Deep Learning_](http://neuralnetworksanddeeplearning.com/)
  Free online book. Chapter 2 is the canonical, readable derivation of backpropagation - the deck's
  four-step BP1-BP4 equations come straight from here. Chapter 4 is a visual proof of universal
  approximation. Use for: anything where the deck's notation feels unexplained.
- [Stanford CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
  Course notes. Best concise treatment of activation functions, including the dying-ReLU problem.
  Use for: activation function trade-offs, layer sizing.
- [Stanford CS231n: Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/)
  Data preprocessing, weight initialisation, and regularization (L2, L1, dropout, max-norm).
  Use for: why we scale inputs, why dropout works, when to use which penalty.

### Reference / theory

- [Universal Approximation Theorem (Wikipedia)](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
  Careful statement of both the arbitrary-width (Cybenko 1989, Hornik 1991, Leshno 1993) and
  arbitrary-depth (Lu 2017, Hanin & Sellke 2018, Kidger & Lyons 2020) versions. Use for: correcting
  the deck's mis-statement of the theorem. See `learning-records/0002`.
- [Sebastian Ruder, _An Overview of Gradient Descent Optimization Algorithms_](https://www.ruder.io/optimizing-gradient-descent/)
  The definitive plain-English tour of SGD, momentum, Nesterov, Adagrad, RMSProp, Adam.
  Also on arXiv as [1609.04747](https://arxiv.org/abs/1609.04747). Use for: why each optimizer exists.
- [Kingma & Ba, _Adam: A Method for Stochastic Optimization_ (2015)](https://arxiv.org/abs/1412.6980)
  The paper the deck cites on slide 128. Section 1 and Algorithm 1 are readable in ten minutes.
- [Srivastava et al., _Dropout_ (JMLR 2014)](https://jmlr.org/papers/v15/srivastava14a/srivastava14a.pdf)
  The original dropout paper. Figure 1 and Section 2 give the "training an ensemble of thinned
  networks" intuition in two pages.

### API documentation (always check against these, not memory)

- [Keras 3: Callbacks API](https://keras.io/api/callbacks/) - EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard.
- [Keras 3: Optimizers API](https://keras.io/api/optimizers/) - signatures and defaults. Adam default lr is `0.001`.
- [Keras 3: Layer weight regularizers](https://keras.io/api/layers/regularizers/) - `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`.
- [Keras 3: Layer activations](https://keras.io/api/layers/activations/) - formulas for relu, sigmoid, softmax, tanh, elu, gelu, leaky_relu.
- [TensorFlow: Overfit and underfit tutorial](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
  Runnable walkthrough of exactly the regularization story in Lesson 4.

## Wisdom (communities)

- [r/learnmachinelearning](https://reddit.com/r/learnmachinelearning)
  Beginner-tolerant and well moderated. Use for: "is my training curve normal?", architecture sanity checks.
- [Cross Validated (stats.stackexchange.com)](https://stats.stackexchange.com/)
  High rigour, low tolerance for vagueness. Use for: conceptual questions where you want a
  citation-backed answer rather than an opinion.
- [Kaggle](https://www.kaggle.com/)
  Where the skills get tested against reality. The Getting Started competitions (Titanic, House
  Prices) are the natural first place to apply Lessons 1-4 end to end.

## Gaps

- No source yet for *practical debugging heuristics* (what a specific bad loss curve means).
  Candidate: Karpathy's "A Recipe for Training Neural Networks" - needs review before adding.
- Nothing yet on the coding half of the deck (TensorBoard walkthrough, the regression/classification
  exercises). Add when we reach those.
- Community preference not yet stated by the learner - do not keep pushing communities if declined.
