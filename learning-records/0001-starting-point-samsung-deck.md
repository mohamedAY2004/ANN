# Starting point: working from the Samsung Innovation Campus ANN deck

The learner arrived with the full 165-slide Samsung Innovation Campus "Understanding
Artificial Neural Networks" deck and asked for the intuition behind ANNs, activation
functions, optimizers, and the TensorFlow/Keras feature set (callbacks, regularization) -
explicitly at a **fundamentals** level, not depth.

This sets the floor: they have had *exposure* to perceptrons, layer terminology, sigmoid /
tanh / ReLU / softmax, quadratic and cross-entropy cost, gradient descent, Adam, and the
backpropagation equations. Exposure is not understanding, so nothing is assumed learned -
but lessons should not spend time introducing vocabulary the deck already introduced.
They should spend their time on the *why*, which the deck almost entirely omits.

**Implications.** The deck never explains why a non-linearity is necessary at all (it
introduces activations purely as "a way to bound the output"), never explains why more
than one optimizer exists, and stops before regularization and callbacks entirely. Those
gaps are exactly where the teaching value is, so they became Lessons 1-4.
