# Mission: Artificial Neural Networks

> **Status: provisional.** Written by the teacher from session-1 context, not yet
> confirmed by the learner. Correct anything that is wrong - a bad mission steers
> every future lesson in the wrong direction.

## Why

Working through the Samsung Innovation Campus AI Course module on Artificial Neural
Networks. The goal is to stop *recognising* the slides and start *making the calls*:
to sit in front of a blank Keras script with a real dataset and choose the architecture,
activation, loss, optimizer and regularization deliberately - and to know what to change
when the training curve misbehaves.

## Success looks like

- Given a task (predict a house price / flag a fraudulent transaction / sort an image into 10 classes), name the correct **output activation + loss function** pairing without looking it up.
- Look at a `history.history` plot and say which of *underfitting, overfitting, learning rate too high, dead network* it shows - and which knob to turn.
- Explain, to another student, why a network without activation functions is useless no matter how many layers it has.
- Write a `model.fit()` call with the right callbacks attached and justify each one.
- Know why the course defaults to `Adam` and when that default is the wrong choice.

## Constraints

- **Fundamentals only.** Stated explicitly: "don't go deep, focus on the fundamentals."
- Intuition and working knowledge over mathematical derivation. Chain-rule-level calculus is fine as *narrative*; proofs are not the goal.
- Learning alongside a fixed course deck, so terminology should stay compatible with it - while flagging where the deck is wrong or dated.

## Out of scope (for now)

- CNNs, RNNs, transformers, attention - the deck covers ANNs only.
- Deriving backpropagation from scratch, or implementing a network in raw NumPy.
- Distributed training, deployment, serving, quantisation.
- Hyperparameter search frameworks (KerasTuner, Optuna).
