# monopoly_ml
Monte Carlo Monopoly simulator with 5 player archetypes. Builds a 40×40 Markov chain transition matrix to compute stationary landing probabilities, uses those as feature weights for expected rental income, then trains RF/GBT/LR classifiers on mid-game snapshots to predict the winner. Accuracy scales from ~38% at turn 25 to ~82% at turn 200.
