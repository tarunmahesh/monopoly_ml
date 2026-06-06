# Monopoly Strategy Simulator

> Monte Carlo game engine + Markov chain board analysis + ML winner prediction

A Monte Carlo simulation of Monopoly that models player behavior as parameterized strategy profiles, predicts game outcomes from mid-game snapshots using ML classifiers, and computes board landing probabilities via Markov chain analysis.

---

## What it does

The project has three main components:

**Markov chain board analysis** — constructs the full 40×40 transition matrix for a standard Monopoly board, accounting for dice roll distributions, the Go to Jail square, Chance and Community Chest card effects, and the three-doubles-to-jail rule. Power iteration is used to compute the stationary distribution, giving the long-run probability of landing on every square.

**Game simulation** — runs thousands of full games between four parameterized player archetypes (Aggressive, Conservative, Balanced, Opportunist, TaxCollector). Each game snapshots ~40 engineered features at turns 25, 50, 100, and 200.

**Winner prediction** — trains Logistic Regression, Random Forest, and Gradient Boosted Trees classifiers on the snapshot data to predict the eventual game winner from mid-game state.

---

## File layout

```
monopoly_sim.py        main source file; run this directly
temporal_accuracy.png  output plot (generated on run)
README.md              this file
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Running it

```bash
python monopoly_sim.py
```

The script runs in four phases:

1. Computes and prints the Markov stationary distribution (top 10 most-visited squares)
2. Runs 1000 simulated games and builds snapshot datasets
3. Trains classifiers on turn-50 snapshots and reports accuracy/F1
4. Runs a 500-game strategy tournament and reports win rates

Total runtime is roughly 2–4 minutes depending on hardware.

---

## Markov chain details

The transition matrix `T[i][j]` is the probability of moving from square `i` to square `j` in one turn. The construction handles:

- **Dice rolls**: two fair d6, giving the standard triangular distribution over 2–12
- **Go to Jail (square 30)**: all probability mass redirects to square 10
- **Chance (squares 7, 22, 36)**: 8 of 16 cards advance the player; each target weighted equally
- **Community Chest (squares 2, 17, 33)**: 2 of 16 cards move the player (Advance to Go, Go to Jail)
- **Three-doubles rule**: approximated by adding p = (1/6)³ extra jail probability from every square, then renormalizing

The stationary distribution `π` is found by power iteration:

```
π_{t+1} = π_t @ T    (repeated until ||π_{t+1} - π_t||_1 < 1e-10)
```

The resulting vector feeds directly into the feature engineering step, where each property's expected rental income is weighted by its landing probability.

---

## Strategy profiles

Each strategy is defined by six parameters:

| Parameter | Description |
|---|---|
| `purchase_aggressiveness` | Probability of buying an unowned property [0, 1] |
| `reserve_cash_target` | Minimum cash buffer to maintain ($) |
| `build_aggressiveness` | Probability of building a house when able [0, 1] |
| `mortgage_threshold` | Max fraction of assets that can be mortgaged [0, 1] |
| `jail_preference` | `early_exit` pays to leave; `late_stay` rolls for doubles |
| `trade_aggressiveness` | Premium willing to pay/accept in trades [0, 1] |

The included profiles are:

- **Aggressive** — buys everything, builds fast, exits jail immediately
- **Conservative** — selective buyer, large cash buffer, avoids debt
- **Balanced** — middle-ground across all axes
- **Opportunist** — low organic buying but extremely trade-hungry; builds at full speed once a monopoly is acquired
- **TaxCollector** — zero cash buffer, buys to deny opponents, refuses trades

---

## Feature engineering

Snapshot feature vectors include:

- Cash, net worth, liquidity ratio
- Property counts and total asset value
- House and hotel counts and values
- Number of mortgaged properties
- Railroad and utility ownership
- Binary monopoly completion flags per color group
- Expected rental income (property rent × Markov landing probability, summed across portfolio)
- Bankruptcy margin (net worth minus reserve target)

---

## Results (typical run)

**Markov top 5 landing squares:**

| Square | Probability |
|---|---|
| Jail / Just Visiting | ~4.8% |
| Illinois Avenue | ~3.2% |
| GO | ~3.1% |
| B&O Railroad | ~2.9% |
| New York Avenue | ~2.9% |

Jail is the single most-visited square by a wide margin, which is why the Orange and Red groups are among the strongest holdings in the game.

**Classifier accuracy (turn 50 snapshots):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | ~0.44 | ~0.43 |
| Random Forest | ~0.51 | ~0.50 |
| Gradient Boosted Trees | ~0.49 | ~0.48 |

Baseline (random guess among 4 players) = 25%.

**Temporal accuracy (Random Forest):**

| Turn | Accuracy |
|---|---|
| 25 | ~38% |
| 50 | ~51% |
| 100 | ~65% |
| 200 | ~82% |

**Strategy win rates (500-game tournament):**

| Strategy | Win Rate |
|---|---|
| Aggressive | ~34% |
| TaxCollector | ~27% |
| Balanced | ~22% |
| Opportunist | ~11% |
| Conservative | ~6% |

---

## Limitations and possible extensions

- The three-doubles rule is approximated; a fully accurate Markov model requires expanding state space to track consecutive doubles count
- Chance and Community Chest decks are treated as sampling with replacement (no deck memory)
- Auction logic is simplified; real Monopoly auctions involve sequential bidding
- The simulation ignores some edge cases in the building evenness rule
- Adding a proper card deck state machine would improve landing probability accuracy by a few percent on affected squares
