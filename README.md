# Monopoly Strategy Simulator

> Monte Carlo game engine + Markov chain board analysis + ML winner prediction

Three components: Markov chain landing probabilities, full game simulation across parameterized player archetypes, and ML winner prediction from mid-game snapshots.

---

## File layout

```
monopoly_sim.py        main source file; run this directly
temporal_accuracy.png  output plot (generated on run)
README.md              this file
```

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

Runs in four phases: Markov analysis → 1000-game simulation → classifier training → 500-game strategy tournament. Total runtime ~2–4 minutes.

---

## Markov chain details

The 40×40 transition matrix `T[i][j]` handles dice roll distributions (triangular over 2–12), Go to Jail (square 30 → 10), Chance (squares 7, 22, 36; 8/16 cards move the player), Community Chest (squares 2, 17, 33; 2/16 cards move the player), and the three-doubles rule (approximated by adding p = (1/6)³ jail probability from every square).

Stationary distribution `π` via power iteration:
```
π_{t+1} = π_t @ T    until ||π_{t+1} - π_t||_1 < 1e-10
```
Converges in ~119 iterations.

> **Known approximation**: the model treats each turn as a single roll. Rolling doubles allows a re-roll, so a turn can cover up to 36 squares (e.g. double-3 + 7 = 13, unreachable in one standard roll). A fully accurate model would use a composite displacement PMF across single-roll, one-reroll, and two-reroll cases. The effect on high-level conclusions is small but non-trivial for squares 13–24 from common departure points.

---

## Strategy profiles

| Parameter | Description |
|---|---|
| `purchase_aggressiveness` | Probability of buying an unowned property [0, 1] |
| `reserve_cash_target` | Minimum cash buffer to maintain ($) |
| `build_aggressiveness` | Probability of building a house when able [0, 1] |
| `mortgage_threshold` | Max fraction of assets that can be mortgaged [0, 1] |
| `jail_preference` | `early_exit` pays to leave; `late_stay` rolls for doubles |
| `trade_aggressiveness` | Premium willing to pay/accept in trades [0, 1] |

**Aggressive** — buys everything, builds fast, exits jail immediately  
**Conservative** — selective buyer, large cash buffer, avoids debt  
**Balanced** — middle-ground across all axes  
**Opportunist** — trade-hungry; builds fast once a monopoly is secured  
**TaxCollector** — zero cash buffer, buys to deny opponents, refuses trades  

---

## Results

**Top 10 landing squares (Markov stationary distribution):**

| Rank | Square | Position | Probability |
|---|---|---|---|
| 1 | Jail / Just Visiting | 10 | 6.30% |
| 2 | Illinois Ave | 24 | 3.17% |
| 3 | GO | 0 | 3.07% |
| 4 | Tennessee Ave | 18 | 2.95% |
| 5 | New York Ave | 19 | 2.91% |
| 6 | B&O Railroad | 25 | 2.88% |
| 7 | Free Parking | 20 | 2.88% |
| 8 | Reading Railroad | 5 | 2.83% |
| 9 | Kentucky Ave | 21 | 2.82% |
| 10 | Water Works | 28 | 2.82% |

Jail lands at ~6.3%, nearly 2× the next square. Orange and Red properties sit 6–9 squares past Jail and are consistently the strongest holdings.

**Classifier accuracy (turn 50, baseline = 25%):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.44 | 0.44 |
| Random Forest | 0.52 | 0.51 |
| Gradient Boosted Trees | 0.45 | 0.44 |

Top features (Random Forest) are all `expected_income` and `bankruptcy_margin` terms — the Markov-weighted rental value is the strongest predictive signal at turn 50.

**Prediction accuracy by turn (Random Forest):**

| Turn | Accuracy |
|---|---|
| 25 | 42.4% |
| 50 | 50.8% |
| 100 | 67.2% |
| 200 | 87.7% |

**Strategy tournament (500 games):**

| Strategy | Wins | Win Rate |
|---|---|---|
| TaxCollector | 199 | 39.8% |
| Aggressive | 149 | 29.8% |
| Balanced | 118 | 23.6% |
| Opportunist | 20 | 4.0% |
| Conservative | 14 | 2.8% |

TaxCollector's dominance suggests property denial outweighs liquidity in 4-player games. Opportunist struggles without reliable trade partners. Conservative's near-elimination (~3%) confirms that hoarding cash is almost always losing.

---

## Limitations

- Three-doubles rule is approximated; full accuracy requires tracking consecutive doubles count in state and a composite per-turn displacement PMF
- Chance/Community Chest treated as sampling with replacement (no deck memory)
- Auction logic simplified; no sequential bidding
- Some edge cases in the building evenness rule are ignored
