import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------------------------------------------------
# Markov Chain Analysis
# -----------------------------------------------------------------------
# Monopoly board movement can be modeled as a Markov chain with 40 states.
# We construct the transition matrix from dice roll distributions, then
# layer in special rules: Go to Jail, Chance/Community Chest cards,
# and the three-doubles-to-jail rule.
#
# References:
#   Ash & Bishop (1972) - original probability work on Monopoly
#   Bewersdorff (2005) - "Luck, Logic, and White Lies"
# -----------------------------------------------------------------------

NUM_SPACES = 40
JAIL_POSITION = 10
GO_TO_JAIL = 30

# Chance card positions and their "advance to" destinations
CHANCE_SQUARES = [7, 22, 36]
COMMUNITY_CHEST_SQUARES = [2, 17, 33]

# Chance card "advance to" targets (simplified to most common moves)
# ~1/16 chance each for Go, Jail, nearest RR, nearest utility, etc.
CHANCE_ADVANCE_TARGETS = {
    7:  {"go": 0, "jail": 10, "illinois": 24, "stcharles": 11, "nearest_rr": 15, "nearest_util": 12, "reading_rr": 5,  "boardwalk": 39},
    22: {"go": 0, "jail": 10, "illinois": 24, "stcharles": 11, "nearest_rr": 25, "nearest_util": 28, "reading_rr": 5,  "boardwalk": 39},
    36: {"go": 0, "jail": 10, "illinois": 24, "stcharles": 11, "nearest_rr": 5,  "nearest_util": 12, "reading_rr": 5,  "boardwalk": 39},
}


def build_dice_distribution():
    """
    Returns a dict {roll_total: probability} for two fair d6.
    """
    counts = defaultdict(int)
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            counts[d1 + d2] += 1
    total = 36
    return {roll: count / total for roll, count in counts.items()}


def build_transition_matrix():
    """
    Constructs the 40x40 Markov transition matrix for Monopoly board movement.

    Assumptions:
    - Three-doubles rule sends player directly to jail (we approximate this
      by adding ~1/216 extra probability mass flowing into jail from every square).
    - Chance cards: 8 of 16 cards move the player; the rest do nothing positional.
      We weight those 8 targets equally as 1/16 each, ~0.5 probability of moving.
    - Community Chest: 2 of 16 cards move the player (Go or Jail).
    - Go to Jail square (30) always redirects to square 10.
    """
    T = np.zeros((NUM_SPACES, NUM_SPACES))
    dice = build_dice_distribution()

    # Probability of rolling doubles on one throw = 6/36
    p_doubles = 6 / 36
    # Probability of three consecutive doubles = (1/6)^3
    p_three_doubles = (1 / 6) ** 3

    for src in range(NUM_SPACES):
        for roll, prob in dice.items():
            raw_dest = (src + roll) % NUM_SPACES

            # Go to Jail square always sends to Jail
            if raw_dest == GO_TO_JAIL:
                T[src][JAIL_POSITION] += prob
                continue

            # Chance squares: ~8/16 cards have positional effect
            if raw_dest in CHANCE_SQUARES:
                targets = CHANCE_ADVANCE_TARGETS[raw_dest]
                p_move = 8 / 16  # half the deck moves you
                p_stay = 1 - p_move
                p_each_target = p_move / len(targets)

                T[src][raw_dest] += prob * p_stay
                for dest in targets.values():
                    if dest == GO_TO_JAIL:
                        T[src][JAIL_POSITION] += prob * p_each_target
                    else:
                        T[src][dest] += prob * p_each_target

            # Community Chest: 2/16 cards move you (Go or Go to Jail)
            elif raw_dest in COMMUNITY_CHEST_SQUARES:
                p_go = 1 / 16
                p_jail = 1 / 16
                p_stay = 1 - p_go - p_jail

                T[src][raw_dest] += prob * p_stay
                T[src][0] += prob * p_go            # Advance to Go
                T[src][JAIL_POSITION] += prob * p_jail

            else:
                T[src][raw_dest] += prob

        # Three-doubles rule: small probability mass redirects to jail from any square
        # This is an approximation; a fully accurate model would require extra state dimensions.
        T[src][JAIL_POSITION] += p_three_doubles
        # Renormalize the row so it sums to 1
        row_sum = T[src].sum()
        if row_sum > 0:
            T[src] /= row_sum

    return T


def compute_stationary_distribution(T, iterations=10000, tol=1e-10):
    """
    Computes the stationary distribution of the Markov chain by power iteration.

    Starting from a uniform distribution, repeatedly multiplies by the
    transition matrix until convergence.

    Args:
        T: (40, 40) row-stochastic transition matrix
        iterations: max iterations before giving up
        tol: L1 convergence threshold

    Returns:
        pi: (40,) stationary probability vector
    """
    # Start uniform
    pi = np.ones(NUM_SPACES) / NUM_SPACES

    for i in range(iterations):
        pi_next = pi @ T
        if np.sum(np.abs(pi_next - pi)) < tol:
            print(f"Stationary distribution converged after {i+1} iterations.")
            break
        pi = pi_next

    return pi_next


def get_markov_landing_probs():
    """
    Builds the transition matrix and returns the stationary distribution.
    This is the main entry point for getting landing probabilities.
    """
    T = build_transition_matrix()
    pi = compute_stationary_distribution(T)
    return pi


def print_top_landing_squares(pi, board_labels=None):
    """
    Prints the top 10 most-visited squares with their probabilities.
    """
    if board_labels is None:
        board_labels = [
            "GO", "Med Ave", "Community Chest", "Baltic Ave", "Income Tax",
            "Reading RR", "Oriental Ave", "Chance", "Vermont Ave", "Connecticut Ave",
            "Jail/Visiting", "St Charles Pl", "Electric Co", "States Ave", "Virginia Ave",
            "Penn RR", "St James Pl", "Community Chest", "Tennessee Ave", "New York Ave",
            "Free Parking", "Kentucky Ave", "Chance", "Indiana Ave", "Illinois Ave",
            "B&O RR", "Atlantic Ave", "Ventnor Ave", "Water Works", "Marvin Gardens",
            "Go to Jail", "Pacific Ave", "N Carolina Ave", "Community Chest", "Penn Ave",
            "Short Line", "Chance", "Park Place", "Luxury Tax", "Boardwalk"
        ]

    ranked = sorted(enumerate(pi), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Most-Landed Squares (Markov Stationary Distribution):")
    print(f"{'Rank':<6} {'Square':<25} {'Position':<10} {'Probability':<12}")
    print("-" * 55)
    for rank, (pos, prob) in enumerate(ranked[:10], start=1):
        label = board_labels[pos] if pos < len(board_labels) else f"Space {pos}"
        print(f"{rank:<6} {label:<25} {pos:<10} {prob:.4f}")


# -----------------------------------------------------------------------
# Strategy Profiles
# -----------------------------------------------------------------------

class Strategy:
    """
    Defines a player's behavioral profile along six axes. All parameters
    are normalized to [0, 1] or a cash integer where noted.
    """

    def __init__(self, name, purchase_aggressiveness, reserve_cash_target,
                 build_aggressiveness, mortgage_threshold, jail_preference,
                 trade_aggressiveness):
        self.name = name
        self.purchase_aggressiveness = purchase_aggressiveness
        self.reserve_cash_target = reserve_cash_target
        self.build_aggressiveness = build_aggressiveness
        self.mortgage_threshold = mortgage_threshold
        self.jail_preference = jail_preference
        self.trade_aggressiveness = trade_aggressiveness

    @classmethod
    def generate_aggressive(cls):
        return cls("Aggressive", 0.90, 100, 0.85, 0.50, "early_exit", 0.80)

    @classmethod
    def generate_conservative(cls):
        return cls("Conservative", 0.45, 450, 0.30, 0.15, "late_stay", 0.35)

    @classmethod
    def generate_balanced(cls):
        return cls("Balanced", 0.65, 250, 0.55, 0.30, "late_stay", 0.55)

    @classmethod
    def generate_opportunist(cls):
        # Waits for the right trade then develops at full speed
        return cls("Opportunist", 0.30, 300, 0.90, 0.25, "early_exit", 0.95)

    @classmethod
    def generate_tax_collector(cls):
        # Tries to lock up the board; extreme mortgaging tolerance
        return cls("TaxCollector", 0.95, 0, 0.99, 0.80, "early_exit", 0.10)


# -----------------------------------------------------------------------
# Board and Property Definitions
# -----------------------------------------------------------------------

class Property:
    """
    Holds all static and mutable state for a single purchasable space.
    """

    def __init__(self, name, prop_type, color, price, rents, house_cost=50):
        self.name = name
        self.prop_type = prop_type
        self.color = color
        self.price = price
        self.rents = rents
        self.house_cost = house_cost
        self.owner = None
        self.houses = 0
        self.hotel = False
        self.mortgaged = False
        self.position = None

    def get_rent(self, dice_roll=7):
        if self.mortgaged:
            return 0

        if self.prop_type == "Property":
            if self.hotel:
                return self.rents[5]
            if self.houses > 0:
                return self.rents[self.houses]
            # Unimproved monopoly doubles base rent
            if self.owner:
                group = [p for p in self.owner.properties if p.color == self.color]
                size = 2 if self.color in ["Brown", "Dark Blue"] else 3
                if len(group) == size:
                    return self.rents[0] * 2
            return self.rents[0]

        elif self.prop_type == "Railroad":
            if not self.owner:
                return 0
            rr_count = len([p for p in self.owner.properties if p.prop_type == "Railroad"])
            return self.rents[min(rr_count - 1, 3)]

        elif self.prop_type == "Utility":
            if not self.owner:
                return 0
            util_count = len([p for p in self.owner.properties if p.prop_type == "Utility"])
            return (4 if util_count == 1 else 10) * dice_roll

        return 0


class Board:
    def __init__(self):
        self.properties_pool = self._build_properties()
        self.spaces = self._layout_board()

    def _build_properties(self):
        props = {
            "Med_Ave":      Property("Mediterranean Avenue",    "Property", "Brown",     60,  [2,   10,  30,  90,  160,  250], 50),
            "Baltic_Ave":   Property("Baltic Avenue",           "Property", "Brown",     80,  [4,   20,  60,  180, 320,  450], 50),
            "Ori_Ave":      Property("Oriental Avenue",         "Property", "Light Blue", 100, [6,   30,  90,  270, 400,  550], 50),
            "Ver_Ave":      Property("Vermont Avenue",          "Property", "Light Blue", 100, [6,   30,  90,  270, 400,  550], 50),
            "Con_Ave":      Property("Connecticut Avenue",      "Property", "Light Blue", 120, [8,   40,  100, 300, 450,  600], 50),
            "StC_Pl":       Property("St. Charles Place",       "Property", "Pink",      140, [10,  50,  150, 450, 625,  750], 100),
            "States_Ave":   Property("States Avenue",           "Property", "Pink",      140, [10,  50,  150, 450, 625,  750], 100),
            "Vir_Ave":      Property("Virginia Avenue",         "Property", "Pink",      160, [12,  60,  180, 500, 700,  900], 100),
            "StJ_Pl":       Property("St. James Place",         "Property", "Orange",    180, [14,  70,  200, 500, 750,  950], 100),
            "Tenn_Ave":     Property("Tennessee Avenue",        "Property", "Orange",    180, [14,  70,  200, 500, 750,  950], 100),
            "NY_Ave":       Property("New York Avenue",         "Property", "Orange",    200, [16,  80,  220, 600, 800,  1000], 100),
            "Ky_Ave":       Property("Kentucky Avenue",         "Property", "Red",       220, [18,  90,  250, 700, 875,  1050], 150),
            "Ind_Ave":      Property("Indiana Avenue",          "Property", "Red",       220, [18,  90,  250, 700, 875,  1050], 150),
            "Ill_Ave":      Property("Illinois Avenue",         "Property", "Red",       240, [20,  100, 300, 750, 925,  1100], 150),
            "Atl_Ave":      Property("Atlantic Avenue",         "Property", "Yellow",    260, [22,  110, 330, 800, 975,  1150], 150),
            "Vent_Ave":     Property("Ventnor Avenue",          "Property", "Yellow",    260, [22,  110, 330, 800, 975,  1150], 150),
            "Marvin_Gdn":   Property("Marvin Gardens",          "Property", "Yellow",    280, [24,  120, 360, 950, 1025, 1200], 150),
            "Pac_Ave":      Property("Pacific Avenue",          "Property", "Green",     300, [26,  130, 390, 900, 1100, 1275], 200),
            "NC_Ave":       Property("North Carolina Avenue",   "Property", "Green",     300, [26,  130, 390, 900, 1100, 1275], 200),
            "Penn_Ave":     Property("Pennsylvania Avenue",     "Property", "Green",     320, [28,  150, 450, 1000,1200, 1400], 200),
            "Park_Pl":      Property("Park Place",              "Property", "Dark Blue", 350, [35,  175, 500, 1100,1300, 1500], 200),
            "Boardwalk":    Property("Boardwalk",               "Property", "Dark Blue", 400, [50,  200, 600, 1400,1700, 2000], 200),
            "Read_RR":      Property("Reading Railroad",        "Railroad", None,        200, [25, 50, 100, 200]),
            "Penn_RR":      Property("Pennsylvania Railroad",   "Railroad", None,        200, [25, 50, 100, 200]),
            "BO_RR":        Property("B. & O. Railroad",        "Railroad", None,        200, [25, 50, 100, 200]),
            "Short_Line":   Property("Short Line",              "Railroad", None,        200, [25, 50, 100, 200]),
            "Elec_Co":      Property("Electric Company",        "Utility",  None,        150, []),
            "Water_Works":  Property("Water Works",             "Utility",  None,        150, []),
        }
        return props

    def _layout_board(self):
        p = self.properties_pool
        layout = [
            "GO",             p["Med_Ave"],    "Community Chest", p["Baltic_Ave"],  "Income Tax",
            p["Read_RR"],     p["Ori_Ave"],    "Chance",          p["Ver_Ave"],     p["Con_Ave"],
            "Jail",           p["StC_Pl"],     p["Elec_Co"],      p["States_Ave"],  p["Vir_Ave"],
            p["Penn_RR"],     p["StJ_Pl"],     "Community Chest", p["Tenn_Ave"],    p["NY_Ave"],
            "Free Parking",   p["Ky_Ave"],     "Chance",          p["Ind_Ave"],     p["Ill_Ave"],
            p["BO_RR"],       p["Atl_Ave"],    p["Vent_Ave"],     p["Water_Works"], p["Marvin_Gdn"],
            "Go to Jail",     p["Pac_Ave"],    p["NC_Ave"],       "Community Chest",p["Penn_Ave"],
            p["Short_Line"],  "Chance",        p["Park_Pl"],      "Luxury Tax",     p["Boardwalk"],
        ]
        for i, space in enumerate(layout):
            if isinstance(space, Property):
                space.position = i
        return layout


# -----------------------------------------------------------------------
# Trading Engine
# -----------------------------------------------------------------------

class TradingEngine:
    """
    Property-for-property swap logic with cash balancing.
    Only fires when a trade completes a monopoly for at least one side.
    """

    @staticmethod
    def negotiate_trades(game):
        active = [p for p in game.players if not p.bankrupt]
        if len(active) < 2:
            return
        for p1 in active:
            for p2 in active:
                if p1 is not p2:
                    TradingEngine._attempt_monopoly_swap(p1, p2, game)

    @staticmethod
    def _attempt_monopoly_swap(p1, p2, game):
        tradable1 = [p for p in p1.properties if p.houses == 0 and not p.hotel and not p.mortgaged]
        tradable2 = [p for p in p2.properties if p.houses == 0 and not p.hotel and not p.mortgaged]

        for prop1 in tradable1:
            for prop2 in tradable2:
                if prop1.color is None or prop2.color is None:
                    continue
                p1_wins = TradingEngine._completes_group(p1, prop2.color)
                p2_wins = TradingEngine._completes_group(p2, prop1.color)

                if not (p1_wins or p2_wins):
                    continue

                diff = prop1.price - prop2.price
                if diff > 0:
                    cash = int(diff * (1.0 + p2.strategy.trade_aggressiveness * 0.2))
                    if p2.cash - cash > p2.strategy.reserve_cash_target:
                        TradingEngine._swap(p1, p2, prop1, prop2, cash)
                        return
                else:
                    cash = int(abs(diff) * (1.0 + p1.strategy.trade_aggressiveness * 0.2))
                    if p1.cash - cash > p1.strategy.reserve_cash_target:
                        TradingEngine._swap(p2, p1, prop2, prop1, cash)
                        return

    @staticmethod
    def _completes_group(player, color):
        owned = len([p for p in player.properties if p.color == color])
        needed = 2 if color in ["Brown", "Dark Blue"] else 3
        return (owned + 1) == needed

    @staticmethod
    def _swap(recv, give, from_recv, from_give, cash):
        recv.properties.remove(from_recv)
        give.properties.remove(from_give)
        recv.properties.append(from_give);  from_give.owner = recv
        give.properties.append(from_recv);  from_recv.owner = give
        recv.cash += cash
        give.cash -= cash


# -----------------------------------------------------------------------
# Player
# -----------------------------------------------------------------------

class Player:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy
        self.cash = 1500
        self.properties = []
        self.position = 0
        self.in_jail = False
        self.turns_in_jail = 0
        self.get_out_of_jail_cards = 0
        self.bankrupt = False

    def get_net_worth(self):
        worth = self.cash
        for p in self.properties:
            worth += (p.price // 2) if p.mortgaged else p.price
            worth += p.houses * p.house_cost
            if p.hotel:
                worth += 5 * p.house_cost
        return worth

    def liquidate(self, target):
        """
        Attempts to raise `target` cash by selling houses then mortgaging.
        Returns True if successful.
        """
        # Sell houses/hotels first
        while self.cash < target:
            developed = [p for p in self.properties if p.houses > 0 or p.hotel]
            if not developed:
                break
            prop = max(developed, key=lambda p: 5 if p.hotel else p.houses)
            if prop.hotel:
                prop.hotel = False
                prop.houses = 4
            else:
                prop.houses -= 1
            self.cash += prop.house_cost // 2

        # Mortgage cheapest properties
        while self.cash < target:
            candidates = [p for p in self.properties if not p.mortgaged and p.houses == 0 and not p.hotel]
            if not candidates:
                break
            prop = min(candidates, key=lambda p: p.price)
            prop.mortgaged = True
            self.cash += prop.price // 2

        return self.cash >= target


# -----------------------------------------------------------------------
# Simulation Engine
# -----------------------------------------------------------------------

class MonopolyGame:
    """
    Runs a single game, captures snapshot feature vectors at defined turn
    milestones, and returns the winner along with all snapshots.
    """

    SNAPSHOT_TURNS = [25, 50, 100, 200]

    def __init__(self, strategies, landing_probs):
        self.board = Board()
        self.players = [Player(f"Agent_{i+1}", s) for i, s in enumerate(strategies)]
        self.landing_probs = landing_probs
        self.turn = 0
        self.snapshots = {}

    def run(self):
        max_turns = 350
        while self.turn <= max_turns:
            active = [p for p in self.players if not p.bankrupt]
            if len(active) <= 1:
                break

            player = active[self.turn % len(active)]
            self._take_turn(player)

            if self.turn in self.SNAPSHOT_TURNS:
                self.snapshots[self.turn] = self._feature_vector()

            if self.turn % 15 == 0:
                TradingEngine.negotiate_trades(self)

            self.turn += 1

        survivors = [p for p in self.players if not p.bankrupt]
        winner = survivors[0] if len(survivors) == 1 else max(self.players, key=lambda p: p.get_net_worth())

        for snap in self.snapshots.values():
            snap["Target_Winner"] = winner.name

        return winner.name, self.snapshots

    def _take_turn(self, player):
        if player.in_jail:
            if player.get_out_of_jail_cards > 0:
                player.get_out_of_jail_cards -= 1
                player.in_jail = False
            elif player.strategy.jail_preference == "early_exit" or self.turn < 50:
                if player.cash > 50:
                    player.cash -= 50
                    player.in_jail = False
            else:
                player.turns_in_jail += 1
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                if d1 == d2:
                    player.in_jail = False
                    player.turns_in_jail = 0
                elif player.turns_in_jail >= 3:
                    player.cash -= 50
                    player.in_jail = False
                    player.turns_in_jail = 0
                return

        roll = random.randint(1, 6) + random.randint(1, 6)
        old_pos = player.position
        player.position = (player.position + roll) % 40

        if player.position < old_pos:
            player.cash += 200  # passed GO

        space = self.board.spaces[player.position]

        if space == "Go to Jail":
            player.position = 10
            player.in_jail = True
            return

        if isinstance(space, str):
            if "Tax" in space:
                fee = 200 if "Income" in space else 75
                if player.cash < fee and not player.liquidate(fee):
                    player.bankrupt = True
                    return
                player.cash -= fee
            return

        if isinstance(space, Property):
            if space.owner is None:
                margin = player.cash - space.price
                if margin > player.strategy.reserve_cash_target and random.random() < player.strategy.purchase_aggressiveness:
                    player.cash -= space.price
                    player.properties.append(space)
                    space.owner = player
                else:
                    self._auction(space)
            elif space.owner is not player:
                rent = space.get_rent(roll)
                if player.cash < rent and not player.liquidate(rent):
                    player.bankrupt = True
                    for p in player.properties:
                        p.owner = None
                        p.houses = 0
                        p.hotel = False
                        p.mortgaged = False
                    return
                player.cash -= rent
                space.owner.cash += rent

        self._build_houses(player)

    def _auction(self, prop):
        best_bid = 0
        winner = None
        for p in self.players:
            if p.bankrupt:
                continue
            cap = int(prop.price * p.strategy.purchase_aggressiveness)
            bid = min(cap, p.cash - p.strategy.reserve_cash_target)
            if bid > best_bid:
                best_bid = bid
                winner = p
        if winner and best_bid > 0:
            winner.cash -= best_bid
            winner.properties.append(prop)
            prop.owner = winner

    def _build_houses(self, player):
        groups = defaultdict(list)
        for p in player.properties:
            if p.color and not p.mortgaged:
                groups[p.color].append(p)

        for color, props in groups.items():
            needed = 2 if color in ["Brown", "Dark Blue"] else 3
            if len(props) != needed:
                continue
            for prop in sorted(props, key=lambda x: x.houses):
                if prop.hotel:
                    continue
                if player.cash - prop.house_cost <= player.strategy.reserve_cash_target:
                    continue
                if random.random() < player.strategy.build_aggressiveness:
                    if prop.houses < 4:
                        prop.houses += 1
                        player.cash -= prop.house_cost
                    else:
                        prop.houses = 0
                        prop.hotel = True
                        player.cash -= prop.house_cost

    def _feature_vector(self):
        features = {}
        for i, agent in enumerate(self.players):
            pre = f"A{i+1}_"
            nw = agent.get_net_worth()

            features[pre + "cash"]                = agent.cash
            features[pre + "net_worth"]           = nw
            features[pre + "liquidity_ratio"]     = agent.cash / max(1.0, nw)
            features[pre + "property_value"]      = sum(p.price for p in agent.properties)
            features[pre + "house_value"]         = sum(p.houses * p.house_cost for p in agent.properties if not p.hotel)
            features[pre + "hotel_value"]         = sum(5 * p.house_cost for p in agent.properties if p.hotel)
            features[pre + "mortgaged_count"]     = sum(1 for p in agent.properties if p.mortgaged)
            features[pre + "railroads"]           = sum(1 for p in agent.properties if p.prop_type == "Railroad")
            features[pre + "utilities"]           = sum(1 for p in agent.properties if p.prop_type == "Utility")
            features[pre + "property_count"]      = len(agent.properties)
            features[pre + "avg_prop_value"]      = np.mean([p.price for p in agent.properties]) if agent.properties else 0
            features[pre + "total_houses"]        = sum(p.houses for p in agent.properties if not p.hotel)
            features[pre + "total_hotels"]        = sum(1 for p in agent.properties if p.hotel)
            features[pre + "pct_mortgaged"]       = features[pre + "mortgaged_count"] / max(1, features[pre + "property_count"])
            features[pre + "bankruptcy_margin"]   = nw - agent.strategy.reserve_cash_target

            # Expected rental income weighted by Markov landing probabilities
            expected_income = 0.0
            for p in agent.properties:
                if not p.mortgaged and p.position is not None:
                    expected_income += self.landing_probs[p.position] * p.get_rent()
            features[pre + "expected_income"] = expected_income

            for color in ["Brown", "Light Blue", "Pink", "Orange", "Red", "Yellow", "Green", "Dark Blue"]:
                owned = sum(1 for p in agent.properties if p.color == color)
                size = 2 if color in ["Brown", "Dark Blue"] else 3
                features[pre + f"mono_{color.replace(' ', '_')}"] = int(owned == size)

        return features


# -----------------------------------------------------------------------
# Dataset Builder
# -----------------------------------------------------------------------

def build_dataset(n_games=1200):
    print(f"Running {n_games} simulations...")

    # Compute landing probabilities once; reuse across all games
    landing_probs = get_markov_landing_probs()

    pool = [
        Strategy.generate_aggressive(),
        Strategy.generate_conservative(),
        Strategy.generate_balanced(),
        Strategy.generate_opportunist(),
    ]

    records = {t: [] for t in MonopolyGame.SNAPSHOT_TURNS}

    for i in range(n_games):
        if i % 200 == 0 and i > 0:
            print(f"  completed {i} games...")
        random.shuffle(pool)
        game = MonopolyGame(pool, landing_probs)
        _, snapshots = game.run()
        for t, snap in snapshots.items():
            records[t].append(snap)

    return {t: pd.DataFrame(rows) for t, rows in records.items()}, landing_probs


# -----------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Compute and display the Markov chain stationary distribution
    print("=" * 60)
    print("MARKOV CHAIN ANALYSIS")
    print("=" * 60)
    T = build_transition_matrix()
    pi = compute_stationary_distribution(T)
    print_top_landing_squares(pi)

    # 2. Run simulations
    print("\n" + "=" * 60)
    print("SIMULATION PIPELINE")
    print("=" * 60)
    datasets, landing_probs = build_dataset(n_games=1000)

    # 3. ML evaluation at turn 50
    target_turn = 50
    df = datasets[target_turn].dropna()
    print(f"\nTurn {target_turn} dataset: {len(df)} samples, {df.shape[1]} features")

    if not df.empty:
        X = df.drop(columns=["Target_Winner"])
        y = df["Target_Winner"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        models = {
            "Logistic Regression":    LogisticRegression(max_iter=1500, class_weight="balanced"),
            "Random Forest":          RandomForestClassifier(n_estimators=150, random_state=42),
            "Gradient Boosted Trees": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        print(f"\n--- Classifier Performance (Turn {target_turn} Snapshots) ---")
        for name, clf in models.items():
            Xtr = X_train_s if "Logistic" in name else X_train
            Xte = X_test_s  if "Logistic" in name else X_test
            clf.fit(Xtr, y_train)
            preds = clf.predict(Xte)
            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds, average="macro")
            print(f"  {name:<28} Acc: {acc:.4f}  Macro-F1: {f1:.4f}")

        # Feature importances from Random Forest
        rf = models["Random Forest"]
        top_idx = np.argsort(rf.feature_importances_)[::-1][:8]
        print("\n--- Top 8 Predictive Features (Random Forest, Turn 50) ---")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. {X.columns[idx]}  ({rf.feature_importances_[idx]*100:.2f}%)")

    # 4. Temporal horizon analysis
    print("\n--- Temporal Horizon: Prediction Accuracy by Turn ---")
    for t in [25, 50, 100, 200]:
        tdf = datasets[t].dropna()
        if tdf.empty:
            continue
        Xt = tdf.drop(columns=["Target_Winner"])
        yt = tdf["Target_Winner"]
        Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.25, random_state=42)
        rf_t = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_t.fit(Xtr, ytr)
        acc = accuracy_score(yte, rf_t.predict(Xte))
        print(f"  Turn {t:>3}  Accuracy: {acc*100:.2f}%")

    # 5. Strategy win-rate tournament (500 games)
    print("\n--- Strategy Win-Rate Tournament (500 games) ---")
    wins = {"Aggressive": 0, "Conservative": 0, "Balanced": 0, "TaxCollector": 0}
    tournament_pool = [
        Strategy.generate_aggressive(),
        Strategy.generate_conservative(),
        Strategy.generate_balanced(),
        Strategy.generate_tax_collector(),
    ]

    for _ in range(500):
        random.shuffle(tournament_pool)
        game = MonopolyGame(tournament_pool, landing_probs)
        winner_str, _ = game.run()
        idx = int(winner_str.split("_")[-1]) - 1
        wins[tournament_pool[idx].name] += 1

    for strat, w in wins.items():
        print(f"  {strat:<15}  Wins: {w:<4}  Win Rate: {w/500*100:.1f}%")

    # 6. Temporal accuracy plot
    turns_list = [25, 50, 100, 200]
    acc_list = []
    for t in turns_list:
        tdf = datasets[t].dropna()
        if tdf.empty:
            acc_list.append(0)
            continue
        Xt = tdf.drop(columns=["Target_Winner"])
        yt = tdf["Target_Winner"]
        Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.25, random_state=42)
        rf_t = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_t.fit(Xtr, ytr)
        acc_list.append(accuracy_score(yte, rf_t.predict(Xte)))

    plt.figure(figsize=(8, 4.5))
    plt.plot(turns_list, acc_list, marker="o", linewidth=2.5, color="#2b5c8f", label="Random Forest")
    plt.axhline(y=0.25, color="r", linestyle="--", label="Baseline (random guess)")
    plt.title("Temporal Horizon: Winner Predictability Over Game Time", fontsize=12, fontweight="bold")
    plt.xlabel("Game Turn Snapshot")
    plt.ylabel("Prediction Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("temporal_accuracy.png", dpi=150)
    plt.show()
    print("\nPlot saved to temporal_accuracy.png")
