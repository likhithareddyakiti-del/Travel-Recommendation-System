import itertools
import random
import unittest

from src.knapsack import knapsack_select


class TestKnapsack(unittest.TestCase):
    def test_classic_textbook_example(self):
        # weights 1,3,4,5  values 1,4,5,7  capacity 7  -> take items 1 and 2 (3+4) = 9
        r = knapsack_select([1, 3, 4, 5], [1, 4, 5, 7], budget=7, unit=1)
        self.assertEqual(r["selected_indices"], [1, 2])
        self.assertEqual(r["total_value"], 9)

    def test_rupee_example_with_units(self):
        # 1500+2200=3700 <= 4000 -> value 1.3 beats 3100 alone (0.9)
        r = knapsack_select([1500, 2200, 3100], [0.6, 0.7, 0.9], budget=4000, unit=100)
        self.assertEqual(r["selected_indices"], [0, 1])
        self.assertAlmostEqual(r["total_value"], 1.3)
        self.assertEqual(r["remaining_budget"], 300)

    def test_nothing_affordable(self):
        r = knapsack_select([5000, 6000], [0.9, 0.8], budget=1000)
        self.assertEqual(r["selected_indices"], [])
        self.assertEqual(r["total_cost"], 0)
        self.assertEqual(r["remaining_budget"], 1000)

    def test_empty_input(self):
        self.assertEqual(knapsack_select([], [], 5000)["selected_indices"], [])

    def test_never_exceeds_budget_when_unit_rounding(self):
        r = knapsack_select([1050, 1050], [1, 1], budget=2100, unit=100)  # 11+11 units > 21
        self.assertEqual(len(r["selected_indices"]), 1)
        self.assertLessEqual(r["total_cost"], 2100)

    def test_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(50):
            n = rng.randint(1, 8)
            w = [rng.randint(1, 10) for _ in range(n)]
            v = [rng.randint(1, 20) for _ in range(n)]
            cap = rng.randint(0, 25)
            best = max(sum(v[i] for i in s)
                       for k in range(n + 1) for s in itertools.combinations(range(n), k)
                       if sum(w[i] for i in s) <= cap)
            self.assertEqual(knapsack_select(w, v, cap, unit=1)["total_value"], best)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            knapsack_select([1, 2], [1], 10)
        with self.assertRaises(ValueError):
            knapsack_select([-5], [1], 10)


if __name__ == "__main__":
    unittest.main()
