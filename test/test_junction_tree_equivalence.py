"""The optimized greedy_order / hypothetical_model_size must match the originals.

Both are checked against verbatim copies of the original implementations on
many random graphs, including trees (which take a separate code path in
``make_junction_tree``), isolated attributes, singleton cliques and the
stochastic variant of ``greedy_order`` with a fixed seed.
"""

import os
import sys
import unittest

import numpy as np

from mbi import Domain, junction_tree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference_greedy_order import reference_greedy_order  # noqa: E402
from _reference_junction_tree import reference_hypothetical_model_size  # noqa: E402


def _random_case(rng, trial):
  d = int(rng.integers(1, 13))
  domain = Domain(
      [f"v{i}" for i in range(d)], [int(x) for x in rng.integers(1, 7, d)]
  )
  cliques = [
      tuple(rng.choice(domain.attributes,
                       size=int(rng.integers(1, min(4, d) + 1)), replace=False))
      for _ in range(int(rng.integers(0, 2 * d + 1)))
  ]
  if trial % 5 == 0 and d > 1:
    perm = list(rng.permutation(domain.attributes))
    cliques = [(perm[i], perm[int(rng.integers(0, i))]) for i in range(1, d)]
  return domain, cliques


class TestGreedyOrder(unittest.TestCase):

  def test_matches_reference(self):
    rng = np.random.default_rng(0)
    for trial in range(400):
      domain, cliques = _random_case(rng, trial)
      elim = list(rng.choice(domain.attributes,
                             size=int(rng.integers(0, len(domain) + 1)),
                             replace=False))
      for e in (None, elim):
        for stochastic in (False, True):
          np.random.seed(trial)
          expected = reference_greedy_order(
              domain, cliques, stochastic, None if e is None else list(e)
          )
          np.random.seed(trial)
          actual = junction_tree.greedy_order(
              domain, cliques, stochastic, None if e is None else list(e)
          )
          self.assertEqual(expected, actual, (domain, cliques, e, stochastic))

  def test_does_not_mutate_elim(self):
    domain = Domain(list("abc"), [2, 3, 4])
    elim = ["a", "b"]
    junction_tree.greedy_order(domain, [("a", "b"), ("b", "c")], elim=elim)
    self.assertEqual(elim, ["a", "b"])


class TestHypotheticalModelSize(unittest.TestCase):

  def test_matches_reference(self):
    rng = np.random.default_rng(1)
    for trial in range(800):
      domain, cliques = _random_case(rng, trial)
      self.assertEqual(
          junction_tree.hypothetical_model_size(domain, cliques),
          reference_hypothetical_model_size(domain, cliques),
          (domain, cliques),
      )

  def test_matches_make_junction_tree(self):
    domain = Domain(list("abcdef"), [2, 3, 4, 5, 3, 2])
    cliques = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("e",)]
    jtree, _ = junction_tree.make_junction_tree(domain, cliques)
    cells = sum(domain.size(c) for c in junction_tree.maximal_cliques(jtree))
    self.assertEqual(
        junction_tree.hypothetical_model_size(domain, cliques),
        cells * 8 / 2**20,
    )


if __name__ == "__main__":
  unittest.main()
