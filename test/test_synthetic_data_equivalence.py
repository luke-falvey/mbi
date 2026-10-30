"""The optimized MarkovRandomField.synthetic_data must match the original.

``synthetic_data`` replaced ``np.unique(..., axis=0)`` + argsort with an
integer-key grouping (``_group_rows``). The random-number call sequence is
unchanged, so for the same seed the output must be bit-for-bit identical to
the original implementation, which is kept in ``_reference_synthetic_data``.
"""

import itertools
import os
import sys
import unittest

import numpy as np
from parameterized import parameterized

from mbi import CliqueVector, Domain, MarkovRandomField, marginal_oracles
from mbi import markov_random_field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference_synthetic_data import reference_synthetic_data  # noqa: E402


def _model(domain, cliques, total, seed):
  np.random.seed(seed)
  potentials = CliqueVector.random(domain, cliques) * 2.0
  marginals = marginal_oracles.message_passing_stable(potentials, total)
  return MarkovRandomField(
      potentials=potentials, marginals=marginals, total=total
  )


_CASES = [
    # name, domain, cliques
    ("chain", Domain(list("abcd"), [2, 3, 4, 5]), [("a", "b"), ("b", "c"), ("c", "d")]),
    (
        "triangle_and_tail",
        Domain(list("abcdef"), [3, 4, 2, 5, 6, 2]),
        [("a", "b", "c"), ("c", "d"), ("d", "e", "f"), ("a", "f")],
    ),
    (
        "independent",
        Domain(list("xyz"), [7, 1, 3]),
        [("x",), ("y",), ("z",)],
    ),
    (
        "wide_parents",
        Domain([f"v{i}" for i in range(7)], [4, 3, 5, 2, 6, 3, 4]),
        [("v0", "v1", "v2"), ("v2", "v3", "v4"), ("v4", "v5", "v6"), ("v0", "v6")],
    ),
]


class TestGroupRows(unittest.TestCase):

  @parameterized.expand([(1,), (2,), (3,)])
  def test_matches_np_unique(self, ncols):
    rng = np.random.default_rng(ncols)
    sizes = tuple(int(s) for s in rng.integers(1, 9, size=ncols))
    rows = np.stack(
        [rng.integers(0, s, size=5000).astype(np.uint8) for s in sizes], -1
    )
    uniques, inverse, counts, perm = markov_random_field._group_rows(
        rows, sizes
    )
    eu, ei, ec = np.unique(
        rows, axis=0, return_inverse=True, return_counts=True
    )
    np.testing.assert_array_equal(uniques, eu)
    np.testing.assert_array_equal(inverse, ei.reshape(-1))
    np.testing.assert_array_equal(counts, ec)
    np.testing.assert_array_equal(perm, np.argsort(ei.reshape(-1), kind="stable"))

  def test_sparse_fallback_matches_np_unique(self):
    # Force the np.unique(1-D) branch used for huge parent domains.
    rng = np.random.default_rng(0)
    sizes = (5, 7, 3)
    rows = np.stack([rng.integers(0, s, size=3000) for s in sizes], -1)
    old = markov_random_field._DENSE_GROUPING_LIMIT
    markov_random_field._DENSE_GROUPING_LIMIT = 1
    try:
      got = markov_random_field._group_rows(rows, sizes)
    finally:
      markov_random_field._DENSE_GROUPING_LIMIT = old
    eu, ei, ec = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(got[0], eu)
    np.testing.assert_array_equal(got[1], ei.reshape(-1))
    np.testing.assert_array_equal(got[2], ec)
    np.testing.assert_array_equal(got[3], np.argsort(ei.reshape(-1), kind="stable"))


class TestSyntheticDataMatchesReference(unittest.TestCase):

  @parameterized.expand(
      [
          (name, dom, cl, rows, method)
          for (name, dom, cl), rows, method in itertools.product(
              _CASES, (1, 37, 5000), ("round", "sample")
          )
      ]
  )
  def test_bitwise_identical(self, name, domain, cliques, rows, method):
    del name
    model = _model(domain, cliques, total=1000.0, seed=1)
    np.random.seed(123)
    expected = reference_synthetic_data(model, rows=rows, method=method)
    np.random.seed(123)
    actual = model.synthetic_data(rows=rows, method=method)
    self.assertEqual(expected.domain, actual.domain)
    for attr in domain.attributes:
      np.testing.assert_array_equal(
          expected.data[attr], actual.data[attr], err_msg=attr
      )
      self.assertEqual(expected.data[attr].dtype, actual.data[attr].dtype)

  def test_default_rows_uses_model_total(self):
    _, domain, cliques = _CASES[1]
    model = _model(domain, cliques, total=321.0, seed=2)
    np.random.seed(7)
    expected = reference_synthetic_data(model)
    np.random.seed(7)
    actual = model.synthetic_data()
    for attr in domain.attributes:
      np.testing.assert_array_equal(expected.data[attr], actual.data[attr])


if __name__ == "__main__":
  unittest.main()
