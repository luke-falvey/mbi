"""Tests for the native (Rust) backend in ``mbi.native``.

Every native routine is checked against the JAX / Python implementation it
replaces. Comparisons that should be exact up to rounding run with x64
enabled so both sides compute in float64.
"""

import contextlib
import itertools
import os
import sys
import unittest
import unittest.mock

import jax
import numpy as np
from parameterized import parameterized

from mbi import (
    CliqueVector,
    Constraint,
    Domain,
    Factor,
    LinearMeasurement,
    MarkovRandomField,
    estimation,
    marginal_loss,
    marginal_oracles,
    native,
)
from mbi.marginal_loss import SlicedQuery

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference_junction_tree import reference_hypothetical_model_size  # noqa: E402


@contextlib.contextmanager
def _x64():
  prev = bool(jax.config.jax_enable_x64)  # pylint: disable=no-member
  jax.config.update("jax_enable_x64", True)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", prev)
    jax.clear_caches()


_MODELS = {
    "chain_with_loop": (
        Domain(list("abcdefg"), [2, 3, 4, 5, 3, 2, 3]),
        [("a", "b"), ("b", "c"), ("c", "d", "e"), ("e", "f"), ("a", "f"),
         ("g",)],
    ),
    "non_canonical_orders": (
        Domain(list("abcde"), [3, 2, 4, 2, 3]),
        [("c", "a"), ("e", "d", "b"), ("a", "e"), ("d",)],
    ),
    "disconnected": (
        Domain(list("wxyz"), [4, 3, 2, 5]),
        [("z", "w"), ("y",), ("x", "z")],
    ),
    "single_clique": (Domain(list("pq"), [3, 4]), [("q", "p")]),
}


def _measurements(domain, cliques, seed=0, total=1000.0):
  rng = np.random.default_rng(seed)
  p = Factor(domain, rng.random(domain.shape))
  p = total * p / p.sum()
  extra = [tuple(reversed(cliques[0]))] if len(cliques[0]) > 1 else []
  return [
      LinearMeasurement(
          np.asarray(p.project(cl).datavector())
          + rng.normal(0, 3.0, domain.size(cl)),
          cl,
          stddev=float(rng.uniform(1.0, 5.0)),
      )
      for cl in list(cliques) + extra
  ]


def _max_diff(cv1, cv2, cliques):
  return max(
      float(np.abs(np.asarray(cv1[c].values) - np.asarray(cv2[c].values)).max())
      for c in cliques
  )


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestOracle(unittest.TestCase):

  @parameterized.expand([(name,) for name in _MODELS])
  def test_marginals_match_hugin(self, name):
    domain, cliques = _MODELS[name]
    with _x64():
      np.random.seed(0)
      pot = CliqueVector.random(domain, cliques) * 3
      for total in (1.0, 777.0):
        expected = marginal_oracles.message_passing_hugin(pot, total)
        actual = native.oracle(pot, total)
        self.assertLess(_max_diff(expected, actual, cliques), 1e-10 * total)

  @parameterized.expand([(name,) for name in _MODELS])
  def test_beliefs_are_calibrated_marginals(self, name):
    domain, cliques = _MODELS[name]
    with _x64():
      np.random.seed(1)
      pot = CliqueVector.random(domain, cliques)
      beliefs, edges = native.junction_tree_beliefs(pot, 50.0)
      joint = sum(
          (pot[cl].expand(domain) for cl in cliques), Factor.zeros(domain)
      ).exp()
      joint = joint * (50.0 / float(joint.sum()))
      for cl, table in beliefs.items():
        np.testing.assert_allclose(
            table, np.asarray(joint.project(cl).datavector(flatten=False)),
            rtol=1e-9, atol=1e-12,
        )
      self.assertEqual(len(edges), max(len(beliefs) - 1, 0))


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestNativeMirrorDescent(unittest.TestCase):

  @parameterized.expand([
      (name, kw, tol)
      for name in _MODELS
      for kw in ({}, {"stepsize": 1e-4})
      for tol in (1e-3, None)
  ])
  def test_matches_mirror_descent_float64(self, name, kwargs, tol):
    domain, cliques = _MODELS[name]
    with _x64():
      meas = _measurements(domain, cliques)
      expected = estimation.MirrorDescent(**kwargs).estimate(
          domain, meas, iters=300, tol=tol
      )
      actual = native.NativeMirrorDescent(**kwargs).estimate(
          domain, meas, iters=300, tol=tol
      )
      self.assertEqual(expected.cliques, actual.cliques)
      self.assertAlmostEqual(float(expected.total), float(actual.total))
      # With a fixed step the iterates agree to rounding error. With the
      # Armijo line search, rounding can flip a near-tied accept/reject
      # decision, or put the gap check on the other side of `tol` (one more
      # block of steps), so the results agree to within optimizer precision.
      bound = 1e-9 if kwargs else 1e-5
      self.assertLess(
          _max_diff(expected.marginals, actual.marginals, expected.cliques)
          / float(expected.total),
          bound,
      )

  def test_warm_start_and_known_total(self):
    domain, cliques = _MODELS["chain_with_loop"]
    with _x64():
      meas = _measurements(domain, cliques, seed=3)
      first = estimation.MirrorDescent().estimate(domain, meas[:3], iters=100)
      kw = dict(iters=200, warm_start=first, known_total=1000.0)
      expected = estimation.MirrorDescent().estimate(domain, meas, **kw)
      actual = native.NativeMirrorDescent().estimate(domain, meas, **kw)
      self.assertLess(
          _max_diff(expected.marginals, actual.marginals, expected.cliques),
          1e-6,
      )
      self.assertEqual(float(actual.total), 1000.0)

  def test_marginal_loss_fn_input(self):
    domain, cliques = _MODELS["non_canonical_orders"]
    with _x64():
      meas = _measurements(domain, cliques, seed=4)
      loss_fn = marginal_loss.from_linear_measurements(meas, domain)
      expected = estimation.MirrorDescent().estimate(
          domain, loss_fn, known_total=900.0, iters=200
      )
      actual = native.NativeMirrorDescent().estimate(
          domain, loss_fn, known_total=900.0, iters=200
      )
      self.assertLess(
          _max_diff(expected.marginals, actual.marginals, expected.cliques),
          1e-6,
      )

  def test_unsupported_settings_fall_back_to_jax(self):
    domain, cliques = _MODELS["chain_with_loop"]
    meas = _measurements(domain, cliques, seed=5)
    est = native.NativeMirrorDescent()
    self.assertTrue(est.supports(meas))
    self.assertFalse(est.supports(meas, callback_fn=lambda mu: None))
    x = np.asarray(meas[0].noisy_measurement)
    sliced = [LinearMeasurement(x[1:], meas[0].clique, query=SlicedQuery(1))]
    self.assertFalse(est.supports(meas + sliced))
    constraint = Constraint(domain.project(("a", "b")), invalid=np.array([[0, 0]]))
    self.assertFalse(est.supports(meas, constraints=[constraint]))
    # Falls back and still returns a valid model.
    model = est.estimate(domain, meas, constraints=[constraint], iters=60)
    self.assertIsInstance(model, MarkovRandomField)
    calls = []
    est.estimate(domain, meas, iters=60, callback_fn=calls.append)
    self.assertGreater(len(calls), 0)


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestOutOfModelMarginals(unittest.TestCase):

  @parameterized.expand([(name,) for name in _MODELS])
  def test_matches_variable_elimination(self, name):
    domain, cliques = _MODELS[name]
    with _x64():
      np.random.seed(2)
      pot = CliqueVector.random(domain, cliques) * 2
      queries = list(itertools.permutations(domain.attributes, 2))
      queries += [(a,) for a in domain.attributes]
      if len(domain) >= 3:
        queries.append(tuple(domain.attributes[:3]))
      got = native.out_of_model_marginals(pot, 123.0, queries)
      for q in queries:
        expected = np.asarray(
            marginal_oracles.variable_elimination(pot, q, 123.0).values
        ).reshape(-1)
        np.testing.assert_allclose(got[q], expected, rtol=1e-9, atol=1e-10)


class TestExpand(unittest.TestCase):

  def test_matches_clique_vector_expand(self):
    domain = Domain(list("abcde"), [2, 3, 4, 2, 3])
    pot = CliqueVector.random(
        domain, [("a", "b"), ("c",), ("b", "a"), ("d", "e"), ("e",)]
    )
    new = [("b", "a", "c"), ("e", "d"), ("c", "d"), ("a",)]
    expected = pot.expand(new)
    actual = native.expand(pot, new)
    for cl in new:
      self.assertEqual(expected[cl].domain, actual[cl].domain)
      np.testing.assert_allclose(
          np.asarray(actual[cl].values), np.asarray(expected[cl].values),
          rtol=1e-6, atol=1e-6,
      )


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestHypotheticalModelSizes(unittest.TestCase):

  def test_matches_reference_on_random_graphs(self):
    rng = np.random.default_rng(0)
    for trial in range(300):
      d = int(rng.integers(1, 12))
      domain = Domain(
          [f"v{i}" for i in range(d)], [int(x) for x in rng.integers(1, 9, d)]
      )
      base = [
          tuple(rng.choice(domain.attributes,
                           size=int(rng.integers(1, min(4, d) + 1)),
                           replace=False))
          for _ in range(int(rng.integers(0, 2 * d + 1)))
      ]
      if trial % 5 == 0 and d > 2:  # trees take a different code path
        perm = list(rng.permutation(domain.attributes))
        base = [(perm[i], perm[int(rng.integers(0, i))]) for i in range(1, d - 1)]
      candidates = [
          tuple(rng.choice(domain.attributes,
                           size=int(rng.integers(1, min(3, d) + 1)),
                           replace=False))
          for _ in range(4)
      ]
      got = native.hypothetical_model_sizes(domain, base, candidates)
      for cl, size in zip(candidates, got):
        self.assertEqual(
            size, reference_hypothetical_model_size(domain, base + [cl])
        )

  def test_overflow_falls_back_to_python(self):
    domain = Domain([f"v{i}" for i in range(12)], [2**40] * 12)
    base = [(f"v{i}", f"v{i + 1}", f"v{i + 2}") for i in range(10)]
    got = native.hypothetical_model_sizes(domain, base, [("v0", "v11")])
    self.assertEqual(
        got[0], reference_hypothetical_model_size(domain, base + [("v0", "v11")])
    )


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestSyntheticDataNative(unittest.TestCase):

  def test_equivalent_to_default(self):
    domain = Domain([f"v{i}" for i in range(7)], [4, 3, 5, 2, 6, 3, 4])
    cliques = [("v0", "v1", "v2"), ("v2", "v3", "v4"), ("v4", "v5", "v6"),
               ("v0", "v6")]
    np.random.seed(1)
    pot = CliqueVector.random(domain, cliques) * 2
    model = MarkovRandomField(
        potentials=pot,
        marginals=marginal_oracles.message_passing_stable(pot, 1000.0),
        total=1000.0,
    )
    np.random.seed(5)
    default = model.synthetic_data(rows=20000)
    np.random.seed(5)
    fast = model.synthetic_data(rows=20000, use_native=True)
    self.assertEqual(default.domain, fast.domain)
    for cl in cliques:
      a = np.asarray(default.project(cl).datavector())
      b = np.asarray(fast.project(cl).datavector())
      self.assertLess(0.5 * np.abs(a - b).sum() / 20000, 1e-3)

  def test_constraints_use_default_path(self):
    domain = Domain(list("ab"), [2, 3])
    pot = CliqueVector.random(domain, [("a", "b")])
    constraint = Constraint(domain, invalid=np.array([[0, 0]]))
    model = MarkovRandomField(
        potentials=pot,
        marginals=marginal_oracles.message_passing_shafer_shenoy(
            pot, 100.0, constraints=(constraint,)
        ),
        total=100.0,
        constraints=(constraint,),
    )
    synth = model.synthetic_data(rows=500, use_native=True)
    ab = synth.project(("a", "b")).datavector(flatten=False)
    self.assertEqual(float(ab[0, 0]), 0.0)


class TestWithoutLibrary(unittest.TestCase):
  """Without the native library everything falls back to the JAX code."""

  def test_falls_back(self):
    domain, cliques = _MODELS["chain_with_loop"]
    meas = _measurements(domain, cliques, seed=6)
    with unittest.mock.patch.object(native, "_load", lambda: None):
      self.assertFalse(native.is_available())
      est = native.NativeMirrorDescent()
      self.assertFalse(est.supports(meas))
      actual = est.estimate(domain, meas, iters=100)
      expected = estimation.MirrorDescent().estimate(domain, meas, iters=100)
      self.assertLess(
          _max_diff(expected.marginals, actual.marginals, expected.cliques),
          1e-3,
      )
      with self.assertRaises(RuntimeError):
        native.oracle(CliqueVector.zeros(domain, cliques))


if __name__ == "__main__":
  unittest.main()
