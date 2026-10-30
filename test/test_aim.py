"""Tests that the optimized AIM mechanism behaves exactly like the original.

``mechanisms/aim.py`` was optimized to avoid JAX recompilation (NumPy
candidate scoring, merged measurements, unfused/fast/parallel compilation,
periodic cache clearing). ``_reference_aim.py`` keeps the original
implementation; these tests check each building block for equivalence and
that a full run selects the same cliques and fits the same model.
"""

import itertools
import os
import sys
import unittest

import jax
import numpy as np
from parameterized import parameterized

from mbi import (
    CliqueVector,
    Dataset,
    Domain,
    Factor,
    LinearMeasurement,
    MarkovRandomField,
    estimation,
    marginal_loss,
    marginal_oracles,
)
from mbi.marginal_loss import SlicedQuery

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "mechanisms"))

import aim  # noqa: E402
import _reference_aim  # noqa: E402


def _dataset(rows=2000, seed=0):
  """Small correlated dataset."""
  rng = np.random.default_rng(seed)
  domain = Domain(["a", "b", "c", "d"], [3, 4, 2, 5])
  a = rng.integers(0, 3, rows)
  b = (a + rng.integers(0, 2, rows)) % 4
  c = (b > 1).astype(int) ^ (rng.random(rows) < 0.1)
  d = (a + 2 * c + rng.integers(0, 2, rows)) % 5
  return Dataset({"a": a, "b": b, "c": c, "d": d}, domain)


def _noisy(data, cl, sigma, rng):
  x = np.asarray(data.project(cl).datavector())
  return LinearMeasurement(x + rng.normal(0, sigma, x.size), cl, stddev=sigma)


class TestMergeMeasurements(unittest.TestCase):

  def setUp(self):
    self.data = _dataset()
    rng = np.random.default_rng(1)
    self.measurements = [
        _noisy(self.data, ("a",), 5.0, rng),
        _noisy(self.data, ("a", "b"), 4.0, rng),
        _noisy(self.data, ("b", "c"), 6.0, rng),
        _noisy(self.data, ("a", "b"), 2.0, rng),  # repeat, different sigma
        _noisy(self.data, ("c", "d"), 3.0, rng),
        _noisy(self.data, ("a", "b"), 8.0, rng),  # second repeat
        _noisy(self.data, ("a",), 5.0, rng),
    ]

  def test_one_measurement_per_clique_in_first_seen_order(self):
    merged = aim.merge_measurements(self.measurements)
    self.assertEqual(
        [m.clique for m in merged],
        [("a",), ("a", "b"), ("b", "c"), ("c", "d")],
    )
    # Singletons are passed through untouched.
    self.assertIs(merged[2], self.measurements[2])

  def test_precision_weighted_mean(self):
    group = [m for m in self.measurements if m.clique == ("a", "b")]
    merged = [m for m in aim.merge_measurements(group)][0]
    w = np.array([1 / m.stddev**2 for m in group])
    ys = np.stack([np.asarray(m.noisy_measurement) for m in group])
    np.testing.assert_allclose(
        merged.noisy_measurement, (w[:, None] * ys).sum(0) / w.sum()
    )
    self.assertAlmostEqual(float(merged.stddev), 1 / np.sqrt(w.sum()))

  def test_total_and_lipschitz_unchanged(self):
    merged = aim.merge_measurements(self.measurements)
    self.assertAlmostEqual(
        estimation.minimum_variance_unbiased_total(self.measurements),
        estimation.minimum_variance_unbiased_total(merged),
        places=8,
    )
    domain = self.data.domain
    self.assertAlmostEqual(
        marginal_loss.from_linear_measurements(
            self.measurements, domain
        ).lipschitz,
        marginal_loss.from_linear_measurements(merged, domain).lipschitz,
        places=8,
    )

  def test_gradient_unchanged(self):
    domain = self.data.domain
    merged = aim.merge_measurements(self.measurements)
    loss_full = marginal_loss.from_linear_measurements(self.measurements, domain)
    loss_merged = marginal_loss.from_linear_measurements(merged, domain)
    self.assertEqual(loss_full.cliques, loss_merged.cliques)
    np.random.seed(3)
    mu = marginal_oracles.message_passing_stable(
        CliqueVector.random(domain, loss_full.cliques), 2000.0
    )
    g1 = jax.grad(loss_full)(mu)
    g2 = jax.grad(loss_merged)(mu)
    for cl in loss_full.cliques:
      np.testing.assert_allclose(
          np.asarray(g1[cl].values), np.asarray(g2[cl].values),
          rtol=1e-4, atol=1e-6,
      )

  def _estimate_both(self, estimator, iters):
    domain = self.data.domain
    m1 = estimator.estimate(domain, self.measurements, iters=iters, tol=None)
    m2 = estimator.estimate(
        domain, aim.merge_measurements(self.measurements), iters=iters,
        tol=None,
    )
    return m1, m2

  def test_estimate_unchanged_float64(self):
    prev = bool(jax.config.jax_enable_x64)  # pylint: disable=no-member
    jax.config.update("jax_enable_x64", True)
    try:
      # With a fixed step size the iterates depend only on the gradient, which
      # merging leaves unchanged, so the two runs agree to rounding error.
      m1, m2 = self._estimate_both(estimation.MirrorDescent(stepsize=1e-4), 300)
      for cl in m1.cliques:
        np.testing.assert_allclose(
            np.asarray(m1.marginals[cl].values),
            np.asarray(m2.marginals[cl].values),
            rtol=1e-9, atol=1e-9,
        )
      # The Armijo line search compares loss values, which differ by a
      # constant after merging, so rounding can flip near-tied accept/reject
      # decisions; both runs still converge to the same solution.
      m1, m2 = self._estimate_both(estimation.MirrorDescent(), 5000)
      for cl in m1.cliques:
        np.testing.assert_allclose(
            np.asarray(m1.marginals[cl].values),
            np.asarray(m2.marginals[cl].values),
            rtol=0, atol=1e-3,
        )
    finally:
      jax.config.update("jax_enable_x64", prev)
      jax.clear_caches()

  def test_non_identity_queries_passed_through(self):
    rng = np.random.default_rng(4)
    x = np.asarray(self.data.project(("a", "b")).datavector())
    sliced = [
        LinearMeasurement(x[1:] + rng.normal(0, 1, x.size - 1), ("a", "b"),
                          query=SlicedQuery(1))
        for _ in range(2)
    ]
    merged = aim.merge_measurements(sliced + self.measurements[:2])
    self.assertIs(merged[0], sliced[0])
    self.assertIs(merged[1], sliced[1])
    self.assertEqual(len(merged), 4)


class TestNumpyMarginals(unittest.TestCase):

  def test_matches_model_project(self):
    domain = Domain(list("abcde"), [3, 4, 2, 5, 3])
    cliques = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    np.random.seed(5)
    potentials = CliqueVector.random(domain, cliques)
    marginals = marginal_oracles.message_passing_stable(potentials, 500.0)
    model = MarkovRandomField(
        potentials=potentials, marginals=marginals, total=500.0
    )
    numpy_marginals = aim._NumpyMarginals(model)
    queries = [(a,) for a in domain.attributes] + list(
        itertools.permutations(domain.attributes, 2)
    ) + [("e", "a", "c")]
    n_out_of_model = 0
    for cl in queries:
      n_out_of_model += not model.marginals.supports(cl)
      np.testing.assert_allclose(
          numpy_marginals.datavector(cl),
          np.asarray(model.project(cl).datavector()),
          rtol=1e-4, atol=1e-3, err_msg=str(cl),
      )
    self.assertGreater(n_out_of_model, 0)


class TestFilterCandidates(unittest.TestCase):

  def test_matches_reference_with_and_without_cache(self):
    data = _dataset()
    domain = data.domain
    candidates = aim.compile_workload(
        [(cl, 1.0) for cl in itertools.combinations(domain, 2)]
        + [(("a", "b", "c"), 1.0)]
    )
    rng = np.random.default_rng(6)
    cache = {}
    for model_cliques in ([("a",), ("b",)], [("a", "b"), ("c",)],
                          [("a", "b"), ("b", "c"), ("d",)]):
      meas = [_noisy(data, cl, 5.0, rng) for cl in model_cliques]
      model = estimation.MirrorDescent().estimate(domain, meas, iters=50)
      for limit in (1e-5, 1e-4, 1.0):
        expected = _reference_aim.filter_candidates(candidates, model, limit)
        self.assertEqual(
            aim.filter_candidates(candidates, model, limit), expected
        )
        self.assertEqual(
            aim.filter_candidates(candidates, model, limit, cache), expected
        )
        # Second call is served from the cache and still identical.
        self.assertEqual(
            aim.filter_candidates(candidates, model, limit, cache), expected
        )


def _run(cls, data, workload, seed=0, **kwargs):
  selected = []

  class Recording(cls):

    def worst_approximated(self, *args, **kw):
      cl = super().worst_approximated(*args, **kw)
      selected.append(cl)
      return cl

  np.random.seed(seed)
  mech = Recording(
      1.0, 1e-9, prng=np.random.RandomState(seed), max_iters=200, **kwargs
  )
  model, synth = mech.run(data, workload)
  return selected, model, synth


class TestAIMEndToEnd(unittest.TestCase):
  """The optimized AIM must make the same choices as the original."""

  @classmethod
  def setUpClass(cls):
    cls.data = _dataset(rows=3000, seed=7)
    cls.workload = [
        (cl, 1.0) for cl in itertools.combinations(cls.data.domain, 2)
    ]
    cls.reference = _run(_reference_aim.ReferenceAIM, cls.data, cls.workload)

  def _check(self, result):
    ref_selected, ref_model, ref_synth = self.reference
    selected, model, synth = result
    self.assertEqual(selected, ref_selected)
    self.assertEqual(set(model.cliques), set(ref_model.cliques))
    self.assertAlmostEqual(
        float(model.total), float(ref_model.total), delta=1e-3
    )
    total = float(ref_model.total)
    for cl, _ in self.workload:
      np.testing.assert_allclose(
          np.asarray(model.project(cl).datavector()) / total,
          np.asarray(ref_model.project(cl).datavector()) / total,
          atol=1e-3, err_msg=str(cl),
      )
    self.assertEqual(synth.records, ref_synth.records)
    self.assertEqual(synth.domain, ref_synth.domain)

  @parameterized.expand([
      ("defaults", {}),
      ("serial_compile", dict(parallel_compile=False)),
      ("parallel_compile", dict(parallel_compile=True)),
      ("fused_no_fast_compile", dict(fuse_steps=True, fast_compile_max_cells=0)),
      ("clear_caches_every_round", dict(clear_caches_every=1)),
      ("never_clear_caches", dict(clear_caches_every=None)),
      ("intermediate_iters_equal_max_iters", dict(intermediate_iters=200)),
  ])
  def test_matches_reference(self, name, kwargs):
    del name
    self._check(_run(aim.AIM, self.data, self.workload, **kwargs))


class TestAIMIntermediateIters(unittest.TestCase):

  def test_fewer_intermediate_iters_runs(self):
    data = _dataset(rows=1000, seed=8)
    workload = [(cl, 1.0) for cl in itertools.combinations(data.domain, 2)]
    selected, model, synth = _run(
        aim.AIM, data, workload, intermediate_iters=50
    )
    self.assertGreater(len(selected), 0)
    self.assertEqual(synth.records, int(model.total))
    self.assertEqual(synth.domain, data.domain)


if __name__ == "__main__":
  unittest.main()
