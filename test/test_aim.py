"""Tests that the optimized AIM mechanism behaves exactly like the original.

``mechanisms/aim.py`` was optimized to avoid JAX recompilation (NumPy
candidate scoring, merged measurements, unfused/fast/parallel compilation,
periodic cache clearing). ``_reference_aim.py`` keeps the original
implementation; these tests check each building block for equivalence and
that a full run selects the same cliques and fits the same model.
"""

import dataclasses
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
    Dataset,
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


def _run(cls, data, workload, seed=0, forced=None, **kwargs):
  """Run an AIM implementation, recording each round's selection and scores.

  If ``forced`` (a list of cliques) is given, round ``t`` selects
  ``forced[t]`` instead of the sampled clique. The exponential mechanism is
  still evaluated, so the random stream (and hence all noise draws) stays
  aligned with an unforced run that made the same choices.
  """
  selected, scores, epsilons = [], [], []

  class Recording(cls):

    def exponential_mechanism(self, qualities, eps, *args, **kw):
      cl = super().exponential_mechanism(qualities, eps, *args, **kw)
      scores.append(dict(qualities))
      epsilons.append(float(eps))
      if forced is not None and len(selected) < len(forced):
        cl = forced[len(selected)]
      selected.append(cl)
      return cl

  np.random.seed(seed)
  mech = Recording(
      1.0, 1e-9, prng=np.random.RandomState(seed), max_iters=200, **kwargs
  )
  model, synth = mech.run(data, workload)
  return _Run(selected, scores, epsilons, model, synth)


@dataclasses.dataclass
class _Run:
  selected: list
  scores: list
  epsilons: list
  model: object
  synth: object


class _ReplayTest(unittest.TestCase):
  """Base: replays the reference AIM's selections through the optimized one.

  AIM is chaotic with respect to tiny numerical differences: a rounding-level
  change in one candidate score can flip a draw of the exponential mechanism,
  after which the runs diverge. So instead of requiring a free-running
  optimized AIM to make identical random choices, the optimized AIM is forced
  to select the reference's clique each round, and everything deterministic
  is compared: the candidate set and scores each round, the number of rounds
  (budget accounting / sigma halving), and the final model.

  Tolerances: AIM's intermediate fits stop early (gap tolerance 1e-3), so any
  change in the optimizer's path moves where they stop. As a baseline, the
  *original* AIM run on a mathematically identical problem (the same noisy
  measurements, passed to the estimator in reverse order) gives per-round
  scores that differ by up to ~1e-2 of the model total. The optimized AIM
  (which merges repeated measurements, an exactly equivalent objective) must
  stay within that same band.
  """

  x64 = False
  score_atol = 2e-2  # per unit of model total
  model_atol = 5e-3

  @classmethod
  def setUpClass(cls):
    cls.prev_x64 = bool(jax.config.jax_enable_x64)  # pylint: disable=no-member
    if cls.x64 != cls.prev_x64:
      jax.config.update("jax_enable_x64", cls.x64)
      jax.clear_caches()
    cls.data = _dataset(rows=3000, seed=7)
    cls.workload = [
        (cl, 1.0) for cl in itertools.combinations(cls.data.domain, 2)
    ]
    cls.reference = _run(_reference_aim.ReferenceAIM, cls.data, cls.workload)

  @classmethod
  def tearDownClass(cls):
    if cls.x64 != cls.prev_x64:
      jax.config.update("jax_enable_x64", cls.prev_x64)
      jax.clear_caches()

  def _check(self, **kwargs):
    ref = self.reference
    run = _run(
        aim.AIM, self.data, self.workload, forced=ref.selected, **kwargs
    )
    total = float(ref.model.total)
    # Each round, AIM halves its noise scale if the fitted model barely moved
    # (``||w - z||_1 <= sqrt(2/pi) sigma n``). When that statistic lands
    # within rounding of its threshold, implementations can legitimately
    # decide differently, after which the runs measure with different noise
    # scales and are no longer comparable. Rounds are compared up to the
    # first such split (visible as a different epsilon).
    split = next(
        (t for t, (a, b) in enumerate(zip(run.epsilons, ref.epsilons))
         if abs(a - b) > 1e-9 * abs(b)),
        None,
    )
    compared = len(ref.scores) if split is None else split
    self.assertGreaterEqual(compared, 1)
    for t in range(compared):
      got, want = run.scores[t], ref.scores[t]
      self.assertEqual(list(got), list(want), f"candidates, round {t}")
      for cl in want:
        self.assertAlmostEqual(
            got[cl], want[cl], delta=self.score_atol * total,
            msg=f"score of {cl}, round {t}",
        )
    self.assertEqual(run.synth.domain, ref.synth.domain)
    if split is not None:
      return
    self.assertEqual(run.synth.records, ref.synth.records)
    self.assertEqual(run.selected, ref.selected)  # same number of rounds
    self.assertEqual(set(run.model.cliques), set(ref.model.cliques))
    self.assertAlmostEqual(float(run.model.total), total, delta=1e-3 * total)
    for cl, _ in self.workload:
      np.testing.assert_allclose(
          np.asarray(run.model.project(cl).datavector()) / total,
          np.asarray(ref.model.project(cl).datavector()) / total,
          atol=self.model_atol, err_msg=str(cl),
      )


class TestAIMEndToEnd(_ReplayTest):
  """JAX code path, float32 like the reference."""

  @parameterized.expand([
      ("defaults", {}),
      ("serial_compile", dict(parallel_compile=False)),
      ("parallel_compile", dict(parallel_compile=True)),
      ("fused_no_fast_compile", dict(fuse_steps=True, fast_compile_max_cells=0)),
      ("clear_caches_every_round",
       dict(clear_caches_every=1, clear_caches_map_fraction=0.0)),
      ("never_clear_caches", dict(clear_caches_every=None)),
      ("intermediate_iters_equal_max_iters", dict(intermediate_iters=200)),
  ])
  def test_matches_reference(self, name, kwargs):
    del name
    self._check(use_native=False, **kwargs)


class TestAIMEndToEndFloat64(_ReplayTest):
  """Both code paths (JAX and native) in float64."""

  x64 = True

  def test_jax_path(self):
    self._check(use_native=False)

  @unittest.skipUnless(native.is_available(), "mbi native library not built")
  def test_native_path(self):
    self._check(use_native=True)

  @unittest.skipUnless(native.is_available(), "mbi native library not built")
  def test_native_path_intermediate_iters_equal_max_iters(self):
    self._check(use_native=True, intermediate_iters=200)


class TestMemoryMapUsage(unittest.TestCase):

  def test_usage_fraction(self):
    usage = aim._memory_map_usage()
    if not os.path.exists("/proc/self/maps"):
      self.assertIsNone(usage)
    else:
      self.assertGreater(usage, 0.0)
      self.assertLess(usage, 1.0)

  def test_clears_only_when_over_threshold(self):
    data = _dataset(rows=500, seed=9)
    workload = [(cl, 1.0) for cl in itertools.combinations(data.domain, 2)]
    calls = {"n": 0}
    orig = jax.clear_caches

    def counting():
      calls["n"] += 1
      orig()

    with unittest.mock.patch.object(aim.jax, "clear_caches", counting):
      _run(aim.AIM, data, workload, clear_caches_every=1,
           clear_caches_map_fraction=1.0)
      self.assertEqual(calls["n"], 0)
      if aim._memory_map_usage() is not None:
        _run(aim.AIM, data, workload, clear_caches_every=1,
             clear_caches_map_fraction=0.0)
        self.assertGreater(calls["n"], 0)


class TestAIMIntermediateIters(unittest.TestCase):

  def test_fewer_intermediate_iters_runs(self):
    data = _dataset(rows=1000, seed=8)
    workload = [(cl, 1.0) for cl in itertools.combinations(data.domain, 2)]
    run = _run(aim.AIM, data, workload, intermediate_iters=50)
    self.assertGreater(len(run.selected), 0)
    self.assertEqual(run.synth.records, int(run.model.total))
    self.assertEqual(run.synth.domain, data.domain)


if __name__ == "__main__":
  unittest.main()
