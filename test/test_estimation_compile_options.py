"""Tests for the compile-time options on Estimator.

``fuse_steps``, ``fast_compile`` and ``parallel_compile`` only change *how*
the optimization is compiled, never *what* it computes. These tests check
that every combination gives the same estimates as the default configuration,
that the jitted convergence check matches the original eager one, and that
each option compiles as few programs as intended.
"""

import contextlib
import itertools
import unittest
from unittest import mock

import jax
import numpy as np
from parameterized import parameterized

from mbi import Domain, Factor, estimation, marginal_loss

_DOMAIN = Domain(["a", "b", "c", "d", "e"], [2, 3, 4, 3, 2])
_CLIQUES = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a",)]


def _measurements(seed=0, total=1000.0, cliques=_CLIQUES):
  rng = np.random.default_rng(seed)
  p = Factor(_DOMAIN, rng.random(_DOMAIN.shape))
  p = total * p / p.sum()
  return [
      marginal_loss.LinearMeasurement(
          np.asarray(p.project(cl).datavector())
          + rng.normal(0, 3.0, _DOMAIN.size(cl)),
          cl,
          stddev=3.0,
      )
      for cl in cliques
  ]


def _max_marginal_diff(m1, m2):
  """Largest absolute marginal difference, relative to the model total."""
  return max(
      float(
          np.abs(
              np.asarray(m1.project(cl).datavector())
              - np.asarray(m2.project(cl).datavector())
          ).max()
      )
      for cl in _CLIQUES
  ) / float(m1.total)


# float32 rounding (plus early stopping on the gap) allows ~1e-5 relative
# differences; anything larger would indicate a real behavioural change.
_TOL = 1e-4


@contextlib.contextmanager
def _count_step_traces(cls):
  """Each (re)compile of a step program traces ``cls._step`` exactly once."""
  counter = {"n": 0}
  orig = cls._step

  def wrapper(self, *args, **kwargs):
    counter["n"] += 1
    return orig(self, *args, **kwargs)

  with mock.patch.object(cls, "_step", wrapper):
    yield counter


_OPTION_COMBOS = [
    dict(fuse_steps=f, fast_compile=fc, parallel_compile=pc)
    for f, fc, pc in itertools.product([True, False], repeat=3)
    if (f, fc, pc) != (True, False, False)
]


class TestOptionsGiveSameEstimates(unittest.TestCase):

  @parameterized.expand([(str(kw), kw) for kw in _OPTION_COMBOS])
  def test_mirror_descent(self, name, options):
    del name
    meas = _measurements()
    expected = estimation.MirrorDescent().estimate(_DOMAIN, meas, iters=300)
    actual = estimation.MirrorDescent(**options).estimate(
        _DOMAIN, meas, iters=300
    )
    # Only floating-point differences (XLA fusion / optimization level).
    self.assertLess(_max_marginal_diff(expected, actual), _TOL)
    self.assertAlmostEqual(float(expected.total), float(actual.total), 3)

  @parameterized.expand([
      ("mirror_descent_fixed_step", estimation.MirrorDescent, dict(stepsize=1e-4)),
      ("dual_averaging", estimation.DualAveraging, {}),
      ("interior_gradient", estimation.InteriorGradient, {}),
      ("universal_accelerated", estimation.UniversalAcceleratedMethod, {}),
  ])
  def test_other_estimators_unfused(self, name, cls, kwargs):
    del name
    meas = _measurements(seed=1)
    expected = cls(**kwargs).estimate(_DOMAIN, meas, iters=150, tol=None)
    actual = cls(fuse_steps=False, **kwargs).estimate(
        _DOMAIN, meas, iters=150, tol=None
    )
    self.assertLess(_max_marginal_diff(expected, actual), _TOL)

  def test_warm_start_and_callback(self):
    meas = _measurements(seed=2)
    first = estimation.MirrorDescent().estimate(_DOMAIN, meas[:3], iters=100)
    calls = {"default": 0, "unfused": 0}

    def cb(key):
      def f(_):
        calls[key] += 1
      return f

    expected = estimation.MirrorDescent().estimate(
        _DOMAIN, meas, iters=200, warm_start=first, callback_fn=cb("default"),
        tol=None,
    )
    actual = estimation.MirrorDescent(
        fuse_steps=False, parallel_compile=True
    ).estimate(
        _DOMAIN, meas, iters=200, warm_start=first, callback_fn=cb("unfused"),
        tol=None,
    )
    self.assertLess(_max_marginal_diff(expected, actual), _TOL)
    # Callbacks still fire once per CALLBACK_EVERY steps.
    self.assertEqual(calls["default"], calls["unfused"])
    self.assertEqual(calls["default"], -(-200 // estimation.CALLBACK_EVERY))


class TestOptionsFloat64(unittest.TestCase):
  """In float64 the options must agree to near machine precision."""

  def test_all_combos(self):
    prev = bool(jax.config.jax_enable_x64)  # pylint: disable=no-member
    jax.config.update("jax_enable_x64", True)
    try:
      meas = _measurements(seed=9)
      expected = estimation.MirrorDescent().estimate(
          _DOMAIN, meas, iters=200, tol=None
      )
      for options in _OPTION_COMBOS:
        actual = estimation.MirrorDescent(**options).estimate(
            _DOMAIN, meas, iters=200, tol=None
        )
        self.assertLess(
            _max_marginal_diff(expected, actual), 1e-9, str(options)
        )
    finally:
      jax.config.update("jax_enable_x64", prev)
      jax.clear_caches()


class TestJittedPrimalDualGap(unittest.TestCase):

  def test_matches_eager(self):
    meas = _measurements(seed=3)
    loss_fn = marginal_loss.from_linear_measurements(meas, _DOMAIN)
    model = estimation.MirrorDescent().estimate(_DOMAIN, meas, iters=50)
    mu = model.marginals.contract(loss_fn.cliques)
    eager = float(estimation.primal_dual_gap(loss_fn, mu))
    jitted = float(estimation._jitted_primal_dual_gap(loss_fn, mu, None, ()))
    self.assertAlmostEqual(eager, jitted, places=4)


class TestCompileCounts(unittest.TestCase):

  def setUp(self):
    jax.clear_caches()

  def test_default_uses_original_multi_step(self):
    est = estimation.MirrorDescent()
    self.assertIs(est._step_program(), estimation.MirrorDescent._multi_step)

  def test_unfused_compiles_once_per_estimate(self):
    meas = _measurements(seed=4)
    est = estimation.MirrorDescent(fuse_steps=False)
    with _count_step_traces(estimation.MirrorDescent) as counter:
      est.estimate(
          _DOMAIN, meas, iters=3 * estimation.CALLBACK_EVERY, tol=None
      )
    self.assertEqual(counter["n"], 1)

  def test_unfused_reuses_program_for_same_structure(self):
    est = estimation.MirrorDescent(fuse_steps=False)
    est.estimate(_DOMAIN, _measurements(seed=5), iters=60)
    with _count_step_traces(estimation.MirrorDescent) as counter:
      # New values, same structure: no recompile.
      est.estimate(_DOMAIN, _measurements(seed=6), iters=60)
    self.assertEqual(counter["n"], 0)

  @parameterized.expand([("fused", True), ("unfused", False)])
  def test_precompile_reused(self, name, fuse):
    del name
    meas = _measurements(seed=7)
    est = estimation.MirrorDescent(fuse_steps=fuse, fast_compile=True)
    with _count_step_traces(estimation.MirrorDescent) as counter:
      est.precompile(_DOMAIN, meas).result()
      after = counter["n"]
      self.assertGreaterEqual(after, 1)
      est.estimate(_DOMAIN, meas, iters=60)
      self.assertEqual(counter["n"], after)

  @parameterized.expand([("fused", True), ("unfused", False)])
  def test_parallel_compile_adds_no_compiles(self, name, fuse):
    del name
    meas = _measurements(seed=8)
    backend_compiles = {"n": 0}

    def listener(event, duration, **kwargs):
      del duration, kwargs
      if event == "/jax/core/compile/backend_compile_duration":
        backend_compiles["n"] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
      counts = {}
      for parallel in (False, True):
        jax.clear_caches()
        before = backend_compiles["n"]
        estimation.MirrorDescent(
            fuse_steps=fuse, parallel_compile=parallel
        ).estimate(_DOMAIN, meas, iters=60)
        counts[parallel] = backend_compiles["n"] - before
    finally:
      jax.monitoring.unregister_event_duration_listener(listener)
    self.assertEqual(counts[True], counts[False])


if __name__ == "__main__":
  unittest.main()
