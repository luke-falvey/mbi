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


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestMarginalCounts(unittest.TestCase):

  def _data(self, n, weights=None):
    rng = np.random.default_rng(0)
    domain = Domain(list("abcde"), [2, 7, 3, 16, 5])
    data = {a: rng.integers(0, domain[a], n) for a in domain.attributes}
    from mbi import Dataset  # pylint: disable=import-outside-toplevel
    return Dataset(data, domain, weights=weights), domain

  @parameterized.expand([(10,), (5000,), (2_500_000,)])  # last: several parts
  def test_counts_match_project(self, n):
    data, domain = self._data(n)
    cliques = [("a",), ("d", "b"), ("e", "a", "c"), tuple(domain.attributes)]
    got = native.marginal_counts(data, cliques)
    for cl in cliques:
      expected = np.bincount(
          np.ravel_multi_index(tuple(data.data[a] for a in cl),
                               domain.project(cl).shape),
          minlength=domain.size(cl),
      )
      np.testing.assert_array_equal(got[cl], expected)

  def test_weighted(self):
    w = np.random.default_rng(1).random(20_000)
    data, domain = self._data(20_000, weights=w)
    cl = ("b", "e")
    expected = np.bincount(
        np.ravel_multi_index((data.data["b"], data.data["e"]), (7, 5)),
        weights=w, minlength=35,
    )
    np.testing.assert_allclose(native.marginal_counts(data, [cl])[cl], expected,
                               rtol=1e-12)


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestNativeAcceleratedMethod(unittest.TestCase):

  def test_x_iterate_matches_jax_float64(self):
    domain, cliques = _MODELS["chain_with_loop"]
    with _x64():
      meas = _measurements(domain, cliques, seed=7)
      loss_fn = marginal_loss.from_linear_measurements(meas, domain)
      total = estimation.minimum_variance_unbiased_total(meas)
      seen = []
      estimation.UniversalAcceleratedMethod().estimate(
          domain, loss_fn, total, iters=100, tol=None, callback_fn=seen.append
      )
      jax_x = seen[-1]  # total * x after 100 steps
      est = native.NativeAcceleratedMethod()
      cl = list(loss_fn.cliques)
      theta = np.zeros(sum(domain.size(c) for c in cl))
      _, mu, x, _ = est._run_native(
          domain, meas, loss_fn, total, cl, theta, 100, None
      )
      x_cv = native._unflatten(x, domain, cl)
      self.assertLess(_max_diff(jax_x, x_cv, cl) / total, 1e-6)
      # The returned model's marginals are exactly its potentials' marginals.
      model = est.estimate(domain, meas, iters=100, tol=None)
      recomputed = native.oracle(model.potentials, float(model.total))
      self.assertLess(_max_diff(model.marginals, recomputed, cl), 1e-9 * total)

  def test_accelerated_iterate_fits_better_than_mirror_descent(self):
    domain, cliques = _MODELS["chain_with_loop"]
    with _x64():
      meas = _measurements(domain, cliques, seed=8, total=1e6)
      loss_fn = marginal_loss.from_linear_measurements(meas, domain)
      total = estimation.minimum_variance_unbiased_total(meas)
      md = native.NativeMirrorDescent().estimate(domain, meas, iters=300, tol=None)
      md_loss = float(loss_fn(md.marginals.contract(loss_fn.cliques)))
      cl = list(loss_fn.cliques)
      _, _, _, info = native.NativeAcceleratedMethod()._run_native(
          domain, meas, loss_fn, total, cl,
          np.zeros(sum(domain.size(c) for c in cl)), 300, None,
      )
      self.assertLess(info[3], md_loss)  # info[3] = loss at total * x

  def test_falls_back(self):
    domain, cliques = _MODELS["chain_with_loop"]
    meas = _measurements(domain, cliques, seed=9)
    est = native.NativeAcceleratedMethod()
    self.assertFalse(est.supports(meas, callback_fn=lambda mu: None))
    calls = []
    model = est.estimate(domain, meas, iters=60, callback_fn=calls.append)
    self.assertIsInstance(model, MarkovRandomField)
    self.assertGreater(len(calls), 0)


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestSampleColumn(unittest.TestCase):

  def _setup(self, n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    sizes = (3, 4)
    parents = [rng.integers(0, s, n) for s in sizes]
    probs = rng.random(sizes + (5,))
    probs /= probs.sum(-1, keepdims=True)
    return parents, sizes, probs

  def test_round_matches_conditional_counts(self):
    parents, sizes, probs = self._setup()
    out = native.sample_column(parents, sizes, probs.cumsum(-1), "round", seed=1)
    key = parents[0] * sizes[1] + parents[1]
    for g in range(sizes[0] * sizes[1]):
      rows = key == g
      counts = np.bincount(out[rows], minlength=5)
      expected = rows.sum() * probs.reshape(-1, 5)[g]
      # Systematic sampling: each category within ~1 of its expected count.
      self.assertLessEqual(np.abs(counts - expected).max(), 2.0)

  def test_sample_method_is_unbiased(self):
    parents, sizes, probs = self._setup(n=400_000)
    out = native.sample_column(parents, sizes, probs.cumsum(-1), "sample", seed=2)
    key = parents[0] * sizes[1] + parents[1]
    rows = key == 5
    freq = np.bincount(out[rows], minlength=5) / rows.sum()
    np.testing.assert_allclose(freq, probs.reshape(-1, 5)[5], atol=0.02)

  def test_deterministic_given_seed(self):
    parents, sizes, probs = self._setup()
    a = native.sample_column(parents, sizes, probs.cumsum(-1), seed=3)
    b = native.sample_column(parents, sizes, probs.cumsum(-1), seed=3)
    c = native.sample_column(parents, sizes, probs.cumsum(-1), seed=4)
    np.testing.assert_array_equal(a, b)
    self.assertFalse(np.array_equal(a, c))

  def test_synthetic_data_reproducible_with_np_seed(self):
    domain = Domain(list("abcd"), [3, 4, 2, 5])
    pot = CliqueVector.random(domain, [("a", "b"), ("b", "c"), ("c", "d")])
    model = MarkovRandomField(
        potentials=pot,
        marginals=marginal_oracles.message_passing_stable(pot, 1000.0),
        total=1000.0,
    )
    np.random.seed(11)
    s1 = model.synthetic_data(rows=5000, use_native=True)
    np.random.seed(11)
    s2 = model.synthetic_data(rows=5000, use_native=True)
    for a in domain.attributes:
      np.testing.assert_array_equal(s1.data[a], s2.data[a])
      self.assertEqual(s1.data[a].dtype, np.min_scalar_type(domain[a]))


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestLossRtol(unittest.TestCase):

  def test_native_stops_at_same_block_as_jax(self):
    domain, cliques = _MODELS["chain_with_loop"]
    with _x64():
      meas = _measurements(domain, cliques, seed=12)
      for r in (1e-1, 1e-3):
        kw = dict(stepsize=1e-4, loss_rtol=r)
        expected = estimation.MirrorDescent(**kw).estimate(
            domain, meas, iters=1000, tol=None
        )
        actual = native.NativeMirrorDescent(**kw).estimate(
            domain, meas, iters=1000, tol=None
        )
        # Fixed step: identical iterates, so agreement means both stopped
        # after the same block.
        self.assertLess(
            _max_diff(expected.marginals, actual.marginals, expected.cliques),
            1e-9 * float(expected.total), r,
        )

  def test_jax_estimator_stops_early(self):
    domain, cliques = _MODELS["chain_with_loop"]
    meas = _measurements(domain, cliques, seed=13)
    calls = []
    estimation.MirrorDescent(loss_rtol=10.0).estimate(
        domain, meas, iters=1000, tol=None, callback_fn=calls.append
    )
    # The first block can't be compared with anything; the second always
    # satisfies a loose threshold.
    self.assertEqual(len(calls), 2)


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestMessagesFromProbabilities(unittest.TestCase):

  def test_extreme_ranges_fall_back_exactly(self):
    # A separator slice ~2000 below the clique's max: its probabilities
    # underflow, so the message must take the exact log-space path.
    domain = Domain(list("abc"), [3, 4, 2])
    cliques = [("a", "b"), ("b", "c")]
    with _x64():
      np.random.seed(14)
      pot = CliqueVector.random(domain, cliques)
      ab = np.asarray(pot[("a", "b")].values).copy()
      ab[:, 0] -= 2000.0
      pot[("a", "b")] = Factor(domain.project(("a", "b")), jax.numpy.asarray(ab))
      bc = np.asarray(pot[("b", "c")].values).copy()
      bc[0, :] += 1990.0
      pot[("b", "c")] = Factor(domain.project(("b", "c")), jax.numpy.asarray(bc))
      expected = marginal_oracles.message_passing_hugin(pot, 1.0)
      actual = native.oracle(pot, 1.0)
      for cl in cliques:
        np.testing.assert_allclose(
            np.asarray(actual[cl].values), np.asarray(expected[cl].values),
            rtol=1e-8, atol=1e-12,
        )


@unittest.skipUnless(native.is_available(), "mbi native library not built")
class TestThreadCountIndependence(unittest.TestCase):

  def test_same_results_with_1_and_4_threads(self):
    import subprocess  # pylint: disable=import-outside-toplevel
    script = (
        "import numpy as np, sys\n"
        "from mbi import Domain, Factor, LinearMeasurement, native\n"
        "rng = np.random.default_rng(0)\n"
        "dom = Domain([f'v{i}' for i in range(7)], [16]*7)\n"
        "cl = [('v0','v1','v2','v3','v4'), ('v3','v4','v5','v6'), ('v1',), ('v6',)]\n"
        "meas = [LinearMeasurement(rng.random(dom.size(c))*100, c, stddev=3.0) for c in cl]\n"
        "m = native.NativeAcceleratedMethod().estimate(dom, meas, iters=50, tol=None)\n"
        "np.save(sys.argv[1], np.concatenate([np.asarray(m.marginals[c].values, dtype=np.float64).ravel() for c in m.cliques]))\n"
    )
    import os, tempfile  # pylint: disable=import-outside-toplevel,multiple-imports
    outs = []
    with tempfile.TemporaryDirectory() as d:
      for t in ("1", "4"):
        path = os.path.join(d, f"out{t}.npy")
        env = dict(os.environ, MBI_NATIVE_THREADS=t, JAX_ENABLE_X64="1",
                   PYTHONWARNINGS="ignore")
        subprocess.run([sys.executable, "-c", script, path], env=env,
                       check=True, capture_output=True)
        outs.append(np.load(path))
    np.testing.assert_array_equal(outs[0], outs[1])


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
