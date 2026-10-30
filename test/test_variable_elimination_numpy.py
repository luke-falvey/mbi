"""variable_elimination_numpy must match the jitted variable_elimination."""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized

from mbi import CliqueVector, Domain, Factor, marginal_oracles

_MODELS = {
    "chain": (
        Domain(list("abcdef"), [2, 3, 4, 5, 3, 2]),
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")],
    ),
    "loopy": (
        Domain(list("abcde"), [3, 2, 4, 2, 3]),
        [("a", "b", "c"), ("c", "d"), ("d", "e"), ("a", "e")],
    ),
    "disconnected": (
        Domain(list("wxyz"), [4, 3, 2, 5]),
        [("w", "x"), ("y",), ("z",)],
    ),
    "uncovered_attribute": (
        # "u" is in the domain but no potential mentions it.
        Domain(list("pqru"), [3, 4, 2, 3]),
        [("p", "q"), ("q", "r")],
    ),
}


def _queries(domain):
  attrs = domain.attributes
  out = [(a,) for a in attrs]
  out += list(itertools.combinations(attrs, 2))
  out += [tuple(reversed(c)) for c in itertools.combinations(attrs, 2)][:3]
  out += list(itertools.combinations(attrs, 3))[:4]
  out.append(tuple(attrs))
  return out


class TestVariableEliminationNumpy(unittest.TestCase):

  @parameterized.expand([(name,) for name in _MODELS])
  def test_matches_jax(self, name):
    domain, cliques = _MODELS[name]
    np.random.seed(0)
    potentials = CliqueVector.random(domain, cliques) * 3.0
    for clique in _queries(domain):
      for total in (1.0, 1234.5):
        expected = marginal_oracles.variable_elimination(
            potentials, clique, total
        )
        actual = marginal_oracles.variable_elimination_numpy(
            potentials, clique, total
        )
        self.assertEqual(expected.domain, actual.domain, clique)
        np.testing.assert_allclose(
            np.asarray(actual.values),
            np.asarray(expected.values),
            rtol=1e-4,
            atol=1e-5 * total,
            err_msg=f"{name} {clique} total={total}",
        )
        self.assertAlmostEqual(
            float(np.sum(actual.values)) / total, 1.0, places=5
        )

  def test_matches_brute_force_in_float64(self):
    prev = bool(jax.config.jax_enable_x64)  # pylint: disable=no-member
    jax.config.update("jax_enable_x64", True)
    try:
      domain, cliques = _MODELS["loopy"]
      np.random.seed(1)
      potentials = CliqueVector.random(domain, cliques)
      joint = sum(
          (potentials[cl].expand(domain) for cl in cliques),
          Factor.zeros(domain),
      )
      joint = joint.exp()
      joint = joint / joint.sum()
      for clique in _queries(domain):
        expected = np.asarray(joint.project(clique).datavector(flatten=False))
        actual = np.asarray(
            marginal_oracles.variable_elimination_numpy(
                potentials, clique, 1.0
            ).values
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
    finally:
      jax.config.update("jax_enable_x64", prev)
      jax.clear_caches()

  def test_neg_inf_potentials(self):
    domain = Domain(list("abc"), [2, 3, 2])
    cliques = [("a", "b"), ("b", "c")]
    np.random.seed(2)
    potentials = CliqueVector.random(domain, cliques)
    ab = np.asarray(potentials[("a", "b")].values).copy()
    ab[0, :] = -np.inf  # a == 0 is impossible
    potentials[("a", "b")] = Factor(domain.project(("a", "b")), jnp.asarray(ab))
    for clique in [("a",), ("a", "c"), ("c",)]:
      expected = marginal_oracles.variable_elimination(potentials, clique, 10.0)
      actual = marginal_oracles.variable_elimination_numpy(
          potentials, clique, 10.0
      )
      np.testing.assert_allclose(
          np.asarray(actual.values), np.asarray(expected.values),
          rtol=1e-4, atol=1e-5,
      )
    a = np.asarray(
        marginal_oracles.variable_elimination_numpy(
            potentials, ("a",), 10.0
        ).values
    )
    self.assertEqual(a[0], 0.0)


if __name__ == "__main__":
  unittest.main()
