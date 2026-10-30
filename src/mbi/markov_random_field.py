"""Defines the MarkovRandomField class representing learned graphical models.

This module provides the `MarkovRandomField` class, which encapsulates the
results of learning a graphical model. It stores the learned potentials,
the resulting marginal distributions, and the associated total count (e.g.,
number of records). It also offers methods for querying marginals and
generating synthetic data.
"""

from collections.abc import Sequence
import math

import chex
import numpy as np
from jax.typing import ArrayLike

from . import junction_tree, marginal_oracles
from .clique_utils import Clique
from .clique_vector import CliqueVector
from .constraint import Constraint
from .dataset import Dataset
from .domain import Attribute
from .domain import Domain
from .factor import Factor


# Use a dense bincount over all parent configurations when there are at most
# this many of them; otherwise fall back to a 1-D sort-based np.unique.
_DENSE_GROUPING_LIMIT = 1 << 24


def _group_rows(rows: np.ndarray, sizes: tuple[int, ...]):
  """Fast equivalent of ``np.unique(rows, axis=0, ...)`` plus a stable argsort.

  Returns ``(uniques, inverse, counts, perm)`` where the first three match
  ``np.unique(rows, axis=0, return_inverse=True, return_counts=True)`` and
  ``perm == np.argsort(inverse, kind="stable")``.  Each row is encoded as a
  single integer (row-major, so integer order equals lexicographic row order),
  which avoids ``np.unique``'s slow multi-column sort.  Group ids are sorted
  in the smallest unsigned dtype so NumPy can use radix sort.
  """
  keys = np.ravel_multi_index(tuple(rows.T), sizes)
  n_configs = math.prod(sizes)
  if n_configs <= _DENSE_GROUPING_LIMIT:
    full_counts = np.bincount(keys, minlength=n_configs)
    present = full_counts > 0
    unique_keys = np.flatnonzero(present)
    inverse = (np.cumsum(present) - 1)[keys]
    counts = full_counts[unique_keys]
  else:
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
  small = inverse.astype(np.min_scalar_type(max(len(unique_keys) - 1, 0)))
  perm = np.argsort(small, kind="stable")
  uniques = np.stack(np.unravel_index(unique_keys, sizes), axis=-1)
  return uniques, inverse.astype(np.intp), counts, perm


@chex.dataclass(frozen=True, kw_only=False)
class MarkovRandomField:
  """Represents a learned graphical model.

  This class encapsulates the components of a Markov Random Field that has been
  learned from data. It stores the learned potentials, the resulting marginal
  distributions over specified cliques, and the total count (e.g., number of
  records or equivalent sample size) associated with the model.

  Attributes:
      potentials (CliqueVector): A `CliqueVector` containing the learned
          potential functions for the cliques in the model.
      marginals (CliqueVector): A `CliqueVector` containing the marginal
          distributions for a set of cliques, derived from the potentials.
      total (ArrayLike): The total count or effective sample size
          represented by the model. This is often used for scaling or
          interpreting the marginals.
      constraints (tuple[Constraint, ...]): Structural constraints the model
          was fit under, used by ``synthetic_data`` so generated records
          respect them. Empty if the model was fit without constraints.
  """

  potentials: CliqueVector
  marginals: CliqueVector
  total: ArrayLike = 1
  constraints: tuple[Constraint, ...] = ()

  def project(self, attrs: Attribute | Sequence[Attribute]) -> Factor:
    if isinstance(attrs, (str, int)):
      attrs = (attrs,)
    attrs = tuple(attrs)
    if self.marginals.supports(attrs):
      return self.marginals.project(attrs)
    return marginal_oracles.variable_elimination(
        self.potentials, attrs, float(self.total)  # pyrefly: ignore[bad-argument-type]
    )

  def supports(self, attrs: Attribute | Sequence[Attribute]) -> bool:
    return self.marginals.domain.supports(attrs)

  def synthetic_data(
      self,
      rows: int | None = None,
      method: str = "round",
      use_native: bool = False,
  ) -> Dataset:
    """Generates synthetic data based on the learned model's marginals.

    Args:
        rows: The number of rows to generate. If not provided, uses the
              model total, which is usually estimated automatically.
        method: Specification for strategy to use to generate records.
                - "round" for randomized rounding
                - "sample" for i.i.d sampling
        use_native: Compute the junction-tree marginals with the native
                library (``mbi.native``) in float64 instead of JAX, which
                avoids compiling a message-passing program for this model.
                Ignored if the library is unavailable or the model has
                constraints. The generated data is statistically equivalent
                but, because marginals can differ in the last bits, need not
                be identical to the default for the same seed.

    Returns:
        A synthetic dataset whose marginals should closely match those of the
        model.
    """
    total = max(1, int(rows or self.total))  # pyrefly: ignore[bad-argument-type]
    domain = self.domain
    jtree, elimination_order = junction_tree.make_junction_tree(
        domain, [tuple(cl) for cl in self.cliques]
    )

    # Use maximal cliques from the junction tree for conditioning
    # decisions (not the original measurement cliques).  The junction
    # tree merges overlapping cliques into super-cliques that capture
    # the full dependency structure of the model.
    cliques = [set(cl) for cl in jtree.nodes]

    from . import native  # pylint: disable=import-outside-toplevel

    if use_native and not self.constraints and native.is_available():
      potentials = native.expand(self.potentials, list(jtree.nodes))
      marginals = native.NumpyMarginals.from_potentials(
          potentials, float(self.total)
      )
    elif self.constraints:
      potentials = self.potentials.expand(list(jtree.nodes))
      # Shafer-Shenoy handles the -inf constraint potentials that HUGIN can't.
      marginals = marginal_oracles.message_passing_shafer_shenoy(
          potentials, self.total, constraints=self.constraints
      )
    else:
      potentials = self.potentials.expand(list(jtree.nodes))
      marginals = marginal_oracles.message_passing_stable(
          potentials, self.total
      )

    def synthetic_col(counts, total):
      """Generates a synthetic column by sampling or rounding based on counts and total."""
      counts = np.asarray(counts, dtype=np.float64)
      dtype = np.min_scalar_type(counts.size)
      options = np.arange(counts.size, dtype=dtype)
      if total == 0:
        return np.array([], dtype=int)
      if method == "sample":
        probas = counts / counts.sum()
        return np.random.choice(options, total, True, probas)
      counts = counts * (total / counts.sum())
      frac, integ = np.modf(counts)
      integ = integ.astype(int)
      extra = total - integ.sum()
      if extra > 0:
        idx = np.random.choice(options, extra, False, frac / frac.sum())
        integ[idx] += 1
      vals = np.repeat(options, integ)
      np.random.shuffle(vals)
      return vals

    data = {}
    order = elimination_order[::-1]
    col = order[0]
    marg = marginals.project((col,)).datavector(flatten=False)
    data[col] = synthetic_col(marg, total)
    used = {col}

    for col in order[1:]:
      relevant = [cl for cl in cliques if col in cl]
      relevant = used.intersection(set().union(*relevant))
      proj = tuple(relevant)
      used.add(col)

      if len(proj) >= 1:
        current_proj_data = np.stack(tuple(data[col] for col in proj), -1)

        marg = np.asarray(
            marginals.project(proj + (col,)).datavector(flatten=False)
        )

        marg_parents = marg.sum(axis=-1, keepdims=True)
        cond_probs = np.divide(
            marg,
            marg_parents,
            out=np.zeros_like(marg),
            where=marg_parents != 0,
        )
        cond_cdfs = cond_probs.cumsum(axis=-1)

        uniques, inverse, counts, perm = _group_rows(
            current_proj_data, tuple(self.domain[c] for c in proj)
        )
        if method == "sample":
          u = np.random.rand(total)
        else:
          group_starts = np.zeros(len(counts), dtype=int)
          np.cumsum(counts[:-1], out=group_starts[1:])

          # Shuffle within each parent group to break
          # spurious correlations with previously generated
          # columns that shared the same parent set.
          for gi in range(len(counts)):
            s = group_starts[gi]
            e = s + counts[gi]
            np.random.shuffle(perm[s:e])

          inverse_sorted = inverse[perm]
          sorted_indices = np.arange(total)

          ranks_sorted = sorted_indices - group_starts[inverse_sorted]

          ranks = np.empty(total, dtype=int)
          ranks[perm] = ranks_sorted

          noise = np.random.rand(total)
          u = (ranks + noise) / counts[inverse]

        indices = tuple(uniques.T)
        unique_cdfs = cond_cdfs[indices]

        choices = np.empty(total, dtype=np.min_scalar_type(self.domain[col]))
        domain_size = self.domain[col]

        u_sorted = u[perm]

        start = 0
        for i, count in enumerate(counts):
          end = start + count
          cdf = unique_cdfs[i]
          indices_chunk = np.searchsorted(
              cdf, u_sorted[start:end], side="right"
          )
          if len(indices_chunk) > 0:
            np.minimum(indices_chunk, domain_size - 1, out=indices_chunk)
            choices[perm[start:end]] = indices_chunk
          start = end

        data[col] = choices

      else:
        marg = marginals.project((col,)).datavector(flatten=False)
        data[col] = synthetic_col(marg, total)

    return Dataset(data, domain)

  @property
  def domain(self) -> Domain:
    """Returns the Domain object associated with this graphical model."""
    return self.potentials.domain

  @property
  def cliques(self) -> Sequence[Clique]:
    """Returns the list of cliques the model's potentials are defined over."""
    return self.potentials.cliques
