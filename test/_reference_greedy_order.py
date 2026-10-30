"""Verbatim copy of ``mbi.junction_tree.greedy_order`` before the performance
work, used as a reference to check the optimized version returns identical
elimination orders and costs.

Do not edit: this file intentionally preserves the original implementation.
"""

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np


def reference_greedy_order(
    domain,
    cliques,
    stochastic: bool = False,
    elim=None,
):
  """Compute a greedy elimination order."""
  order = []
  unmarked = elim if elim is not None else list(domain.attributes)
  clique_set = set(cliques)
  total_cost = 0
  for _ in range(len(unmarked)):
    cost = OrderedDict()
    for a in unmarked:
      neighbors = [cl for cl in clique_set if a in cl]
      variables = tuple(set.union(set(), *[set(n) for n in neighbors]))
      newdom = domain.project(variables)
      cost[a] = newdom.size()

    if stochastic:
      choices = list(unmarked)
      costs = np.array([cost[a] for a in choices], dtype=float)
      probas = np.max(costs) - costs + 1
      probas /= probas.sum()
      i = np.random.choice(probas.size, p=probas)
      a = choices[i]
    else:
      a = min(cost, key=lambda a: cost[a])

    order.append(a)
    unmarked.remove(a)
    neighbors = [cl for cl in clique_set if a in cl]
    variables = tuple(set.union(set(), *[set(n) for n in neighbors]) - {a})
    clique_set -= set(neighbors)
    clique_set.add(variables)
    total_cost += cost[a]

  return order, total_cost
