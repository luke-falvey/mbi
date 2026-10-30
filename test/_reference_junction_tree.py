"""Reference implementations from ``mbi.junction_tree`` before the
performance work, used to check the optimized versions give identical results.

Do not edit: this file intentionally preserves the original behaviour.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mbi import junction_tree  # noqa: E402
from _reference_greedy_order import reference_greedy_order  # noqa: E402


def reference_hypothetical_model_size(domain, cliques):
  """Original: build the full junction tree and sum its clique sizes."""
  order = None
  graph = junction_tree._make_graph(domain, [tuple(c) for c in cliques])  # pylint: disable=protected-access
  import networkx as nx  # pylint: disable=import-outside-toplevel

  if not (not nx.is_empty(graph) and nx.is_tree(graph)):
    order = reference_greedy_order(domain, [tuple(c) for c in cliques])[0]
  jtree, _ = junction_tree.make_junction_tree(domain, cliques, order)
  max_cliques = junction_tree.maximal_cliques(jtree)
  cells = sum(domain.size(cl) for cl in max_cliques)
  return cells * 8 / 2**20
