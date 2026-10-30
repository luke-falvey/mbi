"""Utilities for constructing and working with junction trees.

This module provides functions for building junction trees from a given domain
and set of cliques. Junction trees are fundamental structures in graphical model
inference, enabling efficient message passing algorithms. Functions include
finding maximal cliques, determining message passing orders, graph triangulation,
computing greedy elimination orders, and estimating model size.
"""

import collections
import itertools
import math
from collections import OrderedDict
from collections.abc import Collection, Sequence
from typing import cast

import networkx as nx
import numpy as np

from .clique_utils import Clique
from .domain import Attribute
from .domain import Domain


def maximal_cliques(junction_tree: nx.Graph) -> list[Clique]:
  """Return the list of maximal cliques in the model."""
  return cast(list[Clique], list(nx.dfs_preorder_nodes(junction_tree)))


def message_passing_order(
    junction_tree: nx.Graph,
) -> list[tuple[Clique, Clique]]:
  """Return a valid message passing order."""
  edges = []
  messages = list(junction_tree.edges()) + [
      (b, a) for a, b in junction_tree.edges()
  ]
  for m1 in messages:
    for m2 in messages:
      if m1[1] == m2[0] and m1[0] != m2[1]:
        edges.append((m1, m2))
  graph = nx.DiGraph()
  graph.add_nodes_from(messages)
  graph.add_edges_from(edges)
  return cast(list[tuple[Clique, Clique]], list(nx.topological_sort(graph)))


def _make_graph(domain: Domain, cliques: Collection[Clique]) -> nx.Graph:
  """Create a graph from the domain and cliques."""
  graph = nx.Graph()
  graph.add_nodes_from(domain.attributes)  # pyrefly: ignore[bad-argument-type]
  for cl in cliques:
    # pyrefly: ignore[bad-argument-type]
    graph.add_edges_from(itertools.combinations(cl, 2))
  return graph


def _triangulated(graph: nx.Graph, order: list[Attribute]) -> nx.Graph:
  """Triangulate the graph using the given elimination order."""
  edges = set()
  graph2 = nx.Graph(graph)
  for node in order:
    tmp = set(itertools.combinations(graph2.neighbors(node), 2))
    edges |= tmp
    graph2.add_edges_from(tmp)
    graph2.remove_node(node)
  tri = nx.Graph(graph)
  tri.add_edges_from(edges)
  return tri


def greedy_order(
    domain: Domain,
    cliques: Sequence[Clique],
    stochastic: bool = False,
    elim: list[Attribute] | None = None,
) -> tuple[list[Attribute], int]:
  """Compute a greedy elimination order.

  At each step, eliminates the attribute whose elimination creates the
  smallest factor (ties broken by position in ``elim`` / the domain), merging
  every clique that contains it. Costs are cached and only recomputed for the
  attributes adjacent to the one just eliminated, since no other attribute's
  neighborhood changes.
  """
  order = []
  unmarked = list(elim) if elim is not None else list(domain.attributes)
  # Cliques may mention attributes outside ``domain`` (e.g. variable
  # elimination's query clique); like the original, their sizes are only
  # looked up if actually needed.
  size = domain.config
  # Attribute -> set of (ids of) current cliques containing it.
  cliques_by_id: dict[int, frozenset] = {}
  containing: dict = collections.defaultdict(set)
  next_id = itertools.count()
  for cl in dict.fromkeys(frozenset(c) for c in cliques):
    k = next(next_id)
    cliques_by_id[k] = cl
    for v in cl:
      containing[v].add(k)

  def cost_of(a) -> int:
    variables = set()
    for k in containing[a]:
      variables |= cliques_by_id[k]
    return math.prod(size[v] for v in variables)

  cost = {a: cost_of(a) for a in unmarked}
  total_cost = 0
  for _ in range(len(unmarked)):
    if stochastic:
      costs = np.array([cost[a] for a in unmarked], dtype=float)
      probas = np.max(costs) - costs + 1
      probas /= probas.sum()
      a = unmarked[np.random.choice(probas.size, p=probas)]
    else:
      a = min(unmarked, key=cost.__getitem__)

    order.append(a)
    unmarked.remove(a)
    total_cost += cost[a]
    neighbors = containing[a].copy()
    variables = set()
    for k in neighbors:
      variables |= cliques_by_id[k]
    variables.discard(a)
    for k in neighbors:
      for v in cliques_by_id.pop(k):
        containing[v].discard(k)
    merged = frozenset(variables)
    k = next(next_id)
    cliques_by_id[k] = merged
    for v in merged:
      containing[v].add(k)
    for v in merged:
      if v in cost:
        cost[v] = cost_of(v)

  return order, total_cost


def make_junction_tree(
    domain: Domain,
    cliques: Collection[Clique],
    elimination_order: list[Attribute] | int | None = None,
    connected: bool = False,
) -> tuple[nx.Graph, list[Attribute]]:
  """Create a junction tree."""
  cliques = [tuple(cl) for cl in cliques]
  graph = _make_graph(domain, cliques)

  if elimination_order is None and not nx.is_empty(graph) and nx.is_tree(graph):
    elimination_order = cast(
        list[Attribute], list(nx.dfs_postorder_nodes(graph))
    )
  elif elimination_order is None:
    elimination_order = greedy_order(domain, cliques, stochastic=False)[0]
  elif isinstance(elimination_order, int):
    orders = [greedy_order(domain, cliques, stochastic=False)] + [
        greedy_order(domain, cliques, stochastic=True)
        for _ in range(elimination_order)
    ]
    elimination_order = min(orders, key=lambda x: x[1])[0]

  tri = _triangulated(graph, elimination_order)
  cliques = sorted(
      [domain.canonical(cast(Clique, c)) for c in nx.find_cliques(tri)]
  )
  complete = nx.Graph()
  complete.add_nodes_from(cliques)
  for c1, c2 in itertools.combinations(cliques, 2):
    wgt = len(set(c1) & set(c2))
    if wgt > 0:
      complete.add_edge(c1, c2, weight=-wgt)
  spanning = nx.minimum_spanning_tree(complete)
  if connected:
    roots = [next(iter(c)) for c in nx.connected_components(spanning)]
    spanning.add_edges_from(itertools.pairwise(roots))
  return spanning, elimination_order


def hypothetical_model_size(domain: Domain, cliques: Sequence[Clique]) -> float:
  """Size of the full junction tree parameters, measured in megabytes.

  Equal to summing the maximal-clique sizes of ``make_junction_tree(domain,
  cliques)``, but computed directly from the elimination: under a perfect
  elimination order the maximal cliques of the triangulated graph are the
  maximal sets among ``{v} | neighbors(v)`` at the time ``v`` is eliminated,
  so no graph objects, clique enumeration or spanning tree are needed. This
  is called for every candidate clique in every round of AIM.
  """
  cliques = [tuple(cl) for cl in cliques]
  attrs = list(domain.attributes)
  adjacency = {a: set() for a in attrs}
  n_edges = 0
  for cl in cliques:
    for u, v in itertools.combinations(cl, 2):
      if v not in adjacency[u]:
        adjacency[u].add(v)
        adjacency[v].add(u)
        n_edges += 1
  # make_junction_tree uses a DFS post-order (no fill-in) when the graph is a
  # tree; otherwise the greedy order.
  if n_edges > 0 and n_edges == len(attrs) - 1 and _is_connected(adjacency):
    # A tree triangulates without fill-in: its maximal cliques are its edges.
    maximal = list({frozenset((u, v)) for u in attrs for v in adjacency[u]})
  else:
    order = greedy_order(domain, cliques, stochastic=False)[0]
    adj = {a: set(n) for a, n in adjacency.items()}
    elim_cliques = []
    for v in order:
      nbrs = adj.pop(v)
      elim_cliques.append(frozenset(nbrs | {v}))
      for u in nbrs:
        adj[u].discard(v)
        adj[u] |= nbrs - {u}
    elim_cliques = list(dict.fromkeys(elim_cliques))
    maximal = [
        c for c in elim_cliques if not any(c < other for other in elim_cliques)
    ]
  cells = sum(math.prod(domain[a] for a in c) for c in maximal)
  return cells * 8 / 2**20


def _is_connected(adjacency) -> bool:
  if not adjacency:
    return True
  start = next(iter(adjacency))
  seen, stack = {start}, [start]
  while stack:
    for u in adjacency[stack.pop()]:
      if u not in seen:
        seen.add(u)
        stack.append(u)
  return len(seen) == len(adjacency)
