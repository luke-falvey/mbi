"""Native (Rust) implementation of MirrorDescent estimation.

``NativeMirrorDescent`` runs the whole ``MirrorDescent.estimate`` loop
(HUGIN message passing, L2 loss over identity measurements, Armijo line
search, primal-dual-gap early stopping) in a small Rust library
(``native/`` in the repository) that is loaded with ``ctypes``.

Why: the JAX implementation compiles new XLA programs for every new model
structure. Workflows that fit many differently-structured models, such as the
AIM mechanism, spend most of their time compiling. The native library takes
the structure as data (index arrays built here, once per structure), so
nothing is compiled at run time.

The library is optional. Build it with::

    python -m mbi.native build      # requires a Rust toolchain (cargo)

or point ``MBI_NATIVE_LIB`` at a prebuilt ``libmbi_native.so``. When it is
unavailable, or for settings it does not support (constraints, callbacks,
non-identity queries, custom oracles or sharding), ``NativeMirrorDescent``
falls back to the JAX ``MirrorDescent`` with identical semantics.

Results match ``MirrorDescent`` up to floating-point rounding: the native code
computes in float64 (JAX defaults to float32), and it reuses the marginals
from the previous line-search trial instead of recomputing them, which gives
the same values with half the message-passing work.
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
import math
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import networkx as nx
import numpy as np

from . import estimation, junction_tree, marginal_loss
from .clique_utils import Clique, reverse_clique_mapping
from .clique_vector import CliqueVector
from .constraint import Constraint
from .domain import Domain
from .factor import Factor
from .marginal_loss import DatavectorQuery, LinearMeasurement, MarginalLossFn
from .markov_random_field import MarkovRandomField

__all__ = ["NativeMirrorDescent", "NativeAcceleratedMethod", "marginal_counts", "is_available", "build", "library_path"]

_LIB_NAME = {
    "darwin": "libmbi_native.dylib",
    "win32": "mbi_native.dll",
}.get(sys.platform, "libmbi_native.so")
_PACKAGE_LIB = pathlib.Path(__file__).with_name("_native") / _LIB_NAME
_REPO_CRATE = pathlib.Path(__file__).resolve().parents[2] / "native"


def library_path() -> pathlib.Path | None:
  """Location of the native library, or None if it can't be found."""
  candidates = []
  if os.environ.get("MBI_NATIVE_LIB"):
    candidates.append(pathlib.Path(os.environ["MBI_NATIVE_LIB"]))
  candidates.append(_PACKAGE_LIB)
  candidates.append(_REPO_CRATE / "target" / "release" / _LIB_NAME)
  for path in candidates:
    if path.is_file():
      return path
  return None


def build(crate_dir: str | os.PathLike | None = None) -> pathlib.Path:
  """Build the Rust library with cargo and install it next to this module."""
  crate = pathlib.Path(crate_dir) if crate_dir else _REPO_CRATE
  if not (crate / "Cargo.toml").is_file():
    raise FileNotFoundError(f"No Cargo.toml in {crate}")
  env = dict(os.environ)
  # Optimize for the CPU the library is built on (enables FMA / AVX); the
  # library is meant to be built where it runs.
  env.setdefault("RUSTFLAGS", "-C target-cpu=native")
  subprocess.run(
      ["cargo", "build", "--release", "--offline"], cwd=crate, check=True,
      env=env,
  )
  _PACKAGE_LIB.parent.mkdir(exist_ok=True)
  shutil.copy2(crate / "target" / "release" / _LIB_NAME, _PACKAGE_LIB)
  _load.cache_clear()
  return _PACKAGE_LIB


_P = ctypes.POINTER


class _PlanC(ctypes.Structure):
  _fields_ = [
      ("n_l", ctypes.c_uint64), ("l_off", _P(ctypes.c_uint64)),
      ("n_j", ctypes.c_uint64), ("j_off", _P(ctypes.c_uint64)),
      ("exp_parent", _P(ctypes.c_uint32)),
      ("exp_map_off", _P(ctypes.c_uint64)), ("exp_map", _P(ctypes.c_uint32)),
      ("exp_bk", _P(ctypes.c_uint64)),
      ("n_e", ctypes.c_uint64),
      ("e_src", _P(ctypes.c_uint32)), ("e_dst", _P(ctypes.c_uint32)),
      ("e_rev", _P(ctypes.c_uint64)), ("sep_off", _P(ctypes.c_uint64)),
      ("src_map_off", _P(ctypes.c_uint64)), ("src_map", _P(ctypes.c_uint32)),
      ("dst_map_off", _P(ctypes.c_uint64)), ("dst_map", _P(ctypes.c_uint32)),
      ("src_bk", _P(ctypes.c_uint64)), ("dst_bk", _P(ctypes.c_uint64)),
      ("mar_parent", _P(ctypes.c_uint32)),
      ("mar_map_off", _P(ctypes.c_uint64)), ("mar_map", _P(ctypes.c_uint32)),
      ("mar_bk", _P(ctypes.c_uint64)),
      ("total_clique", ctypes.c_uint64),
      ("n_m", ctypes.c_uint64),
      ("m_l", _P(ctypes.c_uint32)), ("m_off", _P(ctypes.c_uint64)),
      ("y", _P(ctypes.c_double)), ("m_sd", _P(ctypes.c_double)),
      ("m_map_off", _P(ctypes.c_uint64)), ("m_map", _P(ctypes.c_uint32)),
      ("m_bk", _P(ctypes.c_uint64)),
      ("n_h", ctypes.c_uint64), ("h_parent", _P(ctypes.c_uint64)),
      ("h_j", _P(ctypes.c_uint32)), ("h_off", _P(ctypes.c_uint64)),
      ("h_map_off", _P(ctypes.c_uint64)), ("h_map", _P(ctypes.c_uint32)),
      ("h_bk", _P(ctypes.c_uint64)), ("l_attach", _P(ctypes.c_uint64)),
      ("mar_lead", _P(ctypes.c_uint64)), ("h_lead", _P(ctypes.c_uint64)),
      ("src_lead", _P(ctypes.c_uint64)),
  ]


@functools.cache
def _load():
  path = library_path()
  if path is None:
    return None
  lib = ctypes.CDLL(str(path))
  dbl, u64, u32 = ctypes.c_double, ctypes.c_uint64, ctypes.c_uint32
  lib.mbi_oracle.argtypes = [_P(_PlanC), _P(dbl), dbl, u32, _P(dbl)]
  lib.mbi_oracle.restype = ctypes.c_int32
  lib.mbi_mirror_descent.argtypes = [
      _P(_PlanC), _P(dbl), _P(dbl), dbl, dbl, dbl, dbl, u64, u64, dbl, dbl,
      _P(dbl),
  ]
  lib.mbi_mirror_descent.restype = ctypes.c_int32
  lib.mbi_accelerated.argtypes = [
      _P(_PlanC), _P(dbl), _P(dbl), _P(dbl), dbl, dbl, u64, u64, dbl, u64,
      dbl, dbl, _P(dbl),
  ]
  lib.mbi_accelerated.restype = ctypes.c_int32
  lib.mbi_count.argtypes = [
      u64, _P(ctypes.c_void_p), u64, _P(u64), _P(u32), _P(u32), _P(u64),
      _P(dbl), _P(dbl),
  ]
  lib.mbi_count.restype = ctypes.c_int32
  lib.mbi_sample_column.argtypes = [
      u64, u64, _P(ctypes.c_void_p), _P(u32), _P(dbl), u64, u32, u64,
      _P(u32),
  ]
  lib.mbi_sample_column.restype = ctypes.c_int32
  lib.mbi_beliefs.argtypes = [_P(_PlanC), _P(dbl), dbl, _P(dbl)]
  lib.mbi_beliefs.restype = ctypes.c_int32
  lib.mbi_model_sizes.argtypes = [
      u64, _P(u64), u64, _P(u64), _P(u32), u64, _P(u64), _P(u32), _P(dbl)
  ]
  lib.mbi_model_sizes.restype = ctypes.c_int32
  return lib


def is_available() -> bool:
  """Whether the native library can be loaded."""
  return _load() is not None


# ---------------------------------------------------------------------------
# Plan construction: the model structure as flat index arrays.
# ---------------------------------------------------------------------------


def _index_map(domain: Domain, big: Clique, small: Clique) -> np.ndarray:
  """Flat index into ``small`` (row-major) of every cell of ``big``."""
  shape = tuple(domain[a] for a in big)
  strides, s = {}, 1
  for a in reversed(small):
    strides[a] = s
    s *= domain[a]
  out = np.zeros(shape, dtype=np.int64)
  for axis, a in enumerate(big):
    if a in strides:
      idx = np.arange(domain[a], dtype=np.int64) * strides[a]
      out += idx.reshape([-1 if k == axis else 1 for k in range(len(big))])
  return out.reshape(-1).astype(np.uint32)


def _block(domain: Domain, big: Clique, small: Clique) -> int:
  """Block descriptor ``block << 1 | kept`` for the map ``big -> small``.

  The big table's trailing axes form runs of ``block`` cells that are either
  *kept* (they are also the trailing axes of ``small``, in the same order, so
  consecutive big cells map to consecutive small cells) or all *reduced*
  (absent from ``small``, so the run maps to a single small cell). The native
  kernels iterate block-wise, which vectorizes.
  """
  if not big:
    return 1 << 1 | 1
  kept = big[-1] in small
  t = 0
  if kept:
    while t < min(len(big), len(small)) and big[-1 - t] == small[-1 - t]:
      t += 1
  else:
    while t < len(big) and big[-1 - t] not in small:
      t += 1
  block = math.prod(domain[a] for a in big[len(big) - t :]) if t else 1
  return block << 1 | int(kept)


def _lead(domain: Domain, big: Clique, small: Clique) -> int:
  """Cells of ``big`` per value of the leading attributes it shares (in the
  same order) with ``small``; 0 if the first attributes differ. Splitting
  ``big`` at multiples of this maps each part to a disjoint range of
  ``small``, so read-outs can run in parallel without partial buffers."""
  t = 0
  while t < min(len(big), len(small)) and big[t] == small[t]:
    t += 1
  return math.prod(domain[a] for a in big[t:]) if t else 0


def _offsets(sizes) -> np.ndarray:
  return np.concatenate([[0], np.cumsum(sizes, dtype=np.uint64)]).astype(
      np.uint64
  )


def _parent(clique: Clique, candidates: Sequence[Clique], domain: Domain) -> int:
  """Mirrors ``CliqueVector.project``'s choice of table for ``clique``."""
  if clique in candidates:
    return list(candidates).index(clique)
  options = [i for i, c in enumerate(candidates) if set(clique) <= set(c)]
  return min(options, key=lambda i: domain.size(candidates[i]))


@dataclasses.dataclass
class _Structure:
  """Index arrays describing one model structure (cached across fits)."""

  arrays: dict
  n_l: int
  n_j: int
  n_e: int
  n_m: int
  total_clique: int
  loss_cliques: tuple
  cells: int
  maximal: tuple = ()
  tree_edges: tuple = ()


def _build_hierarchy(domain, attrs, parent, j, leaves, nodes, attach):
  """Greedily add intermediate tables below table ``attrs`` (see ``Hier``).

  Repeatedly considers dropping one attribute: an intermediate table over
  the remaining attributes pays off when enough of the loss cliques fit in
  it (each is then read from / summed into the smaller table instead of the
  big one, at the price of one extra pass over the big table). Recurses into
  each intermediate table. ``nodes`` collects ``(attrs, parent, j)``;
  ``attach`` maps loss-clique index -> node (``None``: the junction-tree
  table itself).
  """
  remaining = list(leaves)
  size = domain.size(attrs)
  while True:
    best = None
    for a in attrs:
      sub = tuple(x for x in attrs if x != a)
      if not sub:
        continue
      covered = [it for it in remaining if a not in it[1]]
      if len(covered) < 2:
        continue
      saving = len(covered) * (size - domain.size(sub)) - size
      if best is None or saving > best[0]:
        best = (saving, sub, covered)
    if best is None or best[0] <= 0:
      break
    _, sub, covered = best
    node = len(nodes)
    nodes.append((sub, parent, j))
    ids = {it[0] for it in covered}
    remaining = [it for it in remaining if it[0] not in ids]
    _build_hierarchy(domain, sub, node, j, covered, nodes, attach)
  for i, _ in remaining:
    attach[i] = parent


def _rooted_schedule(tree: nx.Graph, root) -> list:
  """Messages (u, v) for a collect (post-order, towards ``root``) then
  distribute (pre-order, away from it) pass over a tree."""
  parent, preorder = {root: None}, []
  stack = [root]
  while stack:
    u = stack.pop()
    preorder.append(u)
    for v in sorted(tree.neighbors(u), key=str):
      if v not in parent:
        parent[v] = u
        stack.append(v)
  collect = [(u, parent[u]) for u in reversed(preorder) if parent[u] is not None]
  distribute = [(parent[u], u) for u in preorder if parent[u] is not None]
  return collect + distribute


@functools.lru_cache(maxsize=64)
def _structure(
    domain: Domain, loss_cliques: tuple, measurement_cliques: tuple
) -> _Structure:
  jtree = junction_tree.make_junction_tree(
      domain, loss_cliques, connected=True
  )[0]
  order = junction_tree.message_passing_order(jtree)
  maximal = junction_tree.maximal_cliques(jtree)
  # Internal layout of each junction-tree table: attributes from smallest to
  # largest domain, so trailing runs (and hence vectorized blocks) are long.
  position = {a: i for i, a in enumerate(domain.attributes)}
  layout_key = lambda a: (domain[a], position[a])
  layout = {c: tuple(sorted(c, key=layout_key)) for c in maximal}
  maximal = [layout[c] for c in maximal]
  jtree = nx.relabel_nodes(jtree, layout)
  # HUGIN schedule rooted at the largest clique: collect towards it, then
  # distribute from it. Any valid schedule gives the same calibration (up to
  # rounding); rooting at the largest clique means it sends all of its
  # messages after it is calibrated, so the native kernels compute them from
  # its probability table (shared with the marginals) without an extra exp
  # pass over it.
  del order
  order = _rooted_schedule(jtree, max(maximal, key=domain.size))
  jindex = {c: i for i, c in enumerate(maximal)}

  rev = reverse_clique_mapping(maximal, loss_cliques, domain=domain)
  exp_parent = np.zeros(len(loss_cliques), dtype=np.uint32)
  for m, cls in rev.items():
    for cl in cls:
      exp_parent[loss_cliques.index(cl)] = jindex[m]
  mar_parent = np.array(
      [_parent(cl, maximal, domain) for cl in loss_cliques], dtype=np.uint32
  )
  # Intermediate tables for loss cliques that are expanded into and read
  # from the same junction-tree table.
  nodes, attach = [], {}
  for j, big in enumerate(maximal):
    leaves = [
        (i, cl) for i, cl in enumerate(loss_cliques)
        if exp_parent[i] == j and mar_parent[i] == j
    ]
    _build_hierarchy(domain, big, None, j, leaves, nodes, attach)
  attach_attrs = lambda i: (
      nodes[attach[i]][0] if attach.get(i) is not None
      else maximal[exp_parent[i]]
  )
  exp_maps = [
      _index_map(domain, attach_attrs(i), cl) for i, cl in enumerate(loss_cliques)
  ]
  exp_bk = [
      _block(domain, attach_attrs(i), cl) for i, cl in enumerate(loss_cliques)
  ]
  parent_attrs = lambda n: (
      nodes[nodes[n][1]][0] if nodes[n][1] is not None else maximal[nodes[n][2]]
  )
  h_maps = [_index_map(domain, parent_attrs(n), nodes[n][0]) for n in range(len(nodes))]
  h_bk = [_block(domain, parent_attrs(n), nodes[n][0]) for n in range(len(nodes))]
  h_lead = [_lead(domain, parent_attrs(n), nodes[n][0]) for n in range(len(nodes))]

  edge_index = {e: k for k, e in enumerate(order)}
  e_src, e_dst, e_rev, sep_sizes, src_maps, dst_maps = [], [], [], [], [], []
  src_bk, dst_bk, src_lead = [], [], []
  for i, j in order:
    # Same layout rule as the cliques, so a message and its reverse share it.
    sep = tuple(sorted(set(i) & set(j), key=layout_key))
    src_bk.append(_block(domain, i, sep))
    src_lead.append(_lead(domain, i, sep))
    dst_bk.append(_block(domain, j, sep))
    e_src.append(jindex[i])
    e_dst.append(jindex[j])
    e_rev.append(edge_index[(j, i)])
    sep_sizes.append(domain.size(sep))
    src_maps.append(_index_map(domain, i, sep))
    dst_maps.append(_index_map(domain, j, sep))

  mar_attrs = lambda i: (
      nodes[attach[i]][0] if attach.get(i) is not None
      else maximal[mar_parent[i]]
  )
  mar_maps = [
      _index_map(domain, mar_attrs(i), cl) for i, cl in enumerate(loss_cliques)
  ]
  mar_bk = [
      _block(domain, mar_attrs(i), cl) for i, cl in enumerate(loss_cliques)
  ]
  mar_lead = [
      _lead(domain, mar_attrs(i), cl) for i, cl in enumerate(loss_cliques)
  ]

  m_l = np.array(
      [_parent(cl, loss_cliques, domain) for cl in measurement_cliques],
      dtype=np.uint32,
  )
  m_maps = [
      _index_map(domain, loss_cliques[m_l[k]], cl)
      for k, cl in enumerate(measurement_cliques)
  ]
  m_bk = [
      _block(domain, loss_cliques[m_l[k]], cl)
      for k, cl in enumerate(measurement_cliques)
  ]
  total_clique = min(
      range(len(loss_cliques)), key=lambda i: domain.size(loss_cliques[i])
  )

  cat = lambda xs, dt: (
      np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
  )
  arrays = dict(
      l_off=_offsets([domain.size(c) for c in loss_cliques]),
      j_off=_offsets([domain.size(c) for c in maximal]),
      exp_parent=exp_parent,
      exp_map_off=_offsets([m.size for m in exp_maps]),
      exp_map=cat(exp_maps, np.uint32),
      exp_bk=np.array(exp_bk, dtype=np.uint64),
      e_src=np.array(e_src, dtype=np.uint32),
      e_dst=np.array(e_dst, dtype=np.uint32),
      e_rev=np.array(e_rev, dtype=np.uint64),
      sep_off=_offsets(sep_sizes),
      src_map_off=_offsets([m.size for m in src_maps]),
      src_map=cat(src_maps, np.uint32),
      dst_map_off=_offsets([m.size for m in dst_maps]),
      dst_map=cat(dst_maps, np.uint32),
      src_bk=np.array(src_bk, dtype=np.uint64),
      dst_bk=np.array(dst_bk, dtype=np.uint64),
      mar_parent=mar_parent,
      mar_map_off=_offsets([m.size for m in mar_maps]),
      mar_map=cat(mar_maps, np.uint32),
      mar_bk=np.array(mar_bk, dtype=np.uint64),
      m_l=m_l,
      m_off=_offsets([domain.size(c) for c in measurement_cliques]),
      m_map_off=_offsets([m.size for m in m_maps]),
      m_map=cat(m_maps, np.uint32),
      m_bk=np.array(m_bk, dtype=np.uint64),
      h_parent=np.array(
          [n[1] if n[1] is not None else 2**64 - 1 for n in nodes],
          dtype=np.uint64,
      ),
      h_j=np.array([n[2] for n in nodes], dtype=np.uint32),
      h_off=_offsets([domain.size(n[0]) for n in nodes]),
      h_map_off=_offsets([m.size for m in h_maps]),
      h_map=cat(h_maps, np.uint32),
      h_bk=np.array(h_bk, dtype=np.uint64),
      h_lead=np.array(h_lead, dtype=np.uint64),
      mar_lead=np.array(mar_lead, dtype=np.uint64),
      src_lead=np.array(src_lead, dtype=np.uint64),
      l_attach=np.array(
          [attach[i] if attach.get(i) is not None else 2**64 - 1
           for i in range(len(loss_cliques))],
          dtype=np.uint64,
      ),
  )
  return _Structure(
      arrays=arrays,
      n_l=len(loss_cliques),
      n_j=len(maximal),
      n_e=len(order),
      n_m=len(measurement_cliques),
      total_clique=total_clique,
      loss_cliques=loss_cliques,
      cells=sum(domain.size(c) for c in loss_cliques),
      maximal=tuple(maximal),
      tree_edges=tuple(jtree.edges()),
  )


class _Plan:
  """A ``_PlanC`` plus the arrays it points into (kept alive here)."""

  def __init__(self, domain, loss_cliques, measurements):
    self.structure = _structure(
        domain, tuple(loss_cliques), tuple(m.clique for m in measurements)
    )
    s = self.structure
    self.y = np.ascontiguousarray(
        np.concatenate(
            [np.asarray(m.noisy_measurement, dtype=np.float64).reshape(-1)
             for m in measurements]
        )
        if measurements
        else np.zeros(0)
    )
    self.sd = np.array([float(m.stddev) for m in measurements], dtype=np.float64)
    a = s.arrays
    ptr = lambda x, t: x.ctypes.data_as(_P(t))
    u64, u32, dbl = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_double
    self.c = _PlanC(
        s.n_l, ptr(a["l_off"], u64), s.n_j, ptr(a["j_off"], u64),
        ptr(a["exp_parent"], u32), ptr(a["exp_map_off"], u64),
        ptr(a["exp_map"], u32), ptr(a["exp_bk"], u64), s.n_e,
        ptr(a["e_src"], u32), ptr(a["e_dst"], u32), ptr(a["e_rev"], u64),
        ptr(a["sep_off"], u64), ptr(a["src_map_off"], u64),
        ptr(a["src_map"], u32), ptr(a["dst_map_off"], u64),
        ptr(a["dst_map"], u32), ptr(a["src_bk"], u64), ptr(a["dst_bk"], u64),
        ptr(a["mar_parent"], u32), ptr(a["mar_map_off"], u64),
        ptr(a["mar_map"], u32), ptr(a["mar_bk"], u64), s.total_clique,
        s.n_m, ptr(a["m_l"], u32), ptr(a["m_off"], u64), ptr(self.y, dbl),
        ptr(self.sd, dbl), ptr(a["m_map_off"], u64), ptr(a["m_map"], u32),
        ptr(a["m_bk"], u64), len(a["h_j"]), ptr(a["h_parent"], u64),
        ptr(a["h_j"], u32), ptr(a["h_off"], u64), ptr(a["h_map_off"], u64),
        ptr(a["h_map"], u32), ptr(a["h_bk"], u64), ptr(a["l_attach"], u64),
        ptr(a["mar_lead"], u64), ptr(a["h_lead"], u64),
        ptr(a["src_lead"], u64),
    )


def _flatten(cv: CliqueVector, cliques) -> np.ndarray:
  return np.ascontiguousarray(
      np.concatenate(
          [np.asarray(cv[cl].values, dtype=np.float64).reshape(-1)
           for cl in cliques]
      )
  )


def _unflatten(flat: np.ndarray, domain: Domain, cliques) -> CliqueVector:
  tables, start = {}, 0
  for cl in cliques:
    dom = domain.project(cl)
    tables[cl] = Factor(dom, jnp.asarray(flat[start : start + dom.size()].reshape(dom.shape)))
    start += dom.size()
  return CliqueVector(domain, list(cliques), tables)


def oracle(potentials: CliqueVector, total: float = 1.0, measurements=()) -> CliqueVector:
  """Native HUGIN marginals of ``potentials`` (for testing / reuse)."""
  lib = _load()
  if lib is None:
    raise RuntimeError("mbi native library is not available")
  cliques = list(potentials.cliques)
  plan = _Plan(potentials.domain, cliques, list(measurements))
  theta = _flatten(potentials, cliques)
  out = np.empty_like(theta)
  lib.mbi_oracle(
      ctypes.byref(plan.c), theta.ctypes.data_as(_P(ctypes.c_double)),
      float(total), 0, out.ctypes.data_as(_P(ctypes.c_double)),
  )
  return _unflatten(out, potentials.domain, cliques)


def expand(potentials: CliqueVector, cliques: Sequence[Clique]) -> CliqueVector:
  """NumPy equivalent of ``CliqueVector.expand`` (no per-op JAX dispatch).

  Each existing table is added into the smallest new clique containing it
  (``reverse_clique_mapping``), exactly as ``CliqueVector.expand`` does.
  """
  tables = _expand_numpy(potentials, cliques)
  domain = potentials.domain
  return CliqueVector(
      domain,
      list(cliques),
      {cl: Factor(domain.project(cl), jnp.asarray(v)) for cl, v in tables.items()},
  )


def _expand_numpy(potentials: CliqueVector, cliques: Sequence[Clique]) -> dict:
  domain = potentials.domain
  mapping = reverse_clique_mapping(cliques, potentials.cliques, domain=domain)
  out = {}
  for cl in cliques:
    table = np.zeros(domain.project(cl).shape)
    for old in mapping[cl]:
      table = table + _align(
          np.asarray(potentials[old].values, dtype=np.float64), old, cl
      )
    out[cl] = table
  return out


def hypothetical_model_sizes(
    domain: Domain, cliques: Sequence[Clique], candidates: Sequence[Clique]
) -> list[float]:
  """``junction_tree.hypothetical_model_size(domain, cliques + [cl])`` (in MB)
  for every candidate ``cl``, computed natively in one call.

  Falls back to the Python implementation for any candidate whose clique
  sizes overflow 128-bit integers.
  """
  lib = _load()
  if lib is None:
    raise RuntimeError("mbi native library is not available")
  index = {a: i for i, a in enumerate(domain.attributes)}
  sizes = np.array(domain.shape, dtype=np.uint64)

  def pack(cls):
    off = _offsets([len(c) for c in cls])
    flat = np.array([index[a] for c in cls for a in c], dtype=np.uint32)
    return off, flat

  base_off, base = pack(cliques)
  cand_off, cand = pack(candidates)
  out = np.empty(len(candidates))
  u64, u32 = ctypes.c_uint64, ctypes.c_uint32
  ptr = lambda x, t: x.ctypes.data_as(_P(t))
  lib.mbi_model_sizes(
      len(index), ptr(sizes, u64), len(cliques), ptr(base_off, u64),
      ptr(base, u32), len(candidates), ptr(cand_off, u64), ptr(cand, u32),
      ptr(out, ctypes.c_double),
  )
  result = []
  for cl, cells in zip(candidates, out):
    if np.isnan(cells):
      result.append(
          junction_tree.hypothetical_model_size(domain, list(cliques) + [cl])
      )
    else:
      result.append(float(cells) * 8 / 2**20)
  return result


def junction_tree_beliefs(
    potentials: CliqueVector, total: float = 1.0
) -> tuple[dict, tuple]:
  """Calibrated junction-tree beliefs of ``potentials``.

  Returns ``(beliefs, edges)``: a dict mapping each maximal clique of the
  junction tree (the same tree ``message_passing_hugin`` builds) to its
  marginal as a NumPy array summing to ``total``, and the tree's edges.
  """
  lib = _load()
  if lib is None:
    raise RuntimeError("mbi native library is not available")
  cliques = list(potentials.cliques)
  plan = _Plan(potentials.domain, cliques, [])
  s = plan.structure
  theta = _flatten(potentials, cliques)
  out = np.empty(int(s.arrays["j_off"][-1]))
  lib.mbi_beliefs(
      ctypes.byref(plan.c), theta.ctypes.data_as(_P(ctypes.c_double)),
      float(total), out.ctypes.data_as(_P(ctypes.c_double)),
  )
  beliefs, start = {}, 0
  for cl in s.maximal:
    shape = tuple(potentials.domain[a] for a in cl)
    n = math.prod(shape)
    beliefs[cl] = out[start : start + n].reshape(shape)
    start += n
  return beliefs, s.tree_edges


class _Table:
  """Minimal stand-in for a ``Factor`` holding a NumPy array."""

  def __init__(self, values):
    self.values = values

  def datavector(self, flatten: bool = True):
    return self.values.reshape(-1) if flatten else self.values


class NumpyMarginals:
  """Marginal tables backed by NumPy, answering ``project`` like a
  ``CliqueVector`` (from the smallest containing table) without JAX ops."""

  def __init__(self, tables: dict):
    self.tables = tables

  @classmethod
  def from_potentials(cls, potentials: CliqueVector, total: float):
    """Calibrated junction-tree marginals of ``potentials`` (native)."""
    return cls(junction_tree_beliefs(potentials, total)[0])

  def project(self, attrs) -> _Table:
    attrs = tuple(attrs)
    if attrs in self.tables:
      return _Table(self.tables[attrs])
    parent = min(
        (c for c in self.tables if set(attrs) <= set(c)),
        key=lambda c: self.tables[c].size,
    )
    return _Table(_marginalize(self.tables[parent], parent, attrs))


_TAG = object()  # extra leading axis carrying the source attribute's value


def _marginalize(values, attrs, keep):
  """Sum out the axes not in ``keep`` and order the result as ``keep``."""
  axes = tuple(i for i, a in enumerate(attrs) if a not in keep)
  values = values.sum(axis=axes) if axes else values
  rest = [a for a in attrs if a in keep]
  return np.transpose(values, [rest.index(a) for a in keep])


def _align(values, attrs, target):
  """Transpose/reshape ``values`` over ``attrs`` to broadcast over ``target``."""
  present = [a for a in target if a in attrs]
  values = np.transpose(values, [list(attrs).index(a) for a in present])
  shape, it = [], iter(values.shape)
  for a in target:
    shape.append(next(it) if a in attrs else 1)
  return values.reshape(shape)


def _source_sweep(domain, beliefs, adjacency, source):
  """``P(source, x_K)`` for every junction-tree clique ``K``.

  Starts from a clique containing ``source`` and propagates along the tree:
  for an edge ``K -> K'`` with separator ``S``,
  ``P(a, x_K') = sum_{x_S} P(a, x_S) P(x_K') / P(x_S)``, which is exact
  because the junction tree makes ``a`` independent of ``K' \ S`` given
  ``S``.
  """
  root = next(k for k in beliefs if source in k)
  n = domain[source]
  ax = root.index(source)
  eye = np.eye(n).reshape([n] + [n if i == ax else 1 for i in range(len(root))])
  joint = {root: eye * beliefs[root][None]}
  stack, seen = [root], {root}
  while stack:
    k = stack.pop()
    for k2 in adjacency[k]:
      if k2 in seen:
        continue
      seen.add(k2)
      stack.append(k2)
      sep = tuple(a for a in k if a in k2)
      q_sep = _marginalize(joint[k], (_TAG,) + k, (_TAG,) + sep)
      p_sep = _align(_marginalize(beliefs[k2], k2, sep), sep, k2)
      ratio = np.divide(
          beliefs[k2], p_sep, out=np.zeros_like(beliefs[k2]), where=p_sep > 0
      )
      joint[k2] = _align(q_sep, (_TAG,) + sep, (_TAG,) + k2) * ratio[None]
  return joint


def out_of_model_marginals(
    potentials: CliqueVector, total: float, cliques: Sequence[Clique]
) -> dict:
  """Marginals (flattened, in each clique's attribute order) of ``cliques``.

  Equivalent to ``variable_elimination(potentials, cl, total)`` for each
  clique, but computes all pairs ``(a, b)`` from one calibrated junction tree
  with one sweep per distinct source attribute ``a``, instead of a separate
  elimination per pair. Other clique sizes fall back to
  ``marginal_oracles.variable_elimination_numpy``.
  """
  from . import marginal_oracles  # pylint: disable=import-outside-toplevel

  domain = potentials.domain
  result = {}
  pairs = [cl for cl in cliques if len(cl) == 2]
  if pairs:
    beliefs, edges = junction_tree_beliefs(potentials, total)
    adjacency = {k: [] for k in beliefs}
    for u, v in edges:
      adjacency[u].append(v)
      adjacency[v].append(u)
    covered = set().union(*beliefs) if beliefs else set()
    by_source = {}
    for cl in pairs:
      if cl[0] in covered and cl[1] in covered:
        by_source.setdefault(cl[0], []).append(cl)
    for source, group in by_source.items():
      joint = _source_sweep(domain, beliefs, adjacency, source)
      for cl in group:
        b = cl[1]
        k = min(
            (k for k in joint if b in k), key=lambda k: joint[k].size
        )
        result[cl] = _marginalize(joint[k], (_TAG,) + k, (_TAG, b)).reshape(-1)
  for cl in cliques:
    if cl not in result:
      result[cl] = np.asarray(
          marginal_oracles.variable_elimination_numpy(
              potentials, cl, total
          ).values,
          dtype=np.float64,
      ).reshape(-1)
  return result


@dataclasses.dataclass(frozen=True)
class NativeMirrorDescent(estimation.MirrorDescent):
  """``MirrorDescent`` with the optimization loop running in native code.

  Same interface and semantics as :class:`mbi.estimation.MirrorDescent`; see
  the module docstring for when it falls back to the JAX implementation.
  """

  def supports(
      self,
      loss_fn: MarginalLossFn | Sequence[LinearMeasurement],
      constraints: Sequence[Constraint] = (),
      callback_fn: Callable | None = None,
  ) -> bool:
    """Whether ``estimate`` will run natively for these arguments."""
    if not is_available() or constraints or callback_fn is not None:
      return False
    if self.marginal_oracle is not None or self.mesh is not None:
      return False
    if isinstance(loss_fn, MarginalLossFn):
      if loss_fn.loss_fn is not marginal_loss._l2_loss:  # pylint: disable=protected-access
        return False
      measurements = loss_fn.data
    else:
      measurements = loss_fn
    return all(
        isinstance(m, LinearMeasurement) and isinstance(m.query, DatavectorQuery)
        for m in measurements
    )

  def estimate(
      self,
      domain: Domain,
      loss_fn: MarginalLossFn | Sequence[LinearMeasurement],
      known_total: float | None = None,
      constraints: Sequence[Constraint] = (),
      iters: int = 1000,
      callback_fn: Callable | None = None,
      warm_start=None,
      tol: float | None = 1e-3,
  ) -> MarkovRandomField:
    if not self.supports(loss_fn, constraints, callback_fn):
      return super().estimate(
          domain, loss_fn, known_total, constraints, iters, callback_fn,
          warm_start, tol,
      )
    if isinstance(loss_fn, MarginalLossFn):
      measurements = list(loss_fn.data)
    else:
      measurements = list(loss_fn)
      if known_total is None:
        known_total = estimation.minimum_variance_unbiased_total(measurements)
      loss_fn = marginal_loss.from_linear_measurements(measurements, domain)
    if known_total is None:
      known_total = 1.0
    if not loss_fn.cliques:
      return super().estimate(domain, loss_fn, known_total, iters=iters)

    cliques = list(loss_fn.cliques)
    if warm_start is None:
      theta = np.zeros(sum(domain.size(cl) for cl in cliques))
    else:
      w_pot = (
          warm_start
          if isinstance(warm_start, CliqueVector)
          else getattr(warm_start, "potentials")
      )
      tables = _expand_numpy(w_pot, cliques)
      theta = np.ascontiguousarray(
          np.concatenate([tables[cl].reshape(-1) for cl in cliques])
      )
    plan = _Plan(domain, cliques, measurements)
    lipschitz = float(loss_fn.lipschitz)
    alpha0 = 2.0 / (lipschitz * known_total)
    mu = np.empty_like(theta)
    info = np.zeros(4)
    dbl = ctypes.c_double
    status = _load().mbi_mirror_descent(
        ctypes.byref(plan.c),
        theta.ctypes.data_as(_P(dbl)),
        mu.ctypes.data_as(_P(dbl)),
        float(known_total),
        lipschitz,
        alpha0,
        float(self.stepsize) if self.stepsize is not None else 0.0,
        int(math.ceil(iters / estimation.CALLBACK_EVERY))
        * estimation.CALLBACK_EVERY,
        estimation.CALLBACK_EVERY,
        float(tol) if tol is not None else -1.0,
        float(self.loss_rtol) if self.loss_rtol is not None else -1.0,
        info.ctypes.data_as(_P(dbl)),
    )
    if status != 0:
      raise RuntimeError(f"mbi native library failed with status {status}")
    if os.environ.get("MBI_NATIVE_DEBUG"):
      print(f"[native] cells={int(plan.structure.arrays['j_off'][-1])} "
            f"steps={int(info[0])} gap={info[1]:.2e} alpha0={alpha0:.3e} "
            f"alpha_final={info[2]:.3e} loss={info[3]:.6e}", file=sys.stderr)
    return MarkovRandomField(
        potentials=_unflatten(theta, domain, cliques),
        marginals=_unflatten(mu, domain, cliques),
        total=known_total,
    )


if __name__ == "__main__":
  if sys.argv[1:] == ["build"]:
    print(f"Installed {build()}")
  else:
    print(f"usage: python -m mbi.native build\nlibrary: {library_path()}")


def _native_supports(est, loss_fn, constraints, callback_fn) -> bool:
  """Shared checks for running an estimator natively."""
  if not is_available() or constraints or callback_fn is not None:
    return False
  if est.marginal_oracle is not None or getattr(est, "mesh", None) is not None:
    return False
  if isinstance(loss_fn, MarginalLossFn):
    if loss_fn.loss_fn is not marginal_loss._l2_loss:  # pylint: disable=protected-access
      return False
    measurements = loss_fn.data
  else:
    measurements = loss_fn
  return all(
      isinstance(m, LinearMeasurement) and isinstance(m.query, DatavectorQuery)
      for m in measurements
  )


def _prepare(domain, loss_fn, known_total, warm_start):
  """Measurements, loss_fn, total and initial (flat) potentials."""
  if isinstance(loss_fn, MarginalLossFn):
    measurements = list(loss_fn.data)
  else:
    measurements = list(loss_fn)
    if known_total is None:
      known_total = estimation.minimum_variance_unbiased_total(measurements)
    loss_fn = marginal_loss.from_linear_measurements(measurements, domain)
  if known_total is None:
    known_total = 1.0
  cliques = list(loss_fn.cliques)
  if warm_start is None:
    theta = np.zeros(sum(domain.size(cl) for cl in cliques))
  else:
    w_pot = (
        warm_start
        if isinstance(warm_start, CliqueVector)
        else getattr(warm_start, "potentials")
    )
    tables = _expand_numpy(w_pot, cliques)
    theta = np.ascontiguousarray(
        np.concatenate([tables[cl].reshape(-1) for cl in cliques])
    )
  return measurements, loss_fn, float(known_total), cliques, theta


@dataclasses.dataclass(frozen=True)
class NativeAcceleratedMethod(estimation.UniversalAcceleratedMethod):
  """``UniversalAcceleratedMethod`` with the optimization loop in native code.

  Runs the same accelerated method (Nesterov's universal method with
  backtracking line search) in float64 Rust, with no JIT compilation. On
  large, badly conditioned problems, such as AIM at high row counts, it
  reaches a given loss in roughly 10x fewer iterations than
  ``MirrorDescent``.

  Difference from the JAX version in how the final model is produced: the
  accelerated iterate ``x`` is an average of marginals, not itself the
  marginals of a graphical model, so the JAX version finishes with an L-BFGS
  fit (up to 25,000 iterations) of a model to ``x``. This class instead
  returns the model given by the method's dual potentials ``u``, whose
  marginals are the mirror iterate ``z`` (exactly a graphical model, and
  converging to the same optimum). Falls back to the JAX implementation in
  the same cases as ``NativeMirrorDescent``.
  """

  def supports(self, loss_fn, constraints=(), callback_fn=None) -> bool:
    return _native_supports(self, loss_fn, constraints, callback_fn)

  def estimate(
      self,
      domain: Domain,
      loss_fn,
      known_total: float | None = None,
      constraints: Sequence[Constraint] = (),
      iters: int = 1000,
      callback_fn: Callable | None = None,
      warm_start=None,
      tol: float | None = 1e-3,
  ) -> MarkovRandomField:
    if not self.supports(loss_fn, constraints, callback_fn) or not self.linesearch:
      return super().estimate(
          domain, loss_fn, known_total, constraints, iters, callback_fn,
          warm_start, tol,
      )
    measurements, loss_fn, known_total, cliques, theta = _prepare(
        domain, loss_fn, known_total, warm_start
    )
    if not cliques:
      return super().estimate(domain, loss_fn, known_total, iters=iters)
    theta, mu, _, _ = self._run_native(
        domain, measurements, loss_fn, known_total, cliques, theta, iters, tol
    )
    return MarkovRandomField(
        potentials=_unflatten(theta, domain, cliques),
        marginals=_unflatten(mu, domain, cliques),
        total=known_total,
    )

  def _run_native(
      self, domain, measurements, loss_fn, known_total, cliques, theta, iters,
      tol,
  ):
    """Returns (potentials u, total*z, total*x, info) as flat arrays."""
    plan = _Plan(domain, cliques, measurements)
    mu = np.empty_like(theta)
    x = np.empty_like(theta)
    info = np.zeros(4)
    dbl = ctypes.c_double
    block = estimation.CALLBACK_EVERY
    status = _load().mbi_accelerated(
        ctypes.byref(plan.c),
        theta.ctypes.data_as(_P(dbl)),
        mu.ctypes.data_as(_P(dbl)),
        x.ctypes.data_as(_P(dbl)),
        known_total,
        float(loss_fn.lipschitz),
        int(math.ceil(iters / block)) * block,
        block,
        float(tol) if tol is not None else -1.0,
        int(self.max_iter_search),
        float(self.target_acc),
        float(self.loss_rtol) if self.loss_rtol is not None else -1.0,
        info.ctypes.data_as(_P(dbl)),
    )
    if status != 0:
      raise RuntimeError(f"mbi native library failed with status {status}")
    if os.environ.get("MBI_NATIVE_DEBUG"):
      print(f"[native-accel] cells={int(plan.structure.arrays['j_off'][-1])} "
            f"steps={int(info[0])} gap={info[1]:.2e} loss_x={info[3]:.6e}",
            file=sys.stderr)
    return theta, mu, x, info


def marginal_counts(data, cliques: Sequence[Clique]) -> dict:
  """``data.project(cl).datavector()`` for every clique, in one native call.

  Equivalent to projecting the dataset onto each clique (counts, or sums of
  the dataset's weights), but computes all tables in a single multi-threaded
  pass over the rows, without building int64 index arrays per clique. Counts
  are exact; with non-unit weights, sums may differ in the last bits.

  Returns a dict mapping each clique to its flat float64 table (row-major in
  the clique's attribute order).
  """
  lib = _load()
  if lib is None:
    raise RuntimeError("mbi native library is not available")
  cliques = [tuple(cl) for cl in cliques]
  domain = data.domain
  used = list(dict.fromkeys(a for cl in cliques for a in cl))
  index = {a: i for i, a in enumerate(used)}
  columns = [np.ascontiguousarray(data.data[a], dtype=np.uint32) for a in used]
  n_rows = data.records
  col_ptrs = (ctypes.c_void_p * max(len(columns), 1))(
      *[c.ctypes.data for c in columns]
  )
  off = _offsets([len(cl) for cl in cliques])
  attrs = np.array([index[a] for cl in cliques for a in cl], dtype=np.uint32)
  sizes = np.array([domain[a] for cl in cliques for a in cl], dtype=np.uint32)
  out_off = _offsets([domain.size(cl) for cl in cliques])
  out = np.empty(int(out_off[-1]))
  weights = None
  if data.weights is not None and not np.all(data.weights == 1):
    weights = np.ascontiguousarray(data.weights, dtype=np.float64)
  ptr = lambda x, t: x.ctypes.data_as(_P(t))
  u64, u32, dbl = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_double
  lib.mbi_count(
      n_rows, col_ptrs, len(cliques), ptr(off, u64), ptr(attrs, u32),
      ptr(sizes, u32), ptr(out_off, u64),
      ptr(weights, dbl) if weights is not None else None,
      ptr(out, dbl),
  )
  return {
      cl: out[int(out_off[k]) : int(out_off[k + 1])]
      for k, cl in enumerate(cliques)
  }


def sample_column(parents, sizes, cond_cdfs, method: str = "round", seed=None):
  """Sample a synthetic column given its parent columns (native).

  Native equivalent of one step of ``MarkovRandomField.synthetic_data``:
  rows are grouped by parent configuration; with ``method="round"`` each
  group's values follow its conditional distribution up to rounding
  (randomly permuted ranks plus uniform noise, then the conditional CDF),
  with ``"sample"`` values are drawn i.i.d. ``cond_cdfs`` has shape
  ``sizes + (k,)``. ``seed`` defaults to a draw from ``np.random``, so
  ``np.random.seed`` makes results reproducible.
  """
  lib = _load()
  if lib is None:
    raise RuntimeError("mbi native library is not available")
  cols = [np.ascontiguousarray(p, dtype=np.uint32) for p in parents]
  n = cols[0].size
  k = cond_cdfs.shape[-1]
  cdf = np.ascontiguousarray(cond_cdfs, dtype=np.float64).reshape(-1, k)
  if seed is None:
    seed = int(np.random.randint(0, 2**63 - 1, dtype=np.int64))
  out = np.empty(n, dtype=np.uint32)
  ptrs = (ctypes.c_void_p * len(cols))(*[c.ctypes.data for c in cols])
  size_arr = np.array(sizes, dtype=np.uint32)
  u32 = ctypes.c_uint32
  lib.mbi_sample_column(
      n, len(cols), ptrs, size_arr.ctypes.data_as(_P(u32)),
      cdf.ctypes.data_as(_P(ctypes.c_double)), k,
      0 if method == "round" else 1, seed, out.ctypes.data_as(_P(u32)),
  )
  return out
