import mbi

"""Implementation of AIM: An Adaptive and Iterative Mechanism for DP Synthetic Data.

Note that with the default settings, AIM can take many hours to run.  You can configure
the runtime /utility tradeoff via the max_model_size flag.  We recommend setting it to 1.0
for debugging, but keeping the default value of 80 for any official comparisons to this mechanism.

Note that we assume in this file that the data has been appropriately preprocessed so that there are no large-cardinality categorical attributes.  If there are, we recommend using something like "compress_domain" from mst.py.  Since our paper evaluated already-preprocessed datastes, we did not implement that here for simplicity.

Performance notes: AIM fits a differently-structured model every round, so with
default JAX settings almost all of its runtime is spent compiling XLA programs
(see benchmarks/aim/REPORT.md).  This implementation avoids most of that by
(1) scoring candidates with NumPy instead of JIT-compiled JAX code,
(2) merging repeated measurements of the same clique so the fitted program's
structure (and compile cache entry) is reused, (3) fitting with
``MirrorDescent(fuse_steps=False)``, which compiles ~5x faster, optionally with
``fast_compile`` for small models, and (4) clearing JAX's caches when the process
nears its memory-map limit, so long runs don't crash.  None of these change
the selected cliques or the fitted model beyond floating-point rounding.
"""

import numpy as np
import itertools
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import jax
from mbi import (
    Dataset,
    Domain,
    estimation,
    junction_tree,
    marginal_oracles,
    LinearMeasurement,
)
from mbi import native
from mbi.marginal_loss import DatavectorQuery
from mechanism import Mechanism
from collections import defaultdict
from scipy.optimize import bisect
import json
import pandas as pd
from mbi import Factor
import argparse


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def compile_workload(workload):
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit, size_cache=None):
    """Keep candidates that fit under ``size_limit`` or are already supported.

    ``size_cache`` (optional dict) memoizes junction-tree sizes across calls;
    AIM often revisits the same model structure (e.g. when it re-measures a
    clique), so the same sizes get recomputed otherwise.  When the native
    library is available, the sizes of all remaining candidates are computed
    in one native call.
    """
    free_cliques = set(downward_closure(model.cliques))
    model_cliques = tuple(model.cliques)
    if size_cache is None:
        size_cache = {}
    # Candidates already supported by the model are kept regardless of size,
    # so their (expensive) junction-tree sizes are never needed.
    todo = [
        cl
        for cl in candidates
        if cl not in free_cliques and (model_cliques, cl) not in size_cache
    ]
    if todo:
        if native.is_available():
            sizes = native.hypothetical_model_sizes(
                model.domain, list(model_cliques), todo
            )
        else:
            sizes = [
                junction_tree.hypothetical_model_size(
                    model.domain, list(model_cliques) + [cl]
                )
                for cl in todo
            ]
        for cl, size in zip(todo, sizes):
            size_cache[(model_cliques, cl)] = size
    ans = {}
    for cl in candidates:
        if cl in free_cliques or size_cache[(model_cliques, cl)] <= size_limit:
            ans[cl] = candidates[cl]
    return ans


def merge_measurements(measurements):
    """Combine identity measurements of the same clique into one.

    Measurements ``y_i ~ N(x, s_i^2)`` of the same marginal are replaced by their
    precision-weighted mean ``y = (sum_i y_i / s_i^2) / P`` with standard
    deviation ``1 / sqrt(P)``, where ``P = sum_i 1 / s_i^2``.  The squared-error
    loss changes only by an additive constant, so its gradient, the estimated
    total and the Lipschitz constant (and hence the fitted model) are unchanged.
    Keeping one term per clique means re-measuring a clique doesn't change the
    structure of the compiled estimation program, so it is reused.

    Measurements with non-identity queries are passed through unchanged.
    """
    groups = {}
    order = []
    for m in measurements:
        if isinstance(m.query, DatavectorQuery):
            key = (m.clique, m.query)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(m)
        else:
            order.append(m)
    merged = []
    for key in order:
        if isinstance(key, LinearMeasurement):
            merged.append(key)
            continue
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        precisions = [1.0 / float(m.stddev) ** 2 for m in group]
        total_precision = sum(precisions)
        y = sum(
            p * np.asarray(m.noisy_measurement, dtype=np.float64)
            for p, m in zip(precisions, group)
        ) / total_precision
        merged.append(
            LinearMeasurement(
                y, key[0], stddev=1.0 / np.sqrt(total_precision), query=key[1]
            )
        )
    return merged


def _memory_map_usage():
    """Fraction of the per-process memory-map limit in use (Linux), or None.

    Every compiled XLA program holds memory mappings until JAX's caches are
    cleared; a process that exceeds ``vm.max_map_count`` fails with "LLVM
    compilation error: Cannot allocate memory".
    """
    try:
        with open("/proc/self/maps") as f:
            used = sum(1 for _ in f)
        with open("/proc/sys/vm/max_map_count") as f:
            limit = int(f.read())
    except (OSError, ValueError):
        return None
    return used / limit


class _NumpyMarginals:
    """Answers marginal queries on a fitted model with NumPy (no JIT)."""

    def __init__(self, model):
        self.model = model
        self.tables = {}
        self.out_of_model = {}

    def prefetch(self, cliques):
        """Compute the out-of-model marginals among ``cliques`` in one batch.

        Uses ``mbi.native.out_of_model_marginals`` (one calibrated junction
        tree, one sweep per source attribute) instead of a separate variable
        elimination per clique.
        """
        todo = [
            cl
            for cl in cliques
            if cl not in self.out_of_model
            and not self.model.marginals.supports(cl)
        ]
        if todo:
            self.out_of_model.update(
                native.out_of_model_marginals(
                    self.model.potentials, float(self.model.total), todo
                )
            )

    def datavector(self, cl):
        if cl in self.out_of_model:
            return self.out_of_model[cl]
        model = self.model
        if model.marginals.supports(cl):
            parent = model.marginals.parent(cl)
            if parent not in self.tables:
                self.tables[parent] = np.asarray(
                    model.marginals[parent].values, dtype=np.float64
                )
            table = self.tables[parent]
            axes = tuple(i for i, a in enumerate(parent) if a not in cl)
            reduced = table.sum(axis=axes) if axes else table
            kept = [a for a in parent if a in cl]
            reduced = np.transpose(reduced, [kept.index(a) for a in cl])
            return reduced.reshape(-1)
        return np.asarray(
            marginal_oracles.variable_elimination_numpy(
                model.potentials, cl, float(model.total)
            ).values,
            dtype=np.float64,
        ).reshape(-1)


class AIM(Mechanism):

    def __init__(
        self,
        epsilon,
        delta,
        prng=None,
        rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
        fuse_steps=False,
        fast_compile_max_cells=100_000,
        clear_caches_every=25,
        clear_caches_map_fraction=0.5,
        parallel_compile=None,
        intermediate_iters=None,
        use_native=None,
        estimator="mirror_descent",
        loss_rtol=None,
    ):
        """
        :param fuse_steps: passed to ``MirrorDescent``.  ``False`` (default)
            compiles much faster, which is what dominates AIM's runtime.
        :param fast_compile_max_cells: fit with ``MirrorDescent(fast_compile=True)``
            while the model's junction tree has at most this many cells (where
            compile time dwarfs compute time).  Set to 0 to disable.
        :param clear_caches_every: every this many rounds, call
            ``jax.clear_caches()`` if the process uses more than
            ``clear_caches_map_fraction`` of its memory-map limit (Linux only;
            clearing is slow and forces recompiles, so it is only done when
            needed).  Set to ``None`` to disable.
        :param clear_caches_map_fraction: see ``clear_caches_every``.
        :param parallel_compile: passed to ``MirrorDescent``; compiles the
            independent programs of each fit concurrently.  Defaults to ``True``
            when more than one CPU is available.
        :param intermediate_iters: maximum MirrorDescent iterations for the
            fits in intermediate rounds (the initial and final fits always use
            ``max_iters``).  Those fits only guide which clique to measure next
            and are warm-started from the previous round, so a smaller value
            trades a little selection accuracy for speed.  Defaults to
            ``max_iters`` (no change in behaviour).
        :param use_native: fit models with ``mbi.native.NativeMirrorDescent``
            (the optimization loop in Rust, no JIT compilation).  Defaults to
            ``True`` when the native library is available (build it with
            ``python -m mbi.native build``); otherwise the JAX
            ``MirrorDescent`` is used with the options above.
        :param estimator: ``"mirror_descent"`` (default, as in the AIM paper)
            or ``"accelerated"`` (``UniversalAcceleratedMethod``; with the
            native library, ``mbi.native.NativeAcceleratedMethod``).  The
            accelerated method fits the same models but converges much faster
            on large datasets, where mirror descent typically stops at
            ``max_iters`` far from convergence; see benchmarks/aim/REPORT.md.
        :param loss_rtol: optional early stopping for each fit: stop when a
            block of 50 iterations improves the loss by at most this fraction
            (see ``Estimator.loss_rtol``).  ``None`` (default) disables it.
        """
        super(AIM, self).__init__(
            epsilon, delta, bounded=False, prng=prng or np.random
        )
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.fuse_steps = fuse_steps
        self.fast_compile_max_cells = fast_compile_max_cells
        self.clear_caches_every = clear_caches_every
        self.clear_caches_map_fraction = clear_caches_map_fraction
        if parallel_compile is None:
            if hasattr(_os, "sched_getaffinity"):
                ncpu = len(_os.sched_getaffinity(0))
            else:
                ncpu = _os.cpu_count() or 1
            parallel_compile = ncpu > 1
        self.parallel_compile = parallel_compile
        self.intermediate_iters = intermediate_iters or max_iters
        if use_native is None:
            use_native = native.is_available()
        self.use_native = use_native
        if estimator not in ("mirror_descent", "accelerated"):
            raise ValueError(f"Unknown estimator {estimator!r}")
        self.estimator = estimator
        self.loss_rtol = loss_rtol

    def _estimate(self, domain, measurements, warm_start=None, iters=None):
        cliques = list({m.clique for m in measurements})
        jtree, _ = junction_tree.make_junction_tree(domain, cliques)
        cells = sum(domain.size(cl) for cl in junction_tree.maximal_cliques(jtree))
        if self.estimator == "accelerated":
            cls = (
                native.NativeAcceleratedMethod
                if self.use_native
                else estimation.UniversalAcceleratedMethod
            )
            return cls(loss_rtol=self.loss_rtol).estimate(
                domain,
                merge_measurements(measurements),
                iters=iters or self.max_iters,
                warm_start=warm_start,
            )
        cls = (
            native.NativeMirrorDescent
            if self.use_native
            else estimation.MirrorDescent
        )
        estimator = cls(
            fuse_steps=self.fuse_steps,
            fast_compile=cells <= self.fast_compile_max_cells,
            parallel_compile=self.parallel_compile,
            loss_rtol=self.loss_rtol,
        )
        return estimator.estimate(
            domain,
            merge_measurements(measurements),
            iters=iters or self.max_iters,
            warm_start=warm_start,
        )

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        marginals = _NumpyMarginals(model)
        if self.use_native:
            marginals.prefetch(list(candidates))
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = marginals.datavector(cl)
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload)
        if self.use_native:
            answers = native.marginal_counts(data, list(candidates))
        else:
            answers = {
                cl: np.asarray(data.project(cl).datavector())
                for cl in candidates
            }

        def counts(cl):
            # Every measured clique is a candidate, so reuse its counts
            # instead of re-scanning the data.
            if cl in answers:
                return answers[cl]
            return np.asarray(data.project(cl).datavector())

        size_cache = {}

        if not initial_cliques:
            initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]  # use one-way marginals

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        print("Initial Sigma", sigma)
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in initial_cliques:
            x = counts(cl)
            y = x + self.gaussian_noise(sigma, x.size)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))

        zeros = self.structural_zeros
        # NOTE: Haven't incorproated structural zeros back yet after refactoring
        model = self._estimate(data.domain, measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.clear_caches_every and t % self.clear_caches_every == 0:
                usage = _memory_map_usage()
                if usage is not None and usage > self.clear_caches_map_fraction:
                    jax.clear_caches()
            if self.rho - rho_used < 2 * (
                0.5 / sigma**2 + 1.0 / 8 * epsilon**2
            ):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            print("Budget Used", rho_used, "/", self.rho)
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(
                candidates, model, size_limit, size_cache
            )
            cl = self.worst_approximated(
                small_candidates, answers, model, epsilon, sigma
            )
            print("Measuring Clique", cl)
            n = data.domain.size(cl)
            x = counts(cl)
            y = x + self.gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            z = _NumpyMarginals(model).datavector(cl)

            # Warm start potentials from prior round
            # TODO: check if it helps to call maximal_subsets here
            pcliques = list(set(M.clique for M in measurements))
            if self.use_native:
                potentials = native.expand(model.potentials, pcliques)
            else:
                potentials = model.potentials.expand(pcliques)
            model = self._estimate(
                data.domain,
                measurements,
                warm_start=potentials,
                iters=self.intermediate_iters,
            )
            w = _NumpyMarginals(model).datavector(cl)
            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
                sigma /= 2
                epsilon *= 2

        print("Generating Data...")
        model = self._estimate(
            data.domain, measurements, warm_start=potentials
        )
        synth = model.synthetic_data(
            rows=num_synth_rows, use_native=self.use_native
        )

        return model, synth


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["noise"] = "laplace"
    params["max_model_size"] = 80
    params["max_iters"] = 1000
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000

    return params


if __name__ == "__main__":

    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter
    )
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--max_model_size",
        type=float,
        help="maximum size (in megabytes) of model",
    )
    parser.add_argument(
        "--max_iters", type=int, help="maximum number of iterations"
    )
    parser.add_argument(
        "--degree", type=int, help="degree of marginals in workload"
    )
    parser.add_argument(
        "--num_marginals", type=int, help="number of marginals in workload"
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        help="maximum number of cells for marginals in workload",
    )
    parser.add_argument("--save", type=str, help="path to save synthetic data")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    domain = Domain.fromdict(json.load(open(args.domain)))
    data = Dataset(
        pd.read_csv(args.dataset, usecols=domain.attrs).to_dict("list"), domain
    )

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None and args.num_marginals < len(workload):
        workload = [
            workload[i]
            for i in np.random.choice(
                len(workload), args.num_marginals, replace=False
            )
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = AIM(
        args.epsilon,
        args.delta,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    model, synth = mech.run(data, workload)

    if args.save is not None:
        import pandas as pd

        pd.DataFrame(synth.to_dict()).to_csv(args.save, index=False)

    synth_errors = []
    model_errors = []
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        Z = model.project(proj).datavector()
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        synth_errors.append(e)
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Z / Z.sum(), 1)
        model_errors.append(e)
    print("Average Error: ", np.mean(model_errors), np.mean(synth_errors))
