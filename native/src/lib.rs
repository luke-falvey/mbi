//! Native kernels for `mbi`.
//!
//! Implements `MirrorDescent.estimate` (Armijo line search, L2 loss over
//! identity measurements, primal-dual-gap early stopping) on top of HUGIN
//! message passing over a junction tree, exactly mirroring the JAX
//! implementation in `mbi.estimation` / `mbi.marginal_oracles`.
//!
//! The model *structure* (cliques, junction tree, measurement projections) is
//! passed in as flat index arrays built once per fit by Python
//! (`mbi.native`), so nothing is compiled at run time: the JAX version
//! compiles new XLA programs for every new model structure, which dominates
//! AIM's runtime.
//!
//! Every table is a flat, row-major `f64` array. Relationships between tables
//! are encoded as index maps: for a "big" table `B` and a table `S` over a
//! subset of `B`'s attributes, `map[k]` is the flat index into `S` of the cell
//! of `S` that cell `k` of `B` projects onto. Projection is then
//! `S[map[k]] (+)= B[k]` and broadcasting is `B[k] += S[map[k]]`.

use std::slice;

/// Model structure, as passed from Python. All pointers must stay valid for the
/// duration of a call. `*_off` arrays hold `n + 1` offsets into the matching
/// flat data array.
#[repr(C)]
pub struct PlanC {
    /// Loss cliques (the cliques the potentials / marginals live on).
    pub n_l: u64,
    pub l_off: *const u64,
    /// Junction-tree (maximal) cliques.
    pub n_j: u64,
    pub j_off: *const u64,
    /// For each loss clique: the junction-tree clique its potential is added
    /// into, and the map (len = size of that J clique) from J cells to L cells.
    pub exp_parent: *const u32,
    pub exp_map_off: *const u64,
    pub exp_map: *const u32,
    /// Per-map block descriptors (`block << 1 | kept`), one per map; see `Map`.
    pub exp_bk: *const u64,
    /// Directed messages in HUGIN order.
    pub n_e: u64,
    pub e_src: *const u32,
    pub e_dst: *const u32,
    /// Index of the message in the opposite direction.
    pub e_rev: *const u64,
    pub sep_off: *const u64,
    pub src_map_off: *const u64,
    pub src_map: *const u32,
    pub dst_map_off: *const u64,
    pub dst_map: *const u32,
    pub src_bk: *const u64,
    pub dst_bk: *const u64,
    /// For each loss clique: the junction-tree clique its marginal is read
    /// from, and the J -> L map.
    pub mar_parent: *const u32,
    pub mar_map_off: *const u64,
    pub mar_map: *const u32,
    pub mar_bk: *const u64,
    /// Index of the smallest loss clique (used to read the model total).
    pub total_clique: u64,
    /// Identity measurements: loss clique they are read from, offsets into
    /// `y`, noisy values, standard deviations, and L -> measurement maps.
    pub n_m: u64,
    pub m_l: *const u32,
    pub m_off: *const u64,
    pub y: *const f64,
    pub m_sd: *const f64,
    pub m_map_off: *const u64,
    pub m_map: *const u32,
    pub m_bk: *const u64,
}

unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        slice::from_raw_parts(p, n)
    }
}

/// `exp(x)` with ~1 ulp accuracy, written branch-free so loops over it
/// auto-vectorize (the libm `exp` is a scalar call). Cody-Waite range
/// reduction `x = n ln2 + r`, |r| <= ln2/2, then a degree-13 Taylor
/// polynomial for `e^r` and `2^n` built directly in the exponent bits.
#[inline(always)]
fn madd(a: f64, b: f64, c: f64) -> f64 {
    // `mul_add` is a single instruction when compiled with FMA, but a slow
    // software routine otherwise.
    if cfg!(target_feature = "fma") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

#[inline(always)]
fn fast_exp(x: f64) -> f64 {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    // Adding 1.5 * 2^52 rounds to the nearest integer and leaves it in the
    // low mantissa bits.
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    let xc = x.max(-708.0).min(709.0);
    let t = xc * LOG2E + MAGIC;
    let n = t - MAGIC;
    let r = (xc - n * LN2_HI) - n * LN2_LO;
    let mut p = 1.0 / 6_227_020_800.0; // 1/13!
    p = madd(p, r, 1.0 / 479_001_600.0);
    p = madd(p, r, 1.0 / 39_916_800.0);
    p = madd(p, r, 1.0 / 3_628_800.0);
    p = madd(p, r, 1.0 / 362_880.0);
    p = madd(p, r, 1.0 / 40_320.0);
    p = madd(p, r, 1.0 / 5_040.0);
    p = madd(p, r, 1.0 / 720.0);
    p = madd(p, r, 1.0 / 120.0);
    p = madd(p, r, 1.0 / 24.0);
    p = madd(p, r, 1.0 / 6.0);
    p = madd(p, r, 0.5);
    p = madd(p, r, 1.0);
    p = madd(p, r, 1.0);
    // 2^n: the integer n sits in the low bits of t; shifting by 52 drops the
    // magic constant's bits and moves (n + 1023) into the exponent field.
    let scale = f64::from_bits(t.to_bits().wrapping_add(1023) << 52);
    let y = p * scale;
    if x < -708.0 {
        0.0
    } else {
        y
    }
}

/// In-place `exp` over a slice (vectorizes).
#[inline(always)]
fn exp_inplace(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = fast_exp(*x);
    }
}

/// Chunk size for gather -> vectorized exp -> scatter loops.
const CHUNK: usize = 512;

/// Tables with fewer cells than this are processed on one thread (spawning
/// threads costs tens of microseconds).
const PAR_MIN_CELLS: usize = 1 << 20;

/// Worker threads: `MBI_NATIVE_THREADS` if set, else the CPUs this process
/// may run on.
fn n_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MBI_NATIVE_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
            .max(1)
    })
}

/// Cells per part when a large table is processed in parallel.
const PART_CELLS: usize = 1 << 18;

/// Split `0..len` into contiguous parts of about `PART_CELLS` cells whose
/// boundaries are multiples of `block` (so no block straddles two parts).
/// The partition depends only on `len` and `block`, never on the number of
/// threads, so parallel reductions give identical results for any thread
/// count. Tables below `PAR_MIN_CELLS` are a single part.
fn split(len: usize, block: usize) -> Vec<std::ops::Range<usize>> {
    let block = block.max(1);
    let n_blocks = len / block;
    if len < PAR_MIN_CELLS || n_blocks < 2 {
        return vec![0..len];
    }
    let blocks_per_part = (PART_CELLS / block).max(1);
    let parts = (n_blocks + blocks_per_part - 1) / blocks_per_part;
    (0..parts)
        .map(|p| (p * blocks_per_part) * block..((p + 1) * blocks_per_part).min(n_blocks) * block)
        .collect()
}

/// Evaluate `f` on each range (in parallel when there are several) and
/// return the results in range order.
fn par_map<R: Send>(ranges: &[std::ops::Range<usize>], f: impl Fn(std::ops::Range<usize>) -> R + Sync) -> Vec<R> {
    let t = n_threads().min(ranges.len());
    if t <= 1 {
        return ranges.iter().map(|r| f(r.clone())).collect();
    }
    let f = &f;
    let mut out: Vec<(usize, R)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t)
            .map(|w| {
                sc.spawn(move || {
                    (w..ranges.len())
                        .step_by(t)
                        .map(|i| (i, f(ranges[i].clone())))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, r)| r).collect()
}

/// Apply `f` to disjoint mutable sub-slices of `v` given by `ranges`
/// (which must tile `0..v.len()` in order), in parallel.
fn par_chunks_mut(v: &mut [f64], ranges: &[std::ops::Range<usize>], f: impl Fn(&mut [f64], std::ops::Range<usize>) + Sync) {
    let t = n_threads().min(ranges.len());
    if t <= 1 {
        let mut rest = v;
        let mut offset = 0;
        for r in ranges {
            let (head, tail) = rest.split_at_mut(r.end - offset);
            rest = tail;
            offset = r.end;
            f(head, r.clone());
        }
        return;
    }
    let mut pieces: Vec<(&mut [f64], std::ops::Range<usize>)> = Vec::with_capacity(ranges.len());
    let mut rest = v;
    let mut offset = 0;
    for r in ranges {
        let (head, tail) = rest.split_at_mut(r.end - offset);
        rest = tail;
        offset = r.end;
        pieces.push((head, r.clone()));
    }
    let f = &f;
    std::thread::scope(|sc| {
        let mut buckets: Vec<Vec<(&mut [f64], std::ops::Range<usize>)>> = (0..t).map(|_| Vec::new()).collect();
        for (i, p) in pieces.into_iter().enumerate() {
            buckets[i % t].push(p);
        }
        for b in buckets {
            sc.spawn(move || {
                for (chunk, r) in b {
                    f(chunk, r);
                }
            });
        }
    });
}

/// An index map from a "big" table onto a "small" one (see module docs),
/// plus its block structure. The big table's trailing axes form a run of
/// `block` cells that are either all *kept* in the small table (consecutive
/// big cells map to consecutive small cells) or all *reduced* (the whole run
/// maps to one small cell). Iterating block-wise turns per-element scatters,
/// which serialize on repeated targets, into vectorizable loops.
#[derive(Clone, Copy)]
struct Map<'a> {
    idx: &'a [u32],
    block: usize,
    kept: bool,
}

impl<'a> Map<'a> {
    fn new(idx: &'a [u32], bk: u64) -> Map<'a> {
        Map { idx, block: ((bk >> 1) as usize).max(1), kept: bk & 1 == 1 }
    }
    /// The part of the map covering big cells `r`.
    fn range(&self, r: std::ops::Range<usize>) -> Map<'a> {
        Map { idx: &self.idx[r], block: self.block, kept: self.kept }
    }
    /// True if this map is the identity (small == big, same order).
    fn is_identity(&self, small_len: usize) -> bool {
        self.kept && self.block == self.idx.len() && small_len == self.idx.len()
    }
}

/// Borrowed view of a `PlanC`.
struct Plan<'a> {
    l_off: &'a [u64],
    j_off: &'a [u64],
    exp_maps: Vec<Map<'a>>,
    e_src: &'a [u32],
    e_dst: &'a [u32],
    e_rev: &'a [u64],
    sep_off: &'a [u64],
    src_maps: Vec<Map<'a>>,
    dst_maps: Vec<Map<'a>>,
    mar_parent: &'a [u32],
    mar_maps: Vec<Map<'a>>,
    total_clique: usize,
    m_l: &'a [u32],
    m_off: &'a [u64],
    y: &'a [f64],
    m_inv_sd: Vec<f64>,
    m_maps: Vec<Map<'a>>,
    /// Loss cliques expanded into each junction-tree clique.
    exp_groups: Vec<Vec<usize>>,
    /// Loss cliques whose marginal is read from each junction-tree clique.
    mar_groups: Vec<Vec<usize>>,
    /// Measurements read from each loss clique.
    m_groups: Vec<Vec<usize>>,
}

unsafe fn maps<'a>(n: usize, off: *const u64, idx: *const u32, bk: *const u64) -> Vec<Map<'a>> {
    let off = sl(off, n + 1);
    let idx = sl(idx, off[n] as usize);
    let bk = sl(bk, n);
    (0..n).map(|i| Map::new(&idx[off[i] as usize..off[i + 1] as usize], bk[i])).collect()
}

fn groups(n: usize, parent: &[u32]) -> Vec<Vec<usize>> {
    let mut g = vec![Vec::new(); n];
    for (i, &p) in parent.iter().enumerate() {
        g[p as usize].push(i);
    }
    g
}

impl<'a> Plan<'a> {
    unsafe fn new(p: &'a PlanC) -> Plan<'a> {
        let (n_l, n_j, n_e, n_m) = (p.n_l as usize, p.n_j as usize, p.n_e as usize, p.n_m as usize);
        let m_off = sl(p.m_off, n_m + 1);
        Plan {
            l_off: sl(p.l_off, n_l + 1),
            j_off: sl(p.j_off, n_j + 1),
            exp_maps: maps(n_l, p.exp_map_off, p.exp_map, p.exp_bk),
            e_src: sl(p.e_src, n_e),
            e_dst: sl(p.e_dst, n_e),
            e_rev: sl(p.e_rev, n_e),
            sep_off: sl(p.sep_off, n_e + 1),
            src_maps: maps(n_e, p.src_map_off, p.src_map, p.src_bk),
            dst_maps: maps(n_e, p.dst_map_off, p.dst_map, p.dst_bk),
            mar_parent: sl(p.mar_parent, n_l),
            mar_maps: maps(n_l, p.mar_map_off, p.mar_map, p.mar_bk),
            total_clique: p.total_clique as usize,
            m_l: sl(p.m_l, n_m),
            m_off,
            y: sl(p.y, m_off[n_m] as usize),
            m_inv_sd: sl(p.m_sd, n_m).iter().map(|&s| 1.0 / s.max(1e-12)).collect(),
            m_maps: maps(n_m, p.m_map_off, p.m_map, p.m_bk),
            exp_groups: groups(n_j, sl(p.exp_parent, n_l)),
            mar_groups: groups(n_j, sl(p.mar_parent, n_l)),
            m_groups: groups(n_l, sl(p.m_l, n_m)),
        }
    }

    fn n_l(&self) -> usize {
        self.l_off.len() - 1
    }
    fn n_cells(&self) -> usize {
        self.l_off[self.n_l()] as usize
    }
    fn l_range(&self, i: usize) -> std::ops::Range<usize> {
        self.l_off[i] as usize..self.l_off[i + 1] as usize
    }
    fn j_range(&self, j: usize) -> std::ops::Range<usize> {
        self.j_off[j] as usize..self.j_off[j + 1] as usize
    }
    fn sep_range(&self, e: usize) -> std::ops::Range<usize> {
        self.sep_off[e] as usize..self.sep_off[e + 1] as usize
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Reduce {
    LogSumExp,
    Max,
}

/// Scratch buffers reused across oracle calls.
struct Work {
    beliefs: Vec<f64>,
    probs: Vec<f64>,
    /// Scratch for fused projections (several outputs at once).
    mbuf: Vec<f64>,
    messages: Vec<f64>,
    sent: Vec<bool>,
    acc: Vec<f64>,
}

impl Work {
    fn new(plan: &Plan) -> Work {
        let n_e = plan.e_src.len();
        let max_sep = (0..n_e).map(|e| plan.sep_range(e).len()).max().unwrap_or(0);
        let max_l = (0..plan.n_l()).map(|i| plan.l_range(i).len()).max().unwrap_or(0);
        let max_m = (0..plan.m_l.len())
            .map(|m| (plan.m_off[m + 1] - plan.m_off[m]) as usize)
            .max()
            .unwrap_or(0);
        Work {
            beliefs: vec![0.0; *plan.j_off.last().unwrap() as usize],
            probs: vec![0.0; *plan.j_off.last().unwrap() as usize],
            mbuf: vec![
                0.0;
                plan.mar_groups
                    .iter()
                    .map(|g| g.iter().map(|&i| plan.l_range(i).len()).sum::<usize>())
                    .chain(plan.m_groups.iter().map(|g| {
                        g.iter().map(|&m| (plan.m_off[m + 1] - plan.m_off[m]) as usize).sum::<usize>()
                    }))
                    .max()
                    .unwrap_or(0)
            ],
            messages: vec![0.0; plan.sep_off[n_e] as usize],
            sent: vec![false; n_e],
            acc: vec![0.0; max_sep.max(max_l).max(max_m)],
        }
    }
}

/// Sum with 8 independent partial sums so it vectorizes.
#[inline(always)]
fn vsum(xs: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let chunks = xs.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for l in 0..8 {
            lanes[l] += c[l];
        }
    }
    let mut s = rest.iter().sum::<f64>();
    for l in lanes {
        s += l;
    }
    s
}

/// Max with 8 independent partial maxima so it vectorizes.
#[inline(always)]
fn vmax(xs: &[f64]) -> f64 {
    let mut lanes = [f64::NEG_INFINITY; 8];
    let chunks = xs.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for l in 0..8 {
            lanes[l] = if c[l] > lanes[l] { c[l] } else { lanes[l] };
        }
    }
    let mut m = f64::NEG_INFINITY;
    for &x in rest.iter().chain(lanes.iter()) {
        if x > m {
            m = x;
        }
    }
    m
}

/// `sum_j exp(xs[j] - c)`, vectorized via a chunked buffer.
#[inline(always)]
fn sum_exp(xs: &[f64], c: f64, buf: &mut [f64; CHUNK]) -> f64 {
    let mut s = 0.0;
    for chunk in xs.chunks(CHUNK) {
        let b = &mut buf[..chunk.len()];
        for (bb, &x) in b.iter_mut().zip(chunk) {
            *bb = fast_exp(x - c);
        }
        s += vsum(b);
    }
    s
}

/// `out[o] = max(out[o], v[k] - sub[o])` over `o = map[k]`; `out` must be
/// pre-filled with -inf.
#[inline(always)]
fn scatter_max(v: &[f64], sub: Option<&[f64]>, m: Map, out: &mut [f64]) {
    let bsz = m.block;
    if bsz < SMALL_BLOCK {
        match sub {
            None => {
                for (k, &x) in v.iter().enumerate() {
                    let o = m.idx[k] as usize;
                    if x > out[o] {
                        out[o] = x;
                    }
                }
            }
            Some(s) => {
                for (k, &x) in v.iter().enumerate() {
                    let o = m.idx[k] as usize;
                    let t = x - s[o];
                    if t > out[o] {
                        out[o] = t;
                    }
                }
            }
        }
        return;
    }
    for start in (0..v.len()).step_by(bsz) {
        let blk = &v[start..start + bsz];
        let o = m.idx[start] as usize;
        if m.kept {
            let out = &mut out[o..o + bsz];
            match sub {
                None => {
                    for (x, &y) in out.iter_mut().zip(blk) {
                        *x = if y > *x { y } else { *x };
                    }
                }
                Some(s) => {
                    let s = &s[o..o + bsz];
                    for j in 0..bsz {
                        let t = blk[j] - s[j];
                        out[j] = if t > out[j] { t } else { out[j] };
                    }
                }
            }
        } else {
            let mut t = vmax(blk);
            if let Some(s) = sub {
                t -= s[o];
            }
            if t > out[o] {
                out[o] = t;
            }
        }
    }
}

/// `acc[o] += exp(v[k] - sub[o] - mx[o])` over `o = map[k]`, skipping cells
/// with `mx[o] == -inf`. `acc` must be pre-zeroed.
#[inline(always)]
fn scatter_sum_exp(v: &[f64], sub: Option<&[f64]>, m: Map, mx: &[f64], acc: &mut [f64], buf: &mut [f64; CHUNK]) {
    let bsz = m.block;
    if bsz < SMALL_BLOCK {
        // gather -> vectorized exp -> scatter, chunk-wise.
        let mut start = 0;
        while start < v.len() {
            let end = (start + CHUNK).min(v.len());
            let b = &mut buf[..end - start];
            let idx = &m.idx[start..end];
            for (k, bk) in b.iter_mut().enumerate() {
                let o = idx[k] as usize;
                let c = mx[o];
                let sv = match sub {
                    None => 0.0,
                    Some(s) => s[o],
                };
                *bk = if c == f64::NEG_INFINITY { f64::NEG_INFINITY } else { v[start + k] - sv - c };
            }
            exp_inplace(b);
            for (k, &bk) in b.iter().enumerate() {
                acc[idx[k] as usize] += bk;
            }
            start = end;
        }
        return;
    }
    for start in (0..v.len()).step_by(bsz) {
        let blk = &v[start..start + bsz];
        let o = m.idx[start] as usize;
        if m.kept {
            let mut j0 = 0;
            while j0 < bsz {
                let j1 = (j0 + CHUNK).min(bsz);
                let b = &mut buf[..j1 - j0];
                for j in j0..j1 {
                    let c = mx[o + j];
                    let sv = match sub {
                        None => 0.0,
                        Some(s) => s[o + j],
                    };
                    b[j - j0] = if c == f64::NEG_INFINITY { f64::NEG_INFINITY } else { blk[j] - sv - c };
                }
                exp_inplace(b);
                let a = &mut acc[o + j0..o + j1];
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                j0 = j1;
            }
        } else {
            let c = mx[o];
            if c == f64::NEG_INFINITY {
                continue;
            }
            let c = match sub {
                None => c,
                Some(s) => c + s[o],
            };
            acc[o] += sum_exp(blk, c, buf);
        }
    }
}

/// `big[k] += small[map[k]]`.
#[inline(always)]
fn broadcast_add(big: &mut [f64], m: Map, small: &[f64]) {
    let bsz = m.block;
    if bsz < SMALL_BLOCK {
        for (x, &o) in big.iter_mut().zip(m.idx) {
            *x += small[o as usize];
        }
        return;
    }
    for start in (0..big.len()).step_by(bsz) {
        let o = m.idx[start] as usize;
        let blk = &mut big[start..start + bsz];
        if m.kept {
            for (x, &y) in blk.iter_mut().zip(&small[o..o + bsz]) {
                *x += y;
            }
        } else {
            let y = small[o];
            for x in blk.iter_mut() {
                *x += y;
            }
        }
    }
}

/// Maps whose blocks are shorter than this use element-wise gather/scatter
/// loops: per-block loop overhead would dominate, and dependency chains on
/// repeated targets are at most this long.
const SMALL_BLOCK: usize = 16;

/// Cells per cache block in fused multi-map loops (fits in L1).
const FUSE_CHUNK: usize = 4096;

/// Visit the runs of `m` covering big cells `start..start + len`: calls
/// `f(offset_in_range, run_len, small_index_of_first_cell)`. Works for any
/// range, not just block-aligned ones: within a block, kept runs map to
/// consecutive small cells and reduced runs map to a single small cell.
#[inline(always)]
fn for_segments(m: Map, start: usize, len: usize, mut f: impl FnMut(usize, usize, usize)) {
    let b = m.block;
    let end = start + len;
    let mut p = start;
    while p < end {
        let seg = (b - p % b).min(end - p);
        f(p - start, seg, m.idx[p] as usize);
        p += seg;
    }
}

/// `chunk[k] += small[map[start + k]]`.
#[inline(always)]
fn broadcast_add_seg(chunk: &mut [f64], start: usize, m: Map, small: &[f64]) {
    if m.block < SMALL_BLOCK {
        let n = chunk.len();
        for (x, &o) in chunk.iter_mut().zip(&m.idx[start..start + n]) {
            *x += small[o as usize];
        }
        return;
    }
    for_segments(m, start, chunk.len(), |off, seg, o| {
        let c = &mut chunk[off..off + seg];
        if m.kept {
            for (x, &y) in c.iter_mut().zip(&small[o..o + seg]) {
                *x += y;
            }
        } else {
            let y = small[o];
            for x in c.iter_mut() {
                *x += y;
            }
        }
    });
}

/// `out[map[start + k]] += chunk[k]`.
#[inline(always)]
fn scatter_sum_seg(chunk: &[f64], start: usize, m: Map, out: &mut [f64]) {
    if m.block < SMALL_BLOCK {
        for (&x, &o) in chunk.iter().zip(&m.idx[start..start + chunk.len()]) {
            out[o as usize] += x;
        }
        return;
    }
    for_segments(m, start, chunk.len(), |off, seg, o| {
        let c = &chunk[off..off + seg];
        if m.kept {
            for (x, &y) in out[o..o + seg].iter_mut().zip(c) {
                *x += y;
            }
        } else {
            out[o] += vsum(c);
        }
    });
}

/// `big = sum_i broadcast(small_i)` for several maps into the same big table,
/// in one cache-blocked pass (instead of one pass per map).
#[inline(always)]
fn fused_broadcast_set(big: &mut [f64], terms: &[(Map, &[f64])]) {
    let ranges = split(big.len(), 1);
    par_chunks_mut(big, &ranges, |part, r| {
        let mut pos = r.start;
        for chunk in part.chunks_mut(FUSE_CHUNK) {
            chunk.fill(0.0);
            for &(m, small) in terms {
                broadcast_add_seg(chunk, pos, m, small);
            }
            pos += chunk.len();
        }
    });
}

/// `outs[t] = project_sum(v onto map t)` for several maps out of the same big
/// table, in one cache-blocked pass. `outs` is a flat buffer split at `offs`.
///
/// Identity maps are plain (parallel) copies. The others are reduced in
/// parallel into per-part partial outputs when those are small relative to
/// the input (the usual case: projections onto fewer attributes); otherwise
/// on one thread, to avoid allocating and combining large partial buffers.
#[inline(always)]
fn fused_scatter_sum(v: &[f64], maps: &[Map], offs: &[usize], outs: &mut [f64]) {
    let total = *offs.last().unwrap();
    let outs = &mut outs[..total];
    let mut rest: Vec<usize> = Vec::new();
    for (t, m) in maps.iter().enumerate() {
        let out = &mut outs[offs[t]..offs[t + 1]];
        if m.is_identity(out.len()) {
            let ranges = split(v.len(), 1);
            par_chunks_mut(out, &ranges, |c, r| c.copy_from_slice(&v[r]));
        } else {
            out.fill(0.0);
            rest.push(t);
        }
    }
    if rest.is_empty() {
        return;
    }
    let rest_total: usize = rest.iter().map(|&t| offs[t + 1] - offs[t]).sum();
    let run = |r: std::ops::Range<usize>, o: &mut [f64], local: bool| {
        let mut pos = r.start;
        for chunk in v[r].chunks(FUSE_CHUNK) {
            let mut base = 0;
            for &t in &rest {
                let len = offs[t + 1] - offs[t];
                let dst = if local { &mut o[base..base + len] } else { &mut o[offs[t]..offs[t + 1]] };
                scatter_sum_seg(chunk, pos, maps[t], dst);
                base += len;
            }
            pos += chunk.len();
        }
    };
    let ranges = split(v.len(), 1);
    if ranges.len() <= 1 || rest_total * ranges.len() > 2 * v.len() {
        run(0..v.len(), outs, false);
        return;
    }
    let partials = par_map(&ranges, |r| {
        let mut o = vec![0.0; rest_total];
        run(r, &mut o, true);
        o
    });
    let mut base = 0;
    for &t in &rest {
        let len = offs[t + 1] - offs[t];
        let out = &mut outs[offs[t]..offs[t + 1]];
        for p in &partials {
            for (x, &y) in out.iter_mut().zip(&p[base..base + len]) {
                *x += y;
            }
        }
        base += len;
    }
}

/// Reduce `values - sub[map]` onto `out` (log-sum-exp or max over the cells
/// mapping to each output cell).
#[inline(always)]
fn project(values: &[f64], sub: Option<&[f64]>, m: Map, out: &mut [f64], acc: &mut [f64], how: Reduce) {
    if sub.is_none() && m.is_identity(out.len()) {
        out.copy_from_slice(values);
        return;
    }
    let ranges = split(values.len(), m.block);
    if ranges.len() > 1 {
        project_par(values, sub, m, out, acc, how, &ranges);
        return;
    }
    out.fill(f64::NEG_INFINITY);
    scatter_max(values, sub, m, out);
    if how == Reduce::Max {
        return;
    }
    let acc = &mut acc[..out.len()];
    acc.fill(0.0);
    let mut buf = [0.0f64; CHUNK];
    scatter_sum_exp(values, sub, m, out, acc, &mut buf);
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        if *o != f64::NEG_INFINITY {
            *o += a.ln();
        }
    }
}

/// Parallel `project` for large tables: each part is reduced into a private
/// partial result and the partials are combined in part order.
#[inline(always)]
fn project_par(
    values: &[f64],
    sub: Option<&[f64]>,
    m: Map,
    out: &mut [f64],
    acc: &mut [f64],
    how: Reduce,
    ranges: &[std::ops::Range<usize>],
) {
    let n = out.len();
    let partial_max = par_map(ranges, |r| {
        let mut o = vec![f64::NEG_INFINITY; n];
        scatter_max(&values[r.clone()], sub, m.range(r), &mut o);
        o
    });
    out.fill(f64::NEG_INFINITY);
    for p in &partial_max {
        for (x, &y) in out.iter_mut().zip(p) {
            if y > *x {
                *x = y;
            }
        }
    }
    if how == Reduce::Max {
        return;
    }
    let maxes: &[f64] = out;
    let partial_sum = par_map(ranges, |r| {
        let mut a = vec![0.0; n];
        let mut buf = [0.0f64; CHUNK];
        scatter_sum_exp(&values[r.clone()], sub, m.range(r), maxes, &mut a, &mut buf);
        a
    });
    let acc = &mut acc[..n];
    acc.fill(0.0);
    for p in &partial_sum {
        for (x, &y) in acc.iter_mut().zip(p) {
            *x += y;
        }
    }
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        if *o != f64::NEG_INFINITY {
            *o += a.ln();
        }
    }
}

/// `big[k] += small[map[k]]`, in parallel for large tables.
#[inline(always)]
fn broadcast_add_par(big: &mut [f64], m: Map, small: &[f64]) {
    let ranges = split(big.len(), m.block);
    par_chunks_mut(big, &ranges, |chunk, r| broadcast_add(chunk, m.range(r), small));
}

/// In-place `exp(x + shift)`, in parallel for large slices.
#[inline(always)]
fn exp_shift_par(v: &mut [f64], shift: f64) {
    let ranges = split(v.len(), 1);
    par_chunks_mut(v, &ranges, |c, _| {
        for x in c.iter_mut() {
            *x += shift;
        }
        exp_inplace(c);
    });
}

/// Dot product, partitioned so the result doesn't depend on thread count.
#[inline(always)]
fn dot_par(a: &[f64], b: &[f64]) -> f64 {
    let ranges = split(a.len(), 1);
    par_map(&ranges, |r| dot(&a[r.clone()], &b[r])).iter().sum()
}

/// `sum_k grad[k] * (mu[k] - mu2[k])`.
#[inline(always)]
fn armijo_dot(grad: &[f64], mu: &[f64], mu2: &[f64]) -> f64 {
    let ranges = split(grad.len(), 1);
    par_map(&ranges, |r| {
        let (g, a, b) = (&grad[r.clone()], &mu[r.clone()], &mu2[r]);
        let mut lanes = [0.0f64; 8];
        let mut k = 0;
        while k + 8 <= g.len() {
            for l in 0..8 {
                lanes[l] += g[k + l] * (a[k + l] - b[k + l]);
            }
            k += 8;
        }
        let mut s: f64 = lanes.iter().sum();
        while k < g.len() {
            s += g[k] * (a[k] - b[k]);
            k += 1;
        }
        s
    })
    .iter()
    .sum()
}

#[inline(always)]
fn logsumexp(values: &[f64]) -> f64 {
    let m = vmax(values);
    if m == f64::NEG_INFINITY {
        return m;
    }
    let mut buf = [0.0f64; CHUNK];
    m + sum_exp(values, m, &mut buf).ln()
}

/// HUGIN message passing (mirrors `marginal_oracles.message_passing_hugin`).
/// Leaves calibrated log-beliefs in `w.beliefs`.
#[inline(always)]
fn calibrate(plan: &Plan, w: &mut Work, theta: &[f64], how: Reduce) {
    // Expand potentials onto the junction-tree cliques (one fused pass per
    // junction-tree clique).
    for j in 0..plan.j_off.len() - 1 {
        let terms: Vec<(Map, &[f64])> = plan.exp_groups[j]
            .iter()
            .map(|&i| (plan.exp_maps[i], &theta[plan.l_range(i)]))
            .collect();
        fused_broadcast_set(&mut w.beliefs[plan.j_range(j)], &terms);
    }
    w.sent.fill(false);
    for e in 0..plan.e_src.len() {
        let (s, d) = (plan.e_src[e] as usize, plan.e_dst[e] as usize);
        let rev = plan.e_rev[e] as usize;
        let sep = plan.sep_range(e);
        let (before, after) = w.messages.split_at_mut(sep.start);
        let msg = &mut after[..sep.len()];
        let src = &w.beliefs[plan.j_range(s)];
        // HUGIN division: subtract the message previously received from d
        // (same separator, so it is indexed by the same map).
        let sub = if w.sent[rev] {
            let r = plan.sep_range(rev);
            debug_assert!(r.end <= sep.start);
            Some(&before[r])
        } else {
            None
        };
        project(src, sub, plan.src_maps[e], msg, &mut w.acc, how);
        broadcast_add_par(&mut w.beliefs[plan.j_range(d)], plan.dst_maps[e], msg);
        w.sent[e] = true;
    }
}

/// Marginals over the loss cliques, normalized to `total` (mirrors
/// `_finalize_beliefs` with log-sum-exp). Each junction-tree clique that is
/// read from is converted to probabilities once (`exp(b - logsumexp(b)) *
/// total`); every loss-clique marginal is then a plain sum over it, instead
/// of a separate log-sum-exp projection (max pass + exp pass) per loss clique.
#[inline(always)]
fn marginals(plan: &Plan, w: &mut Work, theta: &[f64], total: f64, mu: &mut [f64]) {
    calibrate(plan, w, theta, Reduce::LogSumExp);
    let log_total = total.ln();
    for j in 0..plan.j_off.len() - 1 {
        let group = &plan.mar_groups[j];
        if group.is_empty() {
            continue;
        }
        let jr = plan.j_range(j);
        let shift = log_total - logsumexp(&w.beliefs[jr.clone()]);
        let (ident, other): (Vec<usize>, Vec<usize>) = group
            .iter()
            .partition(|&&i| plan.mar_maps[i].is_identity(plan.l_range(i).len()));
        // Loss cliques equal to this table: exp straight into the output.
        for &i in &ident {
            let out = &mut mu[plan.l_range(i)];
            out.copy_from_slice(&w.beliefs[jr.clone()]);
            exp_shift_par(out, shift);
        }
        if other.is_empty() {
            continue;
        }
        let p = &mut w.probs[jr.clone()];
        match ident.first() {
            Some(&i) => p.copy_from_slice(&mu[plan.l_range(i)]),
            None => {
                p.copy_from_slice(&w.beliefs[jr]);
                exp_shift_par(p, shift);
            }
        }
        let p: &[f64] = p;
        let maps: Vec<Map> = other.iter().map(|&i| plan.mar_maps[i]).collect();
        let mut offs = vec![0usize];
        for &i in &other {
            offs.push(offs.last().unwrap() + plan.l_range(i).len());
        }
        let buf = &mut w.mbuf[..*offs.last().unwrap()];
        fused_scatter_sum(p, &maps, &offs, buf);
        for (t, &i) in other.iter().enumerate() {
            mu[plan.l_range(i)].copy_from_slice(&buf[offs[t]..offs[t + 1]]);
        }
    }
}

/// L2 loss over identity measurements and (optionally) its gradient with
/// respect to the loss-clique marginals (mirrors `marginal_loss._l2_loss`).
#[inline(always)]
fn loss_and_grad(plan: &Plan, w: &mut Work, mu: &[f64], mut grad: Option<&mut [f64]>) -> f64 {
    let mut loss = 0.0;
    for i in 0..plan.n_l() {
        let group = &plan.m_groups[i];
        let lr = plan.l_range(i);
        let mu_l = &mu[lr.clone()];
        let (ident, other): (Vec<usize>, Vec<usize>) = group.iter().partition(|&&m| {
            plan.m_maps[m].is_identity((plan.m_off[m + 1] - plan.m_off[m]) as usize)
        });
        let ys = |m: usize| &plan.y[plan.m_off[m] as usize..plan.m_off[m + 1] as usize];
        // Measurements over the whole loss clique: residuals read mu directly.
        let ranges = split(mu_l.len(), 1);
        for &m in &ident {
            let (y, inv_sd) = (ys(m), plan.m_inv_sd[m]);
            loss += 0.5
                * par_map(&ranges, |r| {
                    let mut s = 0.0;
                    for (&a, &b) in mu_l[r.clone()].iter().zip(&y[r]) {
                        let d = (a - b) * inv_sd;
                        s += d * d;
                    }
                    s
                })
                .iter()
                .sum::<f64>();
        }
        // Other measurements: project mu_L onto all of them in one pass, then
        // turn the projections into d loss / d projection.
        let maps: Vec<Map> = other.iter().map(|&m| plan.m_maps[m]).collect();
        let mut offs = vec![0usize];
        for &m in &other {
            offs.push(offs.last().unwrap() + (plan.m_off[m + 1] - plan.m_off[m]) as usize);
        }
        let buf = &mut w.mbuf[..*offs.last().unwrap()];
        if !other.is_empty() {
            fused_scatter_sum(mu_l, &maps, &offs, buf);
        }
        for (t, &m) in other.iter().enumerate() {
            let (y, inv_sd) = (ys(m), plan.m_inv_sd[m]);
            let proj = &mut buf[offs[t]..offs[t + 1]];
            let mut s = 0.0;
            for (p, &yy) in proj.iter_mut().zip(y) {
                let d = (*p - yy) * inv_sd;
                s += d * d;
                *p = d * inv_sd;
            }
            loss += 0.5 * s;
        }
        // Gradient: one cache-blocked pass over grad_L.
        if let Some(g) = grad.as_deref_mut() {
            let buf: &[f64] = buf;
            let g_l = &mut g[lr];
            par_chunks_mut(g_l, &ranges, |part, r| {
                let mut pos = r.start;
                for chunk in part.chunks_mut(FUSE_CHUNK) {
                    let cr = pos..pos + chunk.len();
                    chunk.fill(0.0);
                    for &m in &ident {
                        let (y, inv_sd) = (ys(m), plan.m_inv_sd[m]);
                        let w2 = inv_sd * inv_sd;
                        for ((x, &a), &b) in chunk.iter_mut().zip(&mu_l[cr.clone()]).zip(&y[cr.clone()]) {
                            *x += (a - b) * w2;
                        }
                    }
                    for (t, &mp) in maps.iter().enumerate() {
                        broadcast_add_seg(chunk, pos, mp, &buf[offs[t]..offs[t + 1]]);
                    }
                    pos += chunk.len();
                }
            });
        }
    }
    loss
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let rest: f64 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            lanes[l] += x[l] * y[l];
        }
    }
    rest + lanes.iter().sum::<f64>()
}

/// Normalized primal-dual gap at marginals `mu` (mirrors
/// `estimation.primal_dual_gap`).
#[inline(always)]
fn primal_dual_gap(plan: &Plan, w: &mut Work, mu: &[f64], grad: &mut [f64], neg: &mut [f64], lipschitz: f64) -> f64 {
    let total: f64 = mu[plan.l_range(plan.total_clique)].iter().sum();
    loss_and_grad(plan, w, mu, Some(grad));
    for (n, &g) in neg.iter_mut().zip(grad.iter()) {
        *n = -g;
    }
    calibrate(plan, w, neg, Reduce::Max);
    let mut best = f64::NEG_INFINITY;
    for i in 0..plan.n_l() {
        let j = plan.mar_parent[i] as usize;
        for &b in &w.beliefs[plan.j_range(j)] {
            let v = b * total;
            if v > best {
                best = v;
            }
        }
    }
    let gap = dot_par(grad, mu) + best;
    let scale = total * (lipschitz * plan.n_cells() as f64).sqrt();
    gap / scale
}

/// Compute marginals (`how == 0`, log-sum-exp) or max-marginals scaled by
/// `total` (`how == 1`) of `theta`. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn mbi_oracle(plan: *const PlanC, theta: *const f64, total: f64, how: u32, out: *mut f64) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return oracle_avx2(plan, theta, total, how, out);
        }
    }
    oracle_impl(plan, theta, total, how, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn oracle_avx2(plan: *const PlanC, theta: *const f64, total: f64, how: u32, out: *mut f64) -> i32 {
    oracle_impl(plan, theta, total, how, out)
}

#[inline(always)]
unsafe fn oracle_impl(plan: *const PlanC, theta: *const f64, total: f64, how: u32, out: *mut f64) -> i32 {
    let plan = Plan::new(&*plan);
    let n = plan.n_cells();
    let theta = sl(theta, n);
    let out = slice::from_raw_parts_mut(out, n);
    let mut w = Work::new(&plan);
    if how == 0 {
        marginals(&plan, &mut w, theta, total, out);
    } else {
        calibrate(&plan, &mut w, theta, Reduce::Max);
        for i in 0..plan.n_l() {
            let j = plan.mar_parent[i] as usize;
            let o = &mut out[plan.l_range(i)];
            project(&w.beliefs[plan.j_range(j)], None, plan.mar_maps[i], o, &mut w.acc, Reduce::Max);
            for v in o.iter_mut() {
                *v *= total;
            }
        }
    }
    0
}

/// Calibrated junction-tree beliefs of `theta`, each clique normalized to sum
/// to `total`, written to `out` (laid out by `j_off`). Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn mbi_beliefs(plan: *const PlanC, theta: *const f64, total: f64, out: *mut f64) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return beliefs_avx2(plan, theta, total, out);
        }
    }
    beliefs_impl(plan, theta, total, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn beliefs_avx2(plan: *const PlanC, theta: *const f64, total: f64, out: *mut f64) -> i32 {
    beliefs_impl(plan, theta, total, out)
}

#[inline(always)]
unsafe fn beliefs_impl(plan: *const PlanC, theta: *const f64, total: f64, out: *mut f64) -> i32 {
    let plan = Plan::new(&*plan);
    let theta = sl(theta, plan.n_cells());
    let n_j = plan.j_off.len() - 1;
    let out = slice::from_raw_parts_mut(out, plan.j_off[n_j] as usize);
    let mut w = Work::new(&plan);
    calibrate(&plan, &mut w, theta, Reduce::LogSumExp);
    let log_total = total.ln();
    for j in 0..n_j {
        let b = &w.beliefs[plan.j_range(j)];
        let shift = log_total - logsumexp(b);
        let o = &mut out[plan.j_range(j)];
        for (x, &v) in o.iter_mut().zip(b) {
            *x = v + shift;
        }
        exp_inplace(o);
    }
    0
}

/// Run `MirrorDescent.estimate` (mirrors `Estimator.estimate` +
/// `MirrorDescent._step`).
///
/// * `theta`: initial potentials over the loss cliques; overwritten with the
///   final potentials.
/// * `mu_out`: receives the marginals of the final potentials.
/// * `stepsize`: if > 0 a fixed step size is used, otherwise Armijo line
///   search starting from `alpha0`.
/// * `tol`: early-stopping tolerance on the normalized primal-dual gap, or a
///   negative value to disable it.
/// * `info` (len 4): steps run, last gap (NaN if never computed), final alpha,
///   final accepted loss.
#[no_mangle]
pub unsafe extern "C" fn mbi_mirror_descent(
    plan: *const PlanC,
    theta: *mut f64,
    mu_out: *mut f64,
    total: f64,
    lipschitz: f64,
    alpha0: f64,
    stepsize: f64,
    iters: u64,
    block: u64,
    tol: f64,
    info: *mut f64,
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return md_avx2(plan, theta, mu_out, total, lipschitz, alpha0, stepsize, iters, block, tol, info);
        }
    }
    md_impl(plan, theta, mu_out, total, lipschitz, alpha0, stepsize, iters, block, tol, info)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn md_avx2(
    plan: *const PlanC, theta: *mut f64, mu_out: *mut f64, total: f64, lipschitz: f64,
    alpha0: f64, stepsize: f64, iters: u64, block: u64, tol: f64, info: *mut f64,
) -> i32 {
    md_impl(plan, theta, mu_out, total, lipschitz, alpha0, stepsize, iters, block, tol, info)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn md_impl(
    plan: *const PlanC, theta: *mut f64, mu_out: *mut f64, total: f64, lipschitz: f64,
    alpha0: f64, stepsize: f64, iters: u64, block: u64, tol: f64, info: *mut f64,
) -> i32 {
    let plan = Plan::new(&*plan);
    let n = plan.n_cells();
    let theta = slice::from_raw_parts_mut(theta, n);
    let mu_out = slice::from_raw_parts_mut(mu_out, n);
    let info = slice::from_raw_parts_mut(info, 4);
    let mut w = Work::new(&plan);

    let fixed = stepsize > 0.0;
    let mut alpha = if fixed { stepsize } else { alpha0 };
    let mut mu = vec![0.0; n]; // marginals of theta
    let mut mu2 = vec![0.0; n];
    let mut theta2 = vec![0.0; n];
    let mut grad = vec![0.0; n];
    let mut neg = vec![0.0; n];
    let mut state_mu = vec![0.0; n]; // `state.mu`: marginals at the start of the last step
    let mut theta_cur = theta.to_vec();

    marginals(&plan, &mut w, &theta_cur, total, &mut mu);
    let mut loss_acc = loss_and_grad(&plan, &mut w, &mu, None);
    state_mu.copy_from_slice(&mu);

    let blocks = (iters + block - 1) / block.max(1);
    let mut steps = 0u64;
    let mut prev_gap = f64::INFINITY;
    let mut last_gap = f64::NAN;
    for _ in 0..blocks {
        for step_in_block in 0..block {
            // mu == marginals(theta) is maintained across steps, so the oracle
            // call at the start of MirrorDescent._step is not repeated.
            let loss = loss_and_grad(&plan, &mut w, &mu, Some(&mut grad));
            {
                let (th, g) = (&theta_cur, &grad);
                let ranges = split(n, 1);
                par_chunks_mut(&mut theta2, &ranges, |c, r| {
                    for ((t2, &t), &gk) in c.iter_mut().zip(&th[r.clone()]).zip(&g[r]) {
                        *t2 = t - alpha * gk;
                    }
                });
            }
            if step_in_block + 1 == block {
                // `state.mu` (marginals at the start of the last step) is only
                // read by the convergence check at the end of the block.
                state_mu.copy_from_slice(&mu);
            }
            marginals(&plan, &mut w, &theta2, total, &mut mu2);
            if fixed {
                std::mem::swap(&mut theta_cur, &mut theta2);
                std::mem::swap(&mut mu, &mut mu2);
                loss_acc = loss;
            } else {
                let loss2 = loss_and_grad(&plan, &mut w, &mu2, None);
                let d = armijo_dot(&grad, &mu, &mu2);
                let sufficient = loss - loss2 >= 0.5 * alpha * d;
                if sufficient {
                    alpha *= 1.01;
                    std::mem::swap(&mut theta_cur, &mut theta2);
                    std::mem::swap(&mut mu, &mut mu2);
                    loss_acc = loss2;
                } else {
                    alpha *= 0.5;
                    loss_acc = loss;
                }
            }
            steps += 1;
        }
        if tol >= 0.0 {
            let gap = primal_dual_gap(&plan, &mut w, &state_mu, &mut grad, &mut neg, lipschitz);
            last_gap = gap;
            if gap <= tol || gap == prev_gap {
                break;
            }
            prev_gap = gap;
        }
    }
    mu_out.copy_from_slice(&mu);
    theta.copy_from_slice(&theta_cur);
    info[0] = steps as f64;
    info[1] = last_gap;
    info[2] = alpha;
    info[3] = loss_acc;
    0
}

// ---------------------------------------------------------------------------
// Junction-tree sizes for AIM's candidate filtering.
// ---------------------------------------------------------------------------

/// Fixed-width bitset over attributes.
#[derive(Clone)]
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Bits {
        Bits(vec![0; (n + 63) / 64])
    }
    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
    fn clear(&mut self, i: usize) {
        self.0[i / 64] &= !(1 << (i % 64));
    }
    fn get(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }
    fn or(&mut self, o: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            *a |= b;
        }
    }
    fn is_subset(&self, o: &Bits) -> bool {
        self.0.iter().zip(&o.0).all(|(a, b)| a & !b == 0)
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &word)| {
            let mut x = word;
            std::iter::from_fn(move || {
                if x == 0 {
                    None
                } else {
                    let t = x.trailing_zeros() as usize;
                    x &= x - 1;
                    Some(w * 64 + t)
                }
            })
        })
    }
    fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

fn prod(bits: &Bits, sizes: &[u64]) -> Option<u128> {
    bits.iter().try_fold(1u128, |acc, i| acc.checked_mul(sizes[i] as u128))
}

/// Mirrors `junction_tree.hypothetical_model_size` (in cells) for the graph
/// whose edges are the pairs within `cliques`. Returns None on overflow.
fn model_cells(n: usize, sizes: &[u64], cliques: &[&[u32]]) -> Option<u128> {
    let mut adj = vec![Bits::new(n); n];
    let mut covered = Bits::new(n);
    for cl in cliques {
        for &u in cl.iter() {
            covered.set(u as usize);
            for &v in cl.iter() {
                if u != v {
                    adj[u as usize].set(v as usize);
                }
            }
        }
    }
    let n_edges: u32 = adj.iter().map(|b| b.count()).sum::<u32>() / 2;
    // Tree: make_junction_tree eliminates in DFS post-order (no fill-in), so
    // the maximal cliques are the edges.
    if n_edges > 0 && n_edges as usize == n - 1 {
        let mut seen = Bits::new(n);
        let mut stack = vec![0usize];
        seen.set(0);
        let mut reached = 1;
        while let Some(u) = stack.pop() {
            for v in adj[u].iter() {
                if !seen.get(v) {
                    seen.set(v);
                    reached += 1;
                    stack.push(v);
                }
            }
        }
        if reached == n {
            let mut cells: u128 = 0;
            for u in 0..n {
                for v in adj[u].iter() {
                    if u < v {
                        cells = cells.checked_add((sizes[u] as u128).checked_mul(sizes[v] as u128)?)?;
                    }
                }
            }
            return Some(cells);
        }
    }
    // Greedy elimination (mirrors junction_tree.greedy_order): eliminate the
    // attribute whose closed neighborhood is smallest (1 if it is in no
    // clique), ties broken by attribute order; connect its neighbors.
    let mut alive = Bits::new(n);
    for i in 0..n {
        alive.set(i);
    }
    let cost = |a: usize, adj: &Vec<Bits>, covered: &Bits| -> Option<u128> {
        if !covered.get(a) {
            return Some(1);
        }
        let mut c = adj[a].clone();
        c.set(a);
        prod(&c, sizes)
    };
    let mut costs: Vec<u128> = Vec::with_capacity(n);
    for a in 0..n {
        costs.push(cost(a, &adj, &covered)?);
    }
    let mut elim_cliques: Vec<Bits> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut best = usize::MAX;
        for a in alive.iter() {
            if best == usize::MAX || costs[a] < costs[best] {
                best = a;
            }
        }
        let a = best;
        let nbrs = adj[a].clone();
        let mut c = nbrs.clone();
        c.set(a);
        elim_cliques.push(c);
        alive.clear(a);
        for u in nbrs.iter() {
            adj[u].clear(a);
            let mut add = nbrs.clone();
            add.clear(u);
            adj[u].or(&add);
        }
        adj[a] = Bits::new(n);
        for u in nbrs.iter() {
            costs[u] = cost(u, &adj, &covered)?;
        }
    }
    // Maximal elimination cliques = maximal cliques of the triangulation.
    let mut cells: u128 = 0;
    for (i, c) in elim_cliques.iter().enumerate() {
        let dominated = elim_cliques.iter().enumerate().any(|(j, o)| {
            j != i && c.is_subset(o) && (!o.is_subset(c) || j < i)
        });
        if !dominated {
            cells = cells.checked_add(prod(c, sizes)?)?;
        }
    }
    Some(cells)
}

/// Junction-tree size in cells of `base cliques + candidate` for each
/// candidate (`out[k]`, NaN on overflow). Cliques are lists of attribute
/// indices delimited by offset arrays.
#[no_mangle]
pub unsafe extern "C" fn mbi_model_sizes(
    n_attrs: u64,
    sizes: *const u64,
    n_base: u64,
    base_off: *const u64,
    base_attrs: *const u32,
    n_cand: u64,
    cand_off: *const u64,
    cand_attrs: *const u32,
    out: *mut f64,
) -> i32 {
    let n = n_attrs as usize;
    let sizes = sl(sizes, n);
    let base_off = sl(base_off, n_base as usize + 1);
    let base_attrs = sl(base_attrs, base_off[n_base as usize] as usize);
    let cand_off = sl(cand_off, n_cand as usize + 1);
    let cand_attrs = sl(cand_attrs, cand_off[n_cand as usize] as usize);
    let out = slice::from_raw_parts_mut(out, n_cand as usize);
    let mut cliques: Vec<&[u32]> = (0..n_base as usize)
        .map(|i| &base_attrs[base_off[i] as usize..base_off[i + 1] as usize])
        .collect();
    for k in 0..n_cand as usize {
        cliques.push(&cand_attrs[cand_off[k] as usize..cand_off[k + 1] as usize]);
        out[k] = match model_cells(n, sizes, &cliques) {
            Some(c) => c as f64,
            None => f64::NAN,
        };
        cliques.pop();
    }
    0
}
