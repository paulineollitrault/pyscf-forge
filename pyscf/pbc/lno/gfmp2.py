"""
k-point MP2 (opposite-spin / direct term) via imaginary-time Green's functions
in the FFT-ISDF interpolation-point (IP) representation.

This is intended to replace the expensive canonical kMP2 OS evaluation in
workflows that already build FFT-ISDF (inpv_kpt, coul_kpt).

Notes / scope:
- Implements the *direct* MP2 correlation energy (RHF opposite-spin component).
- Requires a uniform k-mesh including the Gamma point (so we can use the
  k2gamma phase factors to FFT between k-space and supercell stripe space).
- Designed for FFT-ISDF DF objects that expose:
    df_obj.inpv_kpt: (nkpt, nip, nao)
    df_obj.coul_kpt: (nkpt, nip, nip)

The algorithm follows the imaginary-time Laplace transform formulation:
  1/Δ = -∫_0^∞ dτ exp(Δ τ),  where Δ = ε_i + ε_j - ε_a - ε_b < 0
and uses the IP-factorized Coulomb kernel in stripe space to reduce the k-point
convolutions to elementwise products.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import get_kconserv


def _get_phase(cell, kpts: np.ndarray) -> np.ndarray:
    # Mirror fft.isdf_jk.get_phase_factor to avoid depending on fftisdf at import time.
    kpts = np.asarray(kpts)
    kmesh = pbctools.k2gamma.kpts_to_kmesh(cell, kpts - kpts[0])
    is_wrap_around = np.allclose(kpts, cell.get_kpts(kmesh, wrap_around=True))
    assert np.allclose(kpts, cell.get_kpts(kmesh, wrap_around=is_wrap_around))
    phase = pbctools.k2gamma.get_phase(cell, kpts, kmesh, is_wrap_around)[1]
    return phase


def _kpt_to_spc(m_kpt: np.ndarray, phase: np.ndarray) -> np.ndarray:
    nspc, nkpt = phase.shape
    return lib.dot(phase, m_kpt.reshape(nkpt, -1)).reshape(m_kpt.shape)


def _spc_to_kpt(m_spc: np.ndarray, phase: np.ndarray) -> np.ndarray:
    nspc, nkpt = phase.shape
    # Follow fftisdf.fft.isdf_jk.spc_to_kpt convention (no explicit normalization).
    return lib.dot(phase.conj().T, m_spc.reshape(nspc, -1)).reshape(m_spc.shape)

def _k_convolve_via_stripes(
    a_kpt: np.ndarray,
    b_kpt: np.ndarray,
    phase: np.ndarray,
    *,
    direction: str = "k_minus_q",
) -> np.ndarray:
    """k-space convolution via k2gamma stripe space (Algorithm 1 pattern).

    This is the generalized (two-array) version of the k→stripe→Hadamard→k trick
    used throughout FFTISDF:
      [1] kspace->supercell transform: A^s <- A^k,  B^s <- B^k
      [2] Element-wise product:        X^s = A^s * B^s
      [3] supercell->kspace transform: X^k <- X^s

    With PySCF's k2gamma phase (normalized by 1/sqrt(Nk)), the pointwise stripe
    product corresponds to (1/sqrt(Nk)) times a discrete circular "convolution"
    in k-space. The exact q±k convention depends on the sign convention in
    `phase`.

    Empirically, with `phase` as returned by `pbctools.k2gamma.get_phase`, the
    forward transform `phase @ f(k)` yields a stripe product that corresponds to
      x[q] = (1/sqrt(Nk)) * Σ_k a[k] * b[k-q]
    i.e. a correlation-like form. If you need
      x[q] = (1/sqrt(Nk)) * Σ_k a[k] * b[q-k]
    you can set `direction='q_minus_k'`, which uses the complex-conjugated phase
    for the inverse transform.

    For Eq. (13) we need a k-AVERAGE (1/Nk) Σ_k, so we divide by an extra
    sqrt(Nk) here.

    Note: We do NOT call fftisdf.fft.isdf.contract() because that helper also
    performs an additional inner contraction (over AO/grid index) and then
    squares the stripe-space intermediate (special case A=B), whereas Eq. (13)
    needs the product of two (generally different) τ-dependent factors after
    the inner contraction has already been formed.
    """
    direction = str(direction).lower().strip()
    if direction not in {"k_minus_q", "q_minus_k"}:
        raise ValueError("direction must be 'k_minus_q' or 'q_minus_k'")

    nspc, nkpt = phase.shape
    if a_kpt.shape[0] != nkpt or b_kpt.shape[0] != nkpt:
        raise ValueError(f"nkpt mismatch: phase nkpt={nkpt}, a_kpt nkpt={a_kpt.shape[0]}, b_kpt nkpt={b_kpt.shape[0]}")
    a_spc = _kpt_to_spc(a_kpt, phase).reshape(nspc, *a_kpt.shape[1:])
    b_spc = _kpt_to_spc(b_kpt, phase).reshape(nspc, *b_kpt.shape[1:])
    x_spc = a_spc * b_spc
    if direction == "k_minus_q":
        x_kpt = _spc_to_kpt(x_spc, phase).reshape(a_kpt.shape)
    else:
        x_kpt = _spc_to_kpt(x_spc, phase.conj()).reshape(a_kpt.shape)
    return x_kpt / math.sqrt(nkpt)

def _k_convolve_direct(
    a_kpt: np.ndarray,
    b_kpt: np.ndarray,
    kconserv2: np.ndarray,
    *,
    direction: str = "k_minus_q",
) -> np.ndarray:
    """Direct discrete convolution on the k-mesh using kconserv2 mapping.

    Computes one of:
      - direction='k_minus_q' : x[q] = (1/Nk) * Σ_k a[k] * b[k-q]
        where (k-q) index is kconserv2[k, q]
      - direction='q_minus_k' : x[q] = (1/Nk) * Σ_k a[k] * b[q-k]
        where (q-k) index is kconserv2[q, k]

    assuming kconserv2[i,j] = i-j mod mesh.
    """
    direction = str(direction).lower().strip()
    if direction not in {"k_minus_q", "q_minus_k"}:
        raise ValueError("direction must be 'k_minus_q' or 'q_minus_k'")
    nkpt = a_kpt.shape[0]
    if b_kpt.shape[0] != nkpt:
        raise ValueError("a_kpt and b_kpt nkpt mismatch")
    if kconserv2.shape[:2] != (nkpt, nkpt):
        raise ValueError("kconserv2 shape mismatch")
    out = np.zeros((nkpt,) + a_kpt.shape[1:], dtype=np.complex128)
    if direction == "k_minus_q":
        # Vectorized over k for each q via gather of b[ k-q ].
        for q in range(nkpt):
            out[q] = np.sum(a_kpt * b_kpt[kconserv2[:, q]], axis=0)
    else:
        # Vectorized over q for each k via gather of b[ q-k ].
        for k in range(nkpt):
            out += a_kpt[k][None, ...] * b_kpt[kconserv2[:, k]]
    return out / float(nkpt)


def _k_phase_matrix(cell, kpts: np.ndarray) -> np.ndarray:
    """k-mixing matrix used by PySCF k2gamma to form real combinations of
    conjugate k-point pairs (see pyscf.pbc.tools.k2gamma.mo_k2gamma).
    """
    kpts = np.asarray(kpts)
    nkpt = kpts.shape[0]
    k_conj_groups = kpts_helper.group_by_conj_pairs(cell, kpts, wrap_around=True, return_kpts_pairs=False)
    k_phase = np.eye(nkpt, dtype=np.complex128)
    r2x2 = np.array([[1.0, 1j], [1.0, -1j]], dtype=np.complex128) / math.sqrt(2.0)
    pairs = [[k, k_conj] for k, k_conj in k_conj_groups if k_conj is not None and k != k_conj]
    for idx in np.array(pairs, dtype=int):
        # idx is shape (2,), set 2x2 block
        k_phase[idx[:, None], idx] = r2x2
    return k_phase


def _laguerre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    # Gauss-Laguerre quadrature for ∫_0^∞ exp(-x) f(x) dx
    x, w = np.polynomial.laguerre.laggauss(int(n))
    return x.astype(float), w.astype(float)


@dataclass
class GFMP2Result:
    e_corr_os: float
    meta: dict[str, Any]

#
# NOTE:
# This module intentionally keeps only the validated GFMP2(OS) energy path
# (kernel_os). Earlier experimental GF-based RDM construction and full MP2
# (OS+SS) energy were removed to keep the implementation focused and avoid
# accidental use in KLNO workflows.

def _spectrum_range(mo_energy, mo_occ, *, occ_cutoff: float) -> tuple[float, float, float]:
    """Return (Emin, Emax, R=Emax/Emin) as defined in arXiv:2503.20482.

    Emin = global (LUMO - HOMO) across all k-points
    Emax = global (max_e - min_e) across all k-points
    """
    homos = []
    lumos = []
    all_e = []
    for ek, ock in zip(mo_energy, mo_occ):
        ek = np.asarray(ek, dtype=float)
        ock = np.asarray(ock, dtype=float)
        all_e.append(ek)
        occ = ock > occ_cutoff
        if np.any(occ):
            homos.append(float(np.max(ek[occ])))
        if np.any(~occ):
            lumos.append(float(np.min(ek[~occ])))
    if not homos or not lumos:
        raise RuntimeError("Could not determine HOMO/LUMO across k-points for Emin.")
    e_homo = float(np.max(homos))
    e_lumo = float(np.min(lumos))
    Emin = float(e_lumo - e_homo)
    if Emin <= 0:
        raise RuntimeError(f"Non-positive Emin (global gap) = {Emin}.")
    all_e = np.hstack(all_e) if all_e else np.asarray([0.0])
    Emax = float(np.max(all_e) - np.min(all_e))
    if Emax <= 0:
        Emax = Emin
    R = float(Emax / Emin)
    return Emin, Emax, R


def _parse_minimax_file(path: str) -> tuple[float, int, int, float, np.ndarray, np.ndarray]:
    """Parse a minimax 1/x exponential-sum file.

    Returns:
        R_table, k_terms, poly_deg, omega[k'], alpha[k'].
        If poly_deg == 0 and beta[0] is present, we fold it in as an extra
        exponential term with alpha=0 and omega=beta0.
    """
    R_table = None
    k_terms = None
    poly_deg = None
    omega: list[float] = []
    alpha: list[float] = []
    beta0: Optional[float] = None
    err: Optional[float] = None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "{number of exponential terms" in s or "{number of terms" in s:
                parts = s.split("{", 1)[0].split()
                if len(parts) >= 2:
                    k_terms = int(parts[0])
                    poly_deg = int(parts[1])
                continue
            if "{R}" in s:
                R_table = float(s.split("{", 1)[0].split()[0])
                continue
            if "{omega[" in s:
                omega.append(float(s.split("{", 1)[0].split()[0]))
                continue
            if "{alpha[" in s:
                alpha.append(float(s.split("{", 1)[0].split()[0]))
                continue
            if "{beta[0]}" in s:
                beta0 = float(s.split("{", 1)[0].split()[0])
                continue
            if "{error}" in s:
                # Some files also print extra tokens after {error}; ignore them.
                err = float(s.split("{", 1)[0].split()[0])
                continue
    if R_table is None or k_terms is None or poly_deg is None:
        raise RuntimeError(f"Failed to parse minimax header in {path}")
    if len(omega) != k_terms or len(alpha) != k_terms:
        raise RuntimeError(f"Unexpected omega/alpha lengths in {path}: omega={len(omega)} alpha={len(alpha)} k={k_terms}")
    if err is None:
        # Be permissive; some stripped files don't carry the error.
        err = float("inf")

    if poly_deg == 0 and beta0 is not None:
        omega = [float(beta0)] + omega
        alpha = [0.0] + alpha

    return float(R_table), int(k_terms), int(poly_deg), float(err), np.asarray(omega, float), np.asarray(alpha, float)


def _load_minimax_table(
    *,
    table_dir: str,
    max_terms: int,
    R_target: float,
    err_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select and load a minimax table with R >= R_target and k_terms <= max_terms.

    Strategy:
    - Find the smallest available R_table such that R_table >= R_target.
    - Within that R_table, pick the smallest k_terms that achieves err <= err_tol.
      If none achieves err_tol, pick the k_terms with the smallest err.
    """
    import glob

    pat = os.path.join(table_dir, "k*")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise RuntimeError(f"No minimax tables found under {table_dir}")

    candidates: list[tuple[float, int, float, str, int, np.ndarray, np.ndarray]] = []
    for p in paths:
        try:
            R_table, k_terms, poly_deg, err, omega, alpha = _parse_minimax_file(p)
        except Exception:
            continue
        if k_terms > int(max_terms):
            continue
        candidates.append((R_table, int(k_terms), float(err), p, int(poly_deg), omega, alpha))
    if not candidates:
        raise RuntimeError(f"No parseable minimax tables for max_terms={max_terms} under {table_dir}")

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))  # R, k, err

    # Find smallest admissible R_table
    R_tables = sorted({c[0] for c in candidates})
    R_pick = None
    for Rt in R_tables:
        if Rt >= R_target:
            R_pick = Rt
            break
    if R_pick is None:
        R_pick = R_tables[-1]

    same_R = [c for c in candidates if c[0] == R_pick]
    # Prefer smallest k meeting tolerance; else smallest err
    tol_ok = [c for c in same_R if c[2] <= err_tol]
    if tol_ok:
        tol_ok.sort(key=lambda x: (x[1], x[2]))
        chosen = tol_ok[0]
    else:
        same_R.sort(key=lambda x: (x[2], x[1]))
        chosen = same_R[0]

    R_table, k_terms, err, p, poly_deg, omega, alpha = chosen
    meta = {
        "minimax_path": p,
        "R_target": float(R_target),
        "R_table": float(R_table),
        "minimax_err": float(err),
        "poly_deg": int(poly_deg),
        "max_terms": int(max_terms),
        "k_terms": int(k_terms),
        "n_terms_eff": int(omega.size),
        "err_tol": float(err_tol),
    }
    return omega, alpha, meta


def _default_minimax_dir() -> str:
    """Default location of Hackbusch/Takatsuka minimax tables.

    The user keeps the `1_x/` folder next to this module.
    """
    return os.path.join(os.path.dirname(__file__), "1_x")


def kernel_os(
    kmf,
    *,
    n_tau: int = 32,
    tau_scale: Optional[float] = None,
    occ_cutoff: float = 1e-6,
    tail_eps: float = 1e-12,
    quadrature: str = "minimax",
    minimax_dir: Optional[str] = None,
    integrand: str = "trace_square",
    use_k_phase: bool = False,
    symmetrize_k_conj: bool = True,
    convolution: str = "stripe",
    convolution_direction: str = "q_minus_k",
    verbose: Optional[int] = None,
) -> GFMP2Result:
    """
    Compute the opposite-spin (direct) MP2 correlation energy using FFT-ISDF.

    Args:
        kmf: KRHF object with kmf.with_df being FFT-ISDF and built.
        n_tau: Quadrature budget. For minimax, this is the max number of exponential terms.
        tau_scale: Positive scaling α for τ = x/α. If None, a heuristic is used.
        occ_cutoff: Occupation threshold to define occupied orbitals.
        verbose: Logger verbosity override.
    """
    log = logger.new_logger(kmf, verbose if verbose is not None else getattr(kmf, "verbose", 0))

    df = getattr(kmf, "with_df", None)
    if df is None or not hasattr(df, "inpv_kpt") or not hasattr(df, "coul_kpt"):
        raise RuntimeError("GFMP2 requires kmf.with_df with attributes inpv_kpt and coul_kpt (FFT-ISDF).")

    inpv_kpt = np.asarray(df.inpv_kpt)
    coul_kpt = np.asarray(df.coul_kpt)

    kpts = getattr(kmf, "kpts", None)
    if kpts is None:
        kpts = getattr(df, "kpts", None)
    if kpts is None:
        raise RuntimeError("Could not determine kpts from kmf/df.")
    kpts = np.asarray(kpts)

    nkpt, nip, nao = inpv_kpt.shape
    assert coul_kpt.shape[:2] == (nkpt, nip)

    phase = _get_phase(kmf.cell, kpts)
    nspc, nkpt2 = phase.shape
    if nkpt2 != nkpt:
        raise RuntimeError(f"Phase shape mismatch: phase has nkpt={nkpt2}, but inpv_kpt nkpt={nkpt}")
    if use_k_phase:
        k_phase = _k_phase_matrix(kmf.cell, kpts)
        phase_eff = lib.dot(phase, k_phase)
    else:
        phase_eff = phase

    # Coulomb kernel W^q in the IP representation (Eq. 9 / 13 of Yang et al.).
    # We keep it in k-space (q-index) and use k2gamma transforms only to
    # accelerate the k-convolution needed to build Q^q.

    # MO projection onto IPs
    mo_coeff = kmf.mo_coeff
    mo_energy = kmf.mo_energy
    mo_occ = kmf.mo_occ
    if mo_coeff is None or mo_energy is None or mo_occ is None:
        raise RuntimeError("kmf is missing mo_coeff/mo_energy/mo_occ (run SCF first).")

    # Quadrature setup
    #
    # - "laguerre": Gauss-Laguerre on ∫_0^∞ e^{-x} f(x) dx with τ = x/tau_scale
    # - "minimax": Hackbusch minimax exponential-sum for 1/x on [Emin,Emax]
    quadrature = str(quadrature).lower().strip()
    if quadrature not in {"laguerre", "minimax"}:
        raise ValueError(f"Unknown quadrature='{quadrature}'. Use 'laguerre' or 'minimax'.")

    integrand = str(integrand).lower().strip()
    if integrand not in {"trace_square", "frob"}:
        raise ValueError("integrand must be 'trace_square' or 'frob'")

    convolution = str(convolution).lower().strip()
    if convolution not in {"stripe", "direct"}:
        raise ValueError("convolution must be 'stripe' or 'direct'")

    convolution_direction = str(convolution_direction).lower().strip()
    if convolution_direction not in {"k_minus_q", "q_minus_k"}:
        raise ValueError("convolution_direction must be 'k_minus_q' or 'q_minus_k'")

    kconserv2 = None
    if convolution == "direct":
        # Prefer DF-provided mapping; otherwise compute from kpts.
        kconserv2 = getattr(df, "kconserv2", None)
        if kconserv2 is None:
            kconserv = get_kconserv(kmf.cell, kpts)
            kconserv2 = kconserv[:, :, 0].T
        kconserv2 = np.asarray(kconserv2, dtype=np.int32)

    Emin = Emax = R = None
    minimax_meta: dict[str, Any] = {}
    if quadrature == "minimax":
        Emin, Emax, R = _spectrum_range(mo_energy, mo_occ, occ_cutoff=occ_cutoff)
        table_dir = minimax_dir or _default_minimax_dir()
        if not os.path.isdir(table_dir):
            raise RuntimeError(f"minimax quadrature requested but 1_x table_dir does not exist: {table_dir}")
        omega, alpha, minimax_meta = _load_minimax_table(table_dir=table_dir, max_terms=int(n_tau), R_target=float(R))
        # Scale to interval [Emin, Emax]: w̃ = w/Emin,  t̃ = t/Emin
        tau_nodes = alpha / float(Emin)
        tau_weights = omega / float(Emin)
        # no tail cutoff needed (finite sum), but we keep tau_cut for diagnostics
        tau_cut = None
    else:
        # Heuristic τ scaling (tau = x/tau_scale)
        if tau_scale is None:
            gaps = []
            for k in range(nkpt):
                occ = mo_occ[k] > occ_cutoff
                e_o = np.asarray(mo_energy[k])[occ]
                e_v = np.asarray(mo_energy[k])[~occ]
                if len(e_o) and len(e_v):
                    gaps.append(float(np.median(e_v) - np.median(e_o)))
            gap_med = float(np.median(gaps)) if gaps else 1.0
            tau_scale = max(0.2, min(5.0, 1.0 / max(1e-6, float(gap_med))))
        x, w = _laguerre_nodes_weights(n_tau)
        tau_nodes = x / float(tau_scale)
        # ∫ g(t) dt ≈ Σ wi * exp(xi) * g(xi/α) / α
        tau_weights = (w * np.exp(x)) / float(tau_scale)
        # Long-τ cutoff (optional)
        tau_cut = None
        if tail_eps is not None and tail_eps > 0.0:
            gaps_hl = []
            for k in range(nkpt):
                occ = np.asarray(mo_occ[k]) > occ_cutoff
                eo = np.asarray(mo_energy[k])[occ]
                ev = np.asarray(mo_energy[k])[~occ]
                if eo.size and ev.size:
                    gaps_hl.append(float(np.min(ev) - np.max(eo)))
            if gaps_hl:
                gap_min = max(1e-8, float(np.min(gaps_hl)))
                tau_cut = float(math.log(1.0 / float(tail_eps)) / gap_min)

    # Precompute IP-space MO values per k (nip, nmo)
    xmo_kpt = [lib.dot(inpv_kpt[k], mo_coeff[k]) for k in range(nkpt)]

    # Prepare occ/vir slices per k
    occ_idx = []
    vir_idx = []
    e_occ = []
    e_vir = []
    x_occ = []
    x_vir = []
    homos = []
    lumos = []
    for k in range(nkpt):
        occ = np.asarray(mo_occ[k]) > occ_cutoff
        occ_idx.append(np.where(occ)[0])
        vir_idx.append(np.where(~occ)[0])
        e = np.asarray(mo_energy[k])
        eo = e[occ]
        ev = e[~occ]
        e_occ.append(eo)
        e_vir.append(ev)
        if eo.size:
            homos.append(float(np.max(eo)))
        if ev.size:
            lumos.append(float(np.min(ev)))
        x_occ.append(np.asarray(xmo_kpt[k][:, occ], dtype=np.complex128))
        x_vir.append(np.asarray(xmo_kpt[k][:, ~occ], dtype=np.complex128))

    # Chemical potential shift to stabilize exp factors (mathematically cancels out).
    # Choose midgap μ based on median(HOMO) and median(LUMO) across k.
    if homos and lumos:
        mu = 0.5 * (float(np.median(homos)) + float(np.median(lumos)))
    else:
        mu = 0.0

    # Imag-time quadrature (Eq. 12 / 13)
    e_corr = 0.0
    n_used = 0
    for tau, quad_weight in zip(tau_nodes, tau_weights):
        tau = float(tau)
        if tau_cut is not None and tau > tau_cut:
            break
        quad_weight = float(quad_weight)
        n_used += 1

        # Build imaginary-time Green's function factors in IP representation.
        # We use a symmetric split in the exponent to keep both factors decaying
        # with a mid-gap chemical potential shift, and then use the trace of the
        # square to recover the full MP2 denominator dependence.
        #
        # This corresponds to evaluating (up to conventions) P^q(τ) in Eq. (13).
        Gp_kpt = np.empty((nkpt, nip, nip), dtype=np.complex128)  # "occupied side" at +tau
        Gm_kpt = np.empty((nkpt, nip, nip), dtype=np.complex128)  # "virtual side" at +tau
        for k in range(nkpt):
            if x_occ[k].size:
                wocc = np.exp(0.5 * (np.asarray(e_occ[k], float) - mu) * tau)
                a = x_occ[k] * wocc[None, :]
                Gp_kpt[k] = a @ a.conj().T
            else:
                Gp_kpt[k] = 0.0

            if x_vir[k].size:
                wvir = np.exp(-0.5 * (np.asarray(e_vir[k], float) - mu) * tau)
                b = x_vir[k] * wvir[None, :]
                Gm_kpt[k] = b @ b.conj().T
            else:
                Gm_kpt[k] = 0.0

        # Enforce time-reversal / conjugation symmetry between k and -k points.
        # For dense meshes (esp. odd kmeshes), the SCF eigenvectors can have
        # arbitrary phase/rotation within degenerate subspaces, which can spoil
        # the expected real stripe-space representation. Symmetrizing here
        # makes the k2gamma transform consistent with FFTISDF's assumptions.
        if symmetrize_k_conj:
            for k, k_conj in kpts_helper.group_by_conj_pairs(kmf.cell, kpts, wrap_around=True, return_kpts_pairs=False):
                if k_conj is None or k_conj == k:
                    continue
                gp = 0.5 * (Gp_kpt[k] + Gp_kpt[k_conj].conj())
                gm = 0.5 * (Gm_kpt[k] + Gm_kpt[k_conj].conj())
                Gp_kpt[k] = gp
                Gp_kpt[k_conj] = gp.conj()
                Gm_kpt[k] = gm
                Gm_kpt[k_conj] = gm.conj()

        # Eq. (13) via Algorithm 1 (k→stripe→Hadamard→k).
        # Q^q_{IK}(τ) = (1/Nk) Σ_k Gp^k_{IK}(τ) Gm^{q-k}_{IK}(τ)
        if convolution == "direct":
            Q_kpt = _k_convolve_direct(Gp_kpt, Gm_kpt, kconserv2, direction=convolution_direction).reshape(nkpt, nip, nip)
        else:
            Q_kpt = _k_convolve_via_stripes(Gp_kpt, Gm_kpt, phase_eff, direction=convolution_direction).reshape(nkpt, nip, nip)

        # P^q(τ) = Q^q(τ) W^q  (right-multiplication over K)
        P_kpt = np.einsum("qik,qkj->qij", Q_kpt, coul_kpt, optimize=True)

        # Energy integrand: Σ_q tr[P^q(τ) P^q(-τ)].
        #
        # In exact arithmetic (with time-reversal symmetry), P is Hermitian for the
        # imaginary-frequency form and the two contractions coincide:
        #   tr(P^2) == tr(P P†) == ||P||_F^2
        # For finite precision / approximate factorizations, enforce the stable
        # Hermitian contraction if requested.
        if integrand == "frob":
            integ_val = float(np.einsum("qij,qij->", P_kpt, P_kpt.conj()).real)
        else:
            integ_val = float(np.einsum("qij,qji->", P_kpt, P_kpt).real)

        # Eq. (12): EOS = - Σ_q ∫ dτ (1/Nk) tr[P^q(τ) P^q(-τ)]
        e_corr += (-1.0 / float(nkpt)) * quad_weight * integ_val

        log.debug(
            "GFMP2 tau=%g integrand=%g contrib=%g",
            tau,
            integ_val,
            (-1.0 / float(nkpt)) * quad_weight * integ_val,
        )

    res = GFMP2Result(
        e_corr_os=float(e_corr),
        meta={
            "n_tau": int(n_tau),
            "n_tau_used": int(n_used),
            "quadrature": str(quadrature),
            "integrand": str(integrand),
            "use_k_phase": bool(use_k_phase),
            "symmetrize_k_conj": bool(symmetrize_k_conj),
            "convolution": str(convolution),
            "convolution_direction": str(convolution_direction),
            "tau_scale": None if tau_scale is None else float(tau_scale),
            "Emin": None if Emin is None else float(Emin),
            "Emax": None if Emax is None else float(Emax),
            "R": None if R is None else float(R),
            "tau_cut": None if tau_cut is None else float(tau_cut),
            "tail_eps": float(tail_eps),
            "nkpt": int(nkpt),
            "nip": int(nip),
            "nspc": int(nspc),
            **minimax_meta,
        },
    )
    return res


#
# NOTE: Full MP2 (OS+SS) energy and GF-based RDM/self-energy helpers were removed.

