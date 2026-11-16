#!/usr/bin/env python3
"""
cmb_squealer.py — Squealer for the Planck 2018 CMB TT power spectrum.

Augments the Planck posterior with a single pseudo-data point (ℓ*, D_ℓ*)
and finds the new posterior mode θ*, reporting parameter shifts and fit cost.

Usage:
    python cmb_squealer.py --ell 464 --dl 2225     # panel (c) of the paper figure
    python cmb_squealer.py --ell 464 --test        # D_ℓ* = best fit → cost, shifts ≈ 0

Writes figures/cmb_squealer_ell{ℓ*}_Dl{D_ℓ*}.pdf/png and
results/cmb_squealer_ell{ℓ*}_Dl{D_ℓ*}.json (read by cmb_squealer_multipanel.py).
"""

import numpy as np
import sys, os, time, argparse, json
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from planck_paths import (import_clik, PLIK_LITE_TT, COMMANDER_TT, LOWELL_BANDPOWERS,
                          PLIK_LITE_BANDPOWERS, FIGDIR, RESULTSDIR)

# ── pseudo-data ───────────────────────────────────────────────────────────────
# Paper figure:
#   --ell 22  --dl 600    low-ell power deficit
#   --ell 464 --dl 2225   +3.1σ excursion in trough between 1st and 2nd acoustic peaks
#   --ell 220 --dl 6000   above first acoustic peak (demo)
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ell", type=int, default=464, help="multipole ℓ* of the pseudo-data point")
parser.add_argument("--dl", type=float, default=2225.0, help="pseudo-data value D_ℓ* in μK²")
parser.add_argument("--test", action="store_true",
                    help="set D_ℓ* to the best-fit value; cost and shifts should be ≈ 0")
args = parser.parse_args()
ELL_STAR, DL_STAR, TEST_MODE = args.ell, args.dl, args.test

clik = import_clik()
import camb

lkl_hi  = clik.clik(str(PLIK_LITE_TT))
lkl_low = clik.clik(str(COMMANDER_TT))
LMAX_HI  = lkl_hi.get_lmax()[0]    # 2508
LMAX_LOW = lkl_low.get_lmax()[0]   # 29

# ── Planck 2018 plikTTTEEE+lowl+simall best-fit parameters ───────────────────
PARAM_NAMES = ["H0", "ombh2", "omch2", "tau", "ns", "As", "A_planck"]
THETA_HAT   = np.array([67.32, 0.02238, 0.1201, 0.0543, 0.9660, 2.101e-9, 1.0])
# Step sizes for numerical derivatives — larger than strict σ to avoid noise
PARAM_SCALES = np.array([0.5, 1e-4, 1e-3, 1e-2, 3e-3, 5e-11, 1e-3])
# Hard bounds to prevent CAMB failure during optimization
PARAM_BOUNDS_LO = np.array([62.,  0.018, 0.09,  0.02, 0.92, 1.5e-9, 0.9])
PARAM_BOUNDS_HI = np.array([80.,  0.028, 0.15,  0.12, 1.02, 3.0e-9, 1.1])

# ── CAMB ──────────────────────────────────────────────────────────────────────
def camb_dl(theta6):
    """D_ell[0..LMAX_HI] in μK² (lensed TT) from 6 cosmo params."""
    H0, ombh2, omch2, tau, ns, As = theta6
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(LMAX_HI + 50, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers  = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    return powers["total"][:LMAX_HI + 1, 0]   # shape (LMAX_HI+1,)

def dl_to_cl(Dl):
    ell = np.arange(len(Dl), dtype=float)
    Cell = np.zeros_like(Dl)
    Cell[2:] = Dl[2:] * 2 * np.pi / (ell[2:] * (ell[2:] + 1))
    return Cell

# Gaussian prior on tau from Planck 2018 SimAll EE (lowE), equivalent to having polarisation data
TAU_PRIOR_MU    = 0.0543
TAU_PRIOR_SIGMA = 0.0073

# ── log posterior ─────────────────────────────────────────────────────────────
def log_posterior(params):
    """Planck log posterior: plik_lite TT + Commander low-ell + tau prior (lowE)."""
    theta6, A_planck = params[:6], params[6]
    tau = theta6[3]
    try:
        Cell = dl_to_cl(camb_dl(theta6))
        logL = (lkl_hi( np.append(Cell[:LMAX_HI  + 1], A_planck))[0] +
                lkl_low(np.append(Cell[:LMAX_LOW + 1], A_planck))[0])
        logL += -0.5 * ((tau - TAU_PRIOR_MU) / TAU_PRIOR_SIGMA)**2
        return logL
    except Exception:
        return -1e30

# ── find exact TT-only posterior mode ────────────────────────────────────────
print("Finding TT-only posterior mode (Powell)...")
t0 = time.time()

def neg_logp_scaled(x):
    params = THETA_HAT + x * PARAM_SCALES
    if np.any(params < PARAM_BOUNDS_LO) or np.any(params > PARAM_BOUNDS_HI):
        return 1e30
    return -log_posterior(params)

mode_result = minimize(neg_logp_scaled, np.zeros(len(THETA_HAT)),
                       method="Powell",
                       options={"xtol": 1e-4, "ftol": 0.01, "maxiter": 500})
THETA_HAT = THETA_HAT + mode_result.x * PARAM_SCALES
logp_hat  = -mode_result.fun
print(f"  logL(θ̂) = {logp_hat:.3f}  [{time.time()-t0:.1f}s]")
print(f"  Mode: " + "  ".join(f"{n}={v:.5g}" for n,v in zip(PARAM_NAMES, THETA_HAT)))

# ── full Fisher matrix (6×6 cosmo params; A_planck doesn't affect D_ell) ─────
n_cosmo = 6
h = PARAM_SCALES[:n_cosmo]
theta6_hat = THETA_HAT[:n_cosmo]

def logp6(theta6):
    return log_posterior(np.append(theta6, 1.0))

n_F = 1 + 2 * n_cosmo + 4 * (n_cosmo * (n_cosmo - 1) // 2)
print(f"\nComputing full Fisher matrix ({n_F} CAMB calls, ~{n_F * 0.75:.0f}s)...")
t0 = time.time()

logp0 = logp6(theta6_hat)
F = np.zeros((n_cosmo, n_cosmo))

for i in range(n_cosmo):
    tp, tm = theta6_hat.copy(), theta6_hat.copy()
    tp[i] += h[i]; tm[i] -= h[i]
    F[i, i] = -(logp6(tp) + logp6(tm) - 2 * logp0) / h[i]**2

for i in range(n_cosmo):
    for j in range(i + 1, n_cosmo):
        ppp = theta6_hat.copy(); ppp[i] += h[i]; ppp[j] += h[j]
        ppm = theta6_hat.copy(); ppm[i] += h[i]; ppm[j] -= h[j]
        pmp = theta6_hat.copy(); pmp[i] -= h[i]; pmp[j] += h[j]
        pmm = theta6_hat.copy(); pmm[i] -= h[i]; pmm[j] -= h[j]
        F[i, j] = F[j, i] = -(
            logp6(ppp) - logp6(ppm) - logp6(pmp) + logp6(pmm)
        ) / (4 * h[i] * h[j])

eigvals = np.linalg.eigvalsh(F)
print(f"  done [{time.time()-t0:.1f}s].  Min eigenvalue: {eigvals.min():.3g}")
if eigvals.min() <= 0:
    reg = -eigvals.min() * 1.1 + 1e-6
    F += np.eye(n_cosmo) * reg
    frac = reg / np.diag(F).min() * 100
    print(f"  Minimal regularisation: +{reg:.3g} on diagonal ({frac:.2f}% of smallest diag)")
Finv = np.linalg.inv(F)
print("  Marginal 1σ (including correlations):")
for name, std in zip(PARAM_NAMES[:n_cosmo], np.sqrt(np.diag(Finv))):
    print(f"    {name:8s}: {std:.3g}")

# ── σ* from posterior predictive uncertainty at ell_star ──────────────────────
print(f"\nComputing σ* at ℓ* = {ELL_STAR}  ({2 * n_cosmo} CAMB calls)...")
t0 = time.time()

grad_dl = np.zeros(n_cosmo)
for i in range(n_cosmo):
    tp, tm = theta6_hat.copy(), theta6_hat.copy()
    tp[i] += h[i]; tm[i] -= h[i]
    grad_dl[i] = (camb_dl(tp)[ELL_STAR] - camb_dl(tm)[ELL_STAR]) / (2 * h[i])

sigma_pred = np.sqrt(grad_dl @ Finv @ grad_dl)
sigma_star = 0.5 * sigma_pred
Dl_hat_at_star = camb_dl(theta6_hat)[ELL_STAR]

if TEST_MODE:
    DL_STAR = Dl_hat_at_star
    print("  TEST MODE: DL_STAR set to best-fit value — expect cost≈0, shifts≈0")

print(f"  [{time.time()-t0:.1f}s]")
print(f"  D_ℓ*(θ̂)  = {Dl_hat_at_star:.1f} μK²")
print(f"  σ_pred    = {sigma_pred:.2f} μK²")
print(f"  σ*        = {sigma_star:.2f} μK²  (= 0.5 × σ_pred)")
print(f"  DL_STAR   = {DL_STAR:.1f} μK²  (pseudo-data)")
print(f"  pull      = {(DL_STAR - Dl_hat_at_star) / sigma_star:.1f} σ*")

# ── augmented posterior optimization ─────────────────────────────────────────
print(f"\nOptimizing augmented posterior (Powell, rescaled params)...")
t0 = time.time()

def neg_logp_star(params):
    theta6, A_planck = params[:6], params[6]
    try:
        Dl    = camb_dl(theta6)
        Cell  = dl_to_cl(Dl)
        logL  = (lkl_hi( np.append(Cell[:LMAX_HI  + 1], A_planck))[0] +
                 lkl_low(np.append(Cell[:LMAX_LOW + 1], A_planck))[0])
        pseudo = -0.5 * (DL_STAR - Dl[ELL_STAR])**2 / sigma_star**2
        return -(logL + pseudo)
    except Exception:
        return 1e30

# rescale so all parameters are O(1); enforce bounds via penalty
x0 = np.zeros(len(THETA_HAT))

def neg_logp_star_scaled(x):
    params = THETA_HAT + x * PARAM_SCALES
    if np.any(params < PARAM_BOUNDS_LO) or np.any(params > PARAM_BOUNDS_HI):
        return 1e30
    return neg_logp_star(params)

call_count = [0]
def neg_logp_star_verbose(x):
    call_count[0] += 1
    if call_count[0] % 10 == 0:
        print(f"    [{call_count[0]} calls, {time.time()-t0:.0f}s]", flush=True)
    return neg_logp_star_scaled(x)

result = minimize(
    neg_logp_star_verbose, x0,
    method="Powell",
    options={"xtol": 1e-3, "ftol": 0.05, "maxiter": 500},
)
theta_star = THETA_HAT + result.x * PARAM_SCALES
logp_star_orig = log_posterior(theta_star)
cost = logp_hat - logp_star_orig

print(f"\n── Results ──────────────────────────────────────────────────────────")
print(f"  cost = logp(θ̂) − logp(θ*) = {logp_hat:.3f} − {logp_star_orig:.3f} = {cost:.2f}")
print(f"  ({call_count[0]} function evaluations, {time.time()-t0:.1f}s)")
print(f"\n  Parameter shifts  (in units of σ̂):")
sigmas = np.sqrt(np.diag(np.pad(Finv, ((0,1),(0,1)))))  # add A_planck row
sigmas[-1] = PARAM_SCALES[-1]
for name, t_hat, t_star, sc in zip(PARAM_NAMES, THETA_HAT, theta_star, PARAM_SCALES):
    shift = t_star - t_hat
    print(f"    {name:10s}: {t_hat:12.5g}  →  {t_star:12.5g}  "
          f"(Δ = {shift:+.3g})")

# ── figure ────────────────────────────────────────────────────────────────────
print("\nGenerating figure...")

Dl_hat  = camb_dl(theta6_hat)
Dl_star = camb_dl(theta_star[:6])
ell_t   = np.arange(len(Dl_hat))

# load data for plotting (from cmb_spectrum.py)
full    = np.loadtxt(LOWELL_BANDPOWERS)
lo_mask = full[:, 0] < 30
ell_lo, Dl_lo     = full[lo_mask, 0], full[lo_mask, 1]
eDl_lo_m, eDl_lo_p = full[lo_mask, 2], full[lo_mask, 3]

dat    = np.loadtxt(PLIK_LITE_BANDPOWERS)
ell_hi = dat[:, 0]
norm   = ell_hi * (ell_hi + 1) / (2 * np.pi)
Dl_hi  = norm * dat[:, 1]
sDl_hi = norm * dat[:, 2]

# custom x-scale (linear ≤30, log above)
SPLIT = 30.0
def forward(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x <= SPLIT, x, SPLIT * (1 + np.log(np.maximum(x, 1e-10) / SPLIT)))
def inverse(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= SPLIT, y, SPLIT * np.exp(y / SPLIT - 1))

fig = plt.figure(figsize=(11, 6))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.18,
                       width_ratios=[3, 1])
ax_spec = fig.add_subplot(gs[0, 0])
ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
ax_par  = fig.add_subplot(gs[:, 1])

for ax in [ax_spec, ax_res]:
    ax.set_xscale("function", functions=(forward, inverse))
    ax.axvline(SPLIT, color="gray", lw=0.7, ls=":", alpha=0.7)

# spectrum panel
ax_spec.errorbar(ell_lo, Dl_lo, yerr=[eDl_lo_m, eDl_lo_p],
                 fmt="none", elinewidth=0.9, color="steelblue", alpha=0.85)
ax_spec.errorbar(ell_hi, Dl_hi, sDl_hi,
                 fmt="none", elinewidth=0.5, color="steelblue", alpha=0.6)
ax_spec.plot(ell_t, Dl_hat,  color="firebrick", lw=1.3, zorder=3,
             label=r"Best-fit $\hat\theta$")
ax_spec.plot(ell_t, Dl_star, color="darkorange", lw=1.3, ls="--", zorder=3,
             label=r"Perturbed $\theta^*$")
ax_spec.scatter([ELL_STAR], [DL_STAR], marker="*", s=200, color="darkorange",
                zorder=5, label=fr"Pseudo-data $\ell^*={ELL_STAR}$")
ax_spec.errorbar([ELL_STAR], [DL_STAR], yerr=[[sigma_star], [sigma_star]],
                 fmt="none", elinewidth=2, color="darkorange", zorder=4, capsize=4)
ax_spec.set_ylabel(r"$D_\ell^{TT}\ [\mu\mathrm{K}^2]$", fontsize=12)
ax_spec.set_ylim(0, max(7000, DL_STAR * 1.1))
ax_spec.legend(fontsize=10, loc="upper left")

# residuals panel
res_lo = (Dl_lo - np.interp(ell_lo, ell_t, Dl_hat)) / ((eDl_lo_m + eDl_lo_p) / 2)
res_hi = (Dl_hi - np.interp(ell_hi, ell_t, Dl_hat)) / sDl_hi
res_lo_s = (Dl_lo - np.interp(ell_lo, ell_t, Dl_star)) / ((eDl_lo_m + eDl_lo_p) / 2)
res_hi_s = (Dl_hi - np.interp(ell_hi, ell_t, Dl_star)) / sDl_hi

ax_res.axhline(0,   color="firebrick",  lw=1.0)
ax_res.axhline(0,   color="darkorange", lw=1.0, ls="--")
ax_res.axhline( 2,  color="gray", lw=0.5, ls="--")
ax_res.axhline(-2,  color="gray", lw=0.5, ls="--")
ax_res.scatter(ell_lo, res_lo, s=10, color="firebrick", zorder=3, alpha=0.7)
ax_res.scatter(ell_hi, res_hi, s=3,  color="firebrick", zorder=3, alpha=0.6)
ax_res.scatter(ell_lo, res_lo_s, s=10, color="darkorange", zorder=3, alpha=0.7)
ax_res.scatter(ell_hi, res_hi_s, s=3,  color="darkorange", zorder=3, alpha=0.6)
ax_res.set_ylabel(r"$(D_\ell - D_\ell^{\rm th})/\sigma$", fontsize=11)
ax_res.set_xlabel(r"Multipole $\ell$", fontsize=12)
ax_res.set_ylim(-5, 5)

xticks = [2, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]
for ax in [ax_spec, ax_res]:
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_xlim(2, 2508)

# parameter shift panel
shifts   = theta_star - THETA_HAT
sigma_fi = np.append(np.sqrt(np.diag(Finv)), PARAM_SCALES[-1])
shifts_norm = shifts / sigma_fi

colors = ["tomato" if s > 0 else "steelblue" for s in shifts_norm]
ypos   = np.arange(len(PARAM_NAMES))
ax_par.barh(ypos, shifts_norm, color=colors, alpha=0.8, height=0.6)
ax_par.axvline(0, color="black", lw=0.8)
ax_par.axvline( 1, color="gray", lw=0.7, ls="--", alpha=0.7)
ax_par.axvline(-1, color="gray", lw=0.7, ls="--", alpha=0.7)
ax_par.set_yticks(ypos)
ax_par.set_yticklabels(PARAM_NAMES, fontsize=10)
ax_par.set_xlabel(r"$(\theta^* - \hat\theta)\,/\,\hat\sigma$", fontsize=10)
ax_par.set_title(f"Parameter shifts\ncost = {cost:.2f}", fontsize=10)
ax_par.set_xlim(-max(3, abs(shifts_norm).max() * 1.3),
                 max(3, abs(shifts_norm).max() * 1.3))

fig.suptitle(
    fr"CMB Squealer: pseudo-data $\ell^*={ELL_STAR}$, "
    fr"$D_{{\ell^*}}={DL_STAR:.0f}\ \mu\mathrm{{K}}^2$, "
    fr"$\sigma^*={sigma_star:.1f}\ \mu\mathrm{{K}}^2$",
    fontsize=12, y=1.01,
)

tag = f"ell{ELL_STAR}_Dl{int(DL_STAR)}" + ("_test" if TEST_MODE else "")
FIGDIR.mkdir(exist_ok=True)
plt.savefig(FIGDIR / f"cmb_squealer_{tag}.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / f"cmb_squealer_{tag}.png", dpi=150, bbox_inches="tight")
print(f"Saved figures/cmb_squealer_{tag}.pdf/.png")

# ── results for cmb_squealer_multipanel.py ───────────────────────────────────
RESULTSDIR.mkdir(exist_ok=True)
results = dict(
    ELL_STAR=ELL_STAR, DL_STAR=float(DL_STAR), sigma_star=float(sigma_star),
    cost=float(cost), logp_hat=float(logp_hat), logp_star=float(logp_star_orig),
    param_names=PARAM_NAMES, theta_hat=THETA_HAT.tolist(),
    theta_star=theta_star.tolist(), sigma_fisher=sigma_fi.tolist(),
)
with open(RESULTSDIR / f"cmb_squealer_{tag}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results/cmb_squealer_{tag}.json")
