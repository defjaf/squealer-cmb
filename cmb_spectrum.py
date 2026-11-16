"""
cmb_spectrum.py — Planck 2018 TT band-powers with the published best-fit ΛCDM spectrum.
Needs only the Planck data files (no clik or CAMB). Writes figures/cmb_spectrum.pdf/png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planck_paths import LOWELL_BANDPOWERS, PLIK_LITE_BANDPOWERS, BESTFIT_THEORY, FIGDIR

# --- low-ell (ell=2..29): published Commander per-ell band-powers ---
full     = np.loadtxt(LOWELL_BANDPOWERS)
mask_lo  = full[:, 0] < 30
ell_lo   = full[mask_lo, 0]
Dl_lo    = full[mask_lo, 1]
eDl_lo_m = full[mask_lo, 2]   # lower 1-sigma
eDl_lo_p = full[mask_lo, 3]   # upper 1-sigma

# --- high-ell (ell>=30): plik_lite binned band-powers ---
dat    = np.loadtxt(PLIK_LITE_BANDPOWERS)
ell_hi = dat[:, 0]
norm   = ell_hi * (ell_hi + 1) / (2 * np.pi)
Dl_hi  = norm * dat[:, 1]
sDl_hi = norm * dat[:, 2]

# --- best-fit theory (columns: ell, D_TT, D_EE, D_BB, D_TE, D_PP) ---
bf    = np.loadtxt(BESTFIT_THEORY)
ell_t = bf[:, 0]
Dl_t  = bf[:, 1]

# theory at each data point by interpolation
Dl_th_lo = np.interp(ell_lo, ell_t, Dl_t)
Dl_th_hi = np.interp(ell_hi, ell_t, Dl_t)

# residuals (symmetric mean error for low-ell panel)
res_lo = (Dl_lo - Dl_th_lo) / ((eDl_lo_m + eDl_lo_p) / 2)
res_hi = (Dl_hi - Dl_th_hi) / sDl_hi

# --- custom x-scale: linear for ell<=30, log above ---
SPLIT = 30.0

def forward(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x <= SPLIT, x, SPLIT * (1 + np.log(np.maximum(x, 1e-10) / SPLIT)))

def inverse(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= SPLIT, y, SPLIT * np.exp(y / SPLIT - 1))

# --- figure ---
fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
                         sharex=True)
ax_top, ax_bot = axes

for ax in axes:
    ax.set_xscale("function", functions=(forward, inverse))
    ax.axvline(SPLIT, color="gray", lw=0.7, ls=":", alpha=0.8)

# top: spectrum
ax_top.errorbar(ell_lo, Dl_lo, yerr=[eDl_lo_m, eDl_lo_p],
                fmt="none", elinewidth=0.9, color="steelblue", alpha=0.85)
ax_top.errorbar(ell_hi, Dl_hi, sDl_hi,
                fmt="none", elinewidth=0.5, color="steelblue", alpha=0.6)
ax_top.plot(ell_t, Dl_t, color="firebrick", lw=1.3, zorder=3,
            label="Best-fit $\\Lambda$CDM")
ax_top.set_ylabel(r"$D_\ell^{TT}\ [\mu\mathrm{K}^2]$", fontsize=13)
ax_top.set_ylim(0, 6500)
ax_top.legend(fontsize=11, loc="upper left")

# bottom: residuals
ax_bot.axhline(0,  color="firebrick", lw=1.0)
ax_bot.axhline( 2, color="gray", lw=0.6, ls="--")
ax_bot.axhline(-2, color="gray", lw=0.6, ls="--")
ax_bot.errorbar(ell_lo, res_lo, np.ones_like(res_lo),
                fmt="none", elinewidth=0.9, color="steelblue", alpha=0.85)
ax_bot.scatter(ell_lo, res_lo, s=10, color="steelblue", zorder=3)
ax_bot.errorbar(ell_hi, res_hi, np.ones_like(res_hi),
                fmt="none", elinewidth=0.5, color="steelblue", alpha=0.6)
ax_bot.scatter(ell_hi, res_hi, s=3, color="steelblue", alpha=0.8, zorder=3)
ax_bot.set_ylabel(r"$(D_\ell - D_\ell^{\rm th})/\sigma$", fontsize=12)
ax_bot.set_xlabel(r"Multipole $\ell$", fontsize=13)
ax_bot.set_ylim(-5, 5)

# ticks
xticks = [2, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]
for ax in axes:
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_xlim(2, 2508)

fig.suptitle("CMB TT Power Spectrum — Planck 2018", fontsize=13, y=0.99)
FIGDIR.mkdir(exist_ok=True)
plt.savefig(FIGDIR / "cmb_spectrum.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "cmb_spectrum.png", dpi=150, bbox_inches="tight")
print("Saved figures/cmb_spectrum.pdf/.png")
