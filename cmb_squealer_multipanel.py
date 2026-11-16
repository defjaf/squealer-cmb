#!/usr/bin/env python3
"""
cmb_squealer_multipanel.py — 4-panel squealer summary figure.

(a) upper-left:  CMB data + best-fit + parameter table
(b) upper-right: ell*=22,  Dl*=600   (low-ell power deficit)
(c) lower-left:  ell*=464, Dl*=2225  (+3σ excursion in first trough)
(d) lower-right: ell*=220, Dl*=6000  (above first acoustic peak)

By default θ̂, σ̂ and each panel's θ*, σ*, cost are the values from the
cmb_squealer.py runs used for the paper (hard-coded below), so this script
needs only CAMB and the Planck data files. With --from-results it instead reads
results/cmb_squealer_ell{ℓ*}_Dl{D_ℓ*}.json written by fresh cmb_squealer.py runs.

Writes figures/cmb_squealer_multipanel.pdf/png.
"""

import numpy as np
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import camb

from planck_paths import LOWELL_BANDPOWERS, PLIK_LITE_BANDPOWERS, FIGDIR, RESULTSDIR

parser = argparse.ArgumentParser()
parser.add_argument("--from-results", action="store_true",
                    help="use results/*.json from cmb_squealer.py instead of the paper values")
args = parser.parse_args()

# ── CAMB wrapper ──────────────────────────────────────────────────────────────
def camb_dl(theta6):
    H0, ombh2, omch2, tau, ns, As = theta6
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(2558, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    return results.get_cmb_power_spectra(pars, CMB_unit="muK")["total"][:2509, 0]

# ── parameters ────────────────────────────────────────────────────────────────
PARAM_NAMES  = ["H0", "ombh2", "omch2", "tau", "ns", "As", "A_planck"]
PARAM_LABELS = [r"$H_0$", r"$\Omega_{\rm b} h^2$", r"$\Omega_{\rm c} h^2$",
                r"$\tau$", r"$n_{\rm s}$", r"$10^9 A_{\rm s}$", r"$A_{\rm Pl}$"]
SIGMA_FI     = np.array([0.92, 0.000215, 0.0021, 0.00737, 0.00563, 3.15e-11, 1e-3])
THETA_HAT    = np.array([67.366, 0.022335, 0.12009, 0.055149, 0.96498, 2.1017e-9, 1.0001])

# display values and errors for the table (As rescaled to 10^-9)
# each entry: (label, value, error) — label left-aligned, value+error right-aligned
TABLE_LINES = [
    (r"$A_{\rm Pl}$",          "1.000",    "0.001"),
    (r"$10^9 A_{\rm s}$",      "2.102",    "0.032"),
    (r"$n_{\rm s}$",           "0.9650",   "0.0056"),
    (r"$\tau$",                "0.0552",   "0.0074"),
    (r"$\Omega_{\rm c} h^2$",  "0.1201",   "0.0021"),
    (r"$\Omega_{\rm b} h^2$",  "0.02234",  "0.00022"),
    (r"$H_0$",                 "67.37",    "0.92"),
]

PANELS = [
    dict(pos=(0,0), label="(a) Best-fit $\\Lambda$CDM",
         ELL_STAR=None, DL_STAR=None, sigma_star=None, cost=None,
         theta_star=None),
    dict(pos=(0,1), label=r"(b) $\ell^*=22$, cost $=150.4$",
         ELL_STAR=22,  DL_STAR=600.0,  sigma_star=7.59,  cost=150.39,
         theta_star=np.array([67.227, 0.022122, 0.11962, 0.067675, 1.02, 2.077e-9, 0.98972])),
    dict(pos=(1,0), label=r"(c) $\ell^*=464$, cost $=40.0$",
         ELL_STAR=464, DL_STAR=2225.0, sigma_star=4.24,  cost=40.04,
         theta_star=np.array([68.674, 0.020998, 0.11327, 0.054007, 0.97034, 2.1011e-9, 1.0114])),
    dict(pos=(1,1), label=r"(d) $\ell^*=220$, cost $=20.2$",
         ELL_STAR=220, DL_STAR=6000.0, sigma_star=14.33, cost=20.21,
         theta_star=np.array([68.935, 0.022993, 0.11782, 0.047559, 0.95825, 2.1011e-9, 1.0061])),
]

if args.from_results:
    for p in PANELS:
        if p['ELL_STAR'] is None:
            continue
        fname = RESULTSDIR / f"cmb_squealer_ell{p['ELL_STAR']}_Dl{int(p['DL_STAR'])}.json"
        with open(fname) as f:
            r = json.load(f)
        p.update(sigma_star=r['sigma_star'], cost=r['cost'],
                 theta_star=np.array(r['theta_star']),
                 label=p['label'].split(", cost")[0] + f", cost $={r['cost']:.1f}$")
        THETA_HAT = np.array(r['theta_hat'])     # identical in every run
        SIGMA_FI  = np.array(r['sigma_fisher'])
        print(f"  read {fname.name}")

# ── CAMB evaluations ──────────────────────────────────────────────────────────
print("Computing CAMB spectra...")
Dl_hat = camb_dl(THETA_HAT[:6])
ell_t  = np.arange(len(Dl_hat))
for p in PANELS:
    p['Dl_hat']  = Dl_hat
    p['Dl_star'] = camb_dl(p['theta_star'][:6]) if p['theta_star'] is not None else None
    print(f"  {p['label'][:30]}  done")

# ── observational data ────────────────────────────────────────────────────────
full     = np.loadtxt(LOWELL_BANDPOWERS)
lo_mask  = full[:, 0] < 30
ell_lo   = full[lo_mask, 0];  Dl_lo    = full[lo_mask, 1]
eDl_lo_m = full[lo_mask, 2];  eDl_lo_p = full[lo_mask, 3]
dat      = np.loadtxt(PLIK_LITE_BANDPOWERS)
ell_hi   = dat[:, 0];  norm   = ell_hi*(ell_hi+1)/(2*np.pi)
Dl_hi    = norm*dat[:, 1];  sDl_hi = norm*dat[:, 2]

# ── custom x-scale ────────────────────────────────────────────────────────────
SPLIT = 30.0
def forward(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x<=SPLIT, x, SPLIT*(1+np.log(np.maximum(x,1e-10)/SPLIT)))
def inverse(y):
    y = np.asarray(y, dtype=float)
    return np.where(y<=SPLIT, y, SPLIT*np.exp(y/SPLIT-1))

XTICKS = [2, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]

# ── figure: all panels use same 2-col inner layout ───────────────────────────
fig   = plt.figure(figsize=(14, 9))
outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.15)

for p in PANELS:
    i, j = p['pos']

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[i, j],
        height_ratios=[3, 1], width_ratios=[4, 1],
        hspace=0.05, wspace=0.08,
    )
    ax_s = fig.add_subplot(inner[0, 0])
    ax_r = fig.add_subplot(inner[1, 0], sharex=ax_s)
    ax_x = fig.add_subplot(inner[:, 1])   # table or param shifts

    # x-axis: labels on residuals only
    for ax in (ax_s, ax_r):
        ax.set_xscale("function", functions=(forward, inverse))
        ax.axvline(SPLIT, color="gray", lw=0.6, ls=":", alpha=0.6)
        ax.set_xticks(XTICKS)
        ax.set_xlim(2, 2508)
    ax_s.tick_params(axis='x', labelbottom=False)
    ax_r.set_xticklabels([str(t) for t in XTICKS], fontsize=7)
    ax_r.set_xlabel(r"Multipole $\ell$", fontsize=9)

    # data
    ax_s.errorbar(ell_lo, Dl_lo, yerr=[eDl_lo_m, eDl_lo_p],
                  fmt="none", elinewidth=0.9, color="steelblue", alpha=0.8)
    ax_s.errorbar(ell_hi, Dl_hi, sDl_hi,
                  fmt="none", elinewidth=0.5, color="steelblue", alpha=0.55)

    # curves
    ax_s.plot(ell_t, p['Dl_hat'], color="firebrick", lw=1.2, zorder=3,
              label=r"Best-fit $\hat\theta$")
    if p['Dl_star'] is not None:
        ax_s.plot(ell_t, p['Dl_star'], color="darkorange", lw=1.2, ls="--",
                  zorder=3, label=r"Perturbed $\theta^*$")
        ax_s.scatter([p['ELL_STAR']], [p['DL_STAR']], marker="*", s=150,
                     color="darkorange", zorder=5)
        ax_s.errorbar([p['ELL_STAR']], [p['DL_STAR']],
                      yerr=[[p['sigma_star']], [p['sigma_star']]],
                      fmt="none", elinewidth=1.8, color="darkorange", capsize=3, zorder=4)

    ax_s.set_ylim(0, 7000)
    ax_s.set_yticks([1000, 2000, 3000, 4000, 5000, 6000, 7000])
    ax_s.set_title(p['label'], fontsize=10, pad=4)
    ax_s.set_ylabel(r"$D_\ell^{TT}\ [\mu\mathrm{K}^2]$", fontsize=9)
    ax_s.tick_params(axis='y', labelsize=8)

    # residuals
    res_lo_h = (Dl_lo - np.interp(ell_lo, ell_t, p['Dl_hat'])) / ((eDl_lo_m+eDl_lo_p)/2)
    res_hi_h = (Dl_hi - np.interp(ell_hi, ell_t, p['Dl_hat'])) / sDl_hi
    ax_r.axhline(0,  color="firebrick", lw=0.9)
    ax_r.axhline( 2, color="gray", lw=0.5, ls="--")
    ax_r.axhline(-2, color="gray", lw=0.5, ls="--")
    ax_r.scatter(ell_lo, res_lo_h, s=8, color="firebrick", alpha=0.7, zorder=3)
    ax_r.scatter(ell_hi, res_hi_h, s=2, color="firebrick", alpha=0.6, zorder=3)
    if p['Dl_star'] is not None:
        res_lo_s = (Dl_lo - np.interp(ell_lo, ell_t, p['Dl_star'])) / ((eDl_lo_m+eDl_lo_p)/2)
        res_hi_s = (Dl_hi - np.interp(ell_hi, ell_t, p['Dl_star'])) / sDl_hi
        ax_r.axhline(0, color="darkorange", lw=0.9, ls="--")
        ax_r.scatter(ell_lo, res_lo_s, s=8, color="darkorange", alpha=0.7, zorder=3)
        ax_r.scatter(ell_hi, res_hi_s, s=2, color="darkorange", alpha=0.6, zorder=3)
    ax_r.set_ylim(-5, 5)
    ax_r.set_ylabel(r"$(D_\ell{-}D^{\rm th})/\sigma$", fontsize=8)
    ax_r.tick_params(axis='y', labelsize=7)

    # ── right panel: table (a) or parameter shifts (b,c,d) ───────────────────
    if p['theta_star'] is None:
        ax_x.axis('off')
        ax_x.set_xlim(0, 1); ax_x.set_ylim(0, 1)

        # space: each param gets 2 tight lines + a gap; total height = 1.0
        line_h  = 0.055   # height per text line
        gap_h   = 0.040   # extra gap between parameters
        block_h = 2 * line_h + gap_h
        top     = 0.97
        for i, (label, val, err) in enumerate(TABLE_LINES):
            y_val = top - i * block_h
            y_err = y_val - line_h
            ax_x.text(0.52, y_val, label + r" $=$", ha='right', va='center',
                      transform=ax_x.transAxes, fontsize=7.5)
            ax_x.text(0.97, y_val, val, ha='right', va='center',
                      transform=ax_x.transAxes, fontsize=7.5)
            ax_x.text(0.97, y_err, r"$\pm\,$" + err, ha='right', va='center',
                      transform=ax_x.transAxes, fontsize=7.5)
    else:
        # parameter shift bar chart
        shifts      = p['theta_star'] - THETA_HAT
        shifts_norm = shifts / SIGMA_FI
        colors = ["tomato" if s > 0 else "steelblue" for s in shifts_norm]
        ypos   = np.arange(len(PARAM_LABELS))
        ax_x.barh(ypos, shifts_norm, color=colors, alpha=0.8, height=0.6)
        ax_x.axvline(0,  color="black", lw=0.8)
        ax_x.axvline( 1, color="gray",  lw=0.6, ls="--", alpha=0.7)
        ax_x.axvline(-1, color="gray",  lw=0.6, ls="--", alpha=0.7)
        ax_x.set_yticks([])
        lim = max(3, abs(shifts_norm).max() * 1.3)
        ax_x.set_xlim(-lim, lim)
        ax_x.set_xlabel(r"$\Delta\theta/\hat\sigma$", fontsize=8)
        ax_x.tick_params(axis='x', labelsize=7)
        ax_x.set_title(f"cost$={p['cost']:.1f}$", fontsize=8, pad=3)
        # parameter labels pinned to the far left of the panel
        for y, label in zip(ypos, PARAM_LABELS):
            ax_x.text(-lim * 0.97, y, label, ha='left', va='center',
                      fontsize=7.5, color='black')

# legend in panel (a) spectrum only
fig.axes[0].legend(fontsize=8, loc="upper left")


FIGDIR.mkdir(exist_ok=True)
out = FIGDIR / "cmb_squealer_multipanel"
plt.savefig(f"{out}.pdf", bbox_inches="tight")
plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
print(f"Saved figures/cmb_squealer_multipanel.pdf/.png")
