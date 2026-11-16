# The Squealer: CMB power spectrum example

Code for the cosmology example ("A model of power spectra in cosmology", Fig. CMB1) in

> A. Gelman, A. H. Jaffe, E. Carlson & P. Greengard,
> *The Squealer: Sensification of model exploration and model misfit*,
> [arXiv:2606.29842](https://arxiv.org/abs/2606.29842).

The Squealer drags a fitted curve toward a user-chosen pseudo-data point and reports
how far the model parameters must move, and how much the original posterior dislikes
the move. Here the curve is the Planck 2018 CMB temperature power spectrum
$D_\ell^{TT}(\theta)$ in six-parameter ΛCDM.

This repository covers only the CMB example. The other examples in the paper
(golf putting, dilution assays, Gaussian-process temperature series) are not included.

## Method

1. **Posterior.** $\log p(\theta\mid d) = \log L_\text{plik\_lite}^{TT} + \log L_\text{Commander}^{TT}
   + \log N(\tau \mid 0.0543, 0.0073^2)$, with $\theta = (H_0, \Omega_b h^2, \Omega_c h^2, \tau, n_s, A_s, A_\text{Planck})$.
   The Planck 2018 *temperature-only* likelihoods are used: plik_lite for $30 \le \ell \le 2508$
   and Commander for $2 \le \ell \le 29$. The Gaussian prior on $\tau$ stands in for the
   low-$\ell$ polarisation data. $C_\ell(\theta)$ is computed with CAMB.
2. **Best fit.** Find the posterior mode $\hat\theta$ (Powell, in rescaled parameters).
3. **Uncertainty.** Fisher matrix $F$ of the six cosmological parameters by finite differences
   at $\hat\theta$; marginal widths $\hat\sigma = \sqrt{\mathrm{diag}\,F^{-1}}$.
4. **Pseudo-data width.** $\sigma^* = \tfrac12 \sqrt{\nabla D_{\ell^*}^{\,T} F^{-1} \nabla D_{\ell^*}}$,
   half the posterior-predictive standard deviation of $D_{\ell^*}$.
5. **Perturbed fit.** Maximise $\log p(\theta\mid d) + \log N(D^*_{\ell^*} \mid D_{\ell^*}(\theta), \sigma^{*2})$
   to get $\theta^*$.
6. **Report.** cost $= \log p(\hat\theta\mid d) - \log p(\theta^*\mid d)$, the shifts
   $(\theta^*-\hat\theta)/\hat\sigma$, and the perturbed spectrum.

## Files

| File | Purpose |
|---|---|
| `setup_planck.sh` | Downloads the Planck 2018 likelihood code and data, builds `clik` and its Python bindings, runs a self-check |
| `planck_paths.py` | Locations of the Planck files; `python planck_paths.py` re-runs the self-check |
| `cmb_squealer.py` | The Squealer for one pseudo-data point; writes a figure and a JSON summary |
| `cmb_squealer_multipanel.py` | The four-panel paper figure (Fig. CMB1) |
| `cmb_spectrum.py` | Planck band-powers with the published best-fit spectrum (no `clik`/CAMB needed) |

## Requirements

### External software and data

| Component | Version used | Source | Reference |
|---|---|---|---|
| CAMB (Boltzmann code) | 2.0.5 (paper figure: 1.6.6) | `pip install camb`, <https://camb.info> | Lewis, Challinor & Lasenby 2000, ApJ 538, 473 |
| Planck likelihood code `plc` | 3.1 (`COM_Likelihood_Code-v3.0_R3.10`) | [Planck Legacy Archive](https://pla.esac.esa.int/) | Planck Collaboration 2020, A&A 641, A5 |
| Planck baseline likelihood data | `COM_Likelihood_Data-baseline_R3.00` (60 MB) | Planck Legacy Archive | Planck Collaboration 2020, A&A 641, A5 |
| Planck TT band-powers (display only) | `COM_PowerSpect_CMB-TT-full_R3.01.txt` | Planck Legacy Archive | Planck Collaboration 2020, A&A 641, A1 |

`setup_planck.sh` downloads all three Planck files from the Planck Legacy Archive.

### Build tools

Building `clik` needs a C compiler, `gfortran` and the CFITSIO library. BLAS/LAPACK come from
Apple's Accelerate framework on macOS and from the system on Linux.

- **macOS** (Homebrew): `xcode-select --install; brew install gcc cfitsio`
- **Linux** (Debian/Ubuntu): `sudo apt install build-essential gfortran libcfitsio-dev liblapack-dev`

### Python

Python ≥ 3.10 with the packages in `requirements.txt` (numpy, scipy, matplotlib, camb; cython to build `clik`).

## Installation

```bash
git clone https://github.com/defjaf/squealer-cmb.git
cd squealer-cmb
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./setup_planck.sh
```

`setup_planck.sh` installs everything under `./planck_2018/` (about 190 MB including the
downloaded archives). It finishes by loading both likelihoods at the Planck best-fit spectrum,
which should print

```
plik_lite TT  logL(best fit) = -103.29
Commander TT  logL(best fit) = -11.84
```

Options, all set as environment variables:
- `PLANCK_DIR`: install somewhere else, or use an existing installation. Set it again when running the scripts.
- `PYTHON`: choose the Python interpreter.
- `CFITSIO_PREFIX`: location of CFITSIO when Homebrew isn't available.
- `WAF_CONFIGURE_ARGS`: extra options for `plc`'s `waf configure`, e.g. `--lapack_prefix=...` (see `planck_2018/code/plc_3.0/plc-3.1/readme.md`).

The script patches two problems in `plc` 3.1 for current toolchains. First, waf hard-codes
`-arch x86_64` on macOS, so the script uses the native architecture. Second, waf can no longer
build the Python bindings on Python ≥ 3.12, so the script builds them with Cython directly.
It has been tested on macOS (Apple Silicon) with Homebrew. On Linux you may need to supply
`WAF_CONFIGURE_ARGS` for your LAPACK.

## Usage

```bash
# one pseudo-data point: (ℓ*, D_ℓ*) with D_ℓ* in μK²
python cmb_squealer.py --ell 464 --dl 2225

# sanity check: D_ℓ* set to the best-fit value, so cost and shifts should be ≈ 0
python cmb_squealer.py --ell 464 --test

# the paper figure from the paper's stored results (about 10 s)
python cmb_squealer_multipanel.py

# data and best-fit spectrum only
python cmb_spectrum.py
```

Outputs go to `figures/` and `results/`. Each `cmb_squealer.py` run takes a few minutes on a
laptop: mode finding, 73 CAMB calls for the Fisher matrix, then the perturbed optimisation.

### Reproducing Fig. CMB1 from scratch

```bash
python cmb_squealer.py --ell 22  --dl 600     # (b) low-ℓ power deficit
python cmb_squealer.py --ell 464 --dl 2225    # (c) excursion in the first trough
python cmb_squealer.py --ell 220 --dl 6000    # (d) above the first peak
python cmb_squealer_multipanel.py --from-results
```

Without `--from-results`, `cmb_squealer_multipanel.py` uses the θ̂, σ̂, θ*, σ* and cost values
stored in the script. Those values came from these same runs and were used for the paper figure.

Rerunning with CAMB 2.0.5 rather than the original 1.6.6 gives these values:

| Panel | σ* (paper → rerun) | cost (paper → rerun) |
|---|---|---|
| (b) ℓ* = 22 | 7.59 → 7.63 μK² | 150.4 → 150.3 |
| (c) ℓ* = 464 | 4.24 → 4.25 μK² | 40.0 → 39.0 |
| (d) ℓ* = 220 | 14.33 → 14.32 μK² | 20.2 → 20.0 |

All parameter shifts agree to within 0.1 σ̂. The costs are accurate to roughly ±0.5, set by the
Powell tolerances. For example, `--test` gives a cost of −0.4 rather than exactly 0.

## Notes

- Only the posterior *mode* is perturbed. Unlike the other examples in the paper, no posterior samples are drawn.
- The low-ℓ ($\ell<30$) points in the figures are the published Planck per-multipole estimates, used for display only. The fit uses the full Commander likelihood.
- No prior is placed on the calibration parameter $A_\text{Planck}$. Its bar in the parameter-shift panels is scaled by the finite-difference step ($10^{-3}$), not by a posterior width.

## Citation

```bibtex
@article{Gelman2026Squealer,
  author        = {Gelman, Andrew and Jaffe, Andrew H. and Carlson, Eliot and Greengard, Philip},
  title         = {The Squealer: Sensification of model exploration and model misfit},
  year          = {2026},
  eprint        = {2606.29842},
  archivePrefix = {arXiv},
}
```

If you use the Planck likelihood or CAMB, please also cite the references in the table above.

## License

MIT (see `LICENSE`). The Planck likelihood code and data downloaded by `setup_planck.sh` are
distributed by ESA under their own terms and are not part of this repository.
