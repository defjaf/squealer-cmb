"""
planck_paths.py — locations of the Planck 2018 likelihood code and data.

Everything lives under PLANCK_DIR (default: ./planck_2018 next to this file),
as installed by setup_planck.sh. Override with the PLANCK_DIR environment
variable to use an existing installation.

Run directly (`python planck_paths.py`) to load both likelihoods and check them.
"""

import os
import sys
from pathlib import Path

REPO       = Path(__file__).resolve().parent
PLANCK_DIR = Path(os.environ.get("PLANCK_DIR", REPO / "planck_2018")).resolve()
PLC        = PLANCK_DIR / "code/plc_3.0/plc-3.1"
BASELINE   = PLANCK_DIR / "baseline/plc_3.0"

PLIK_LITE_TT = BASELINE / "hi_l/plik_lite/plik_lite_v22_TT.clik"            # ℓ = 30–2508
COMMANDER_TT = BASELINE / "low_l/commander/commander_dx12_v3_2_29.clik"     # ℓ = 2–29
PLIK_LITE_EXT = PLIK_LITE_TT / "clik/lkl_0/_external"

# display data
LOWELL_BANDPOWERS = PLANCK_DIR / "COM_PowerSpect_CMB-TT-full_R3.01.txt"     # per-ℓ D_ℓ, ℓ < 30
PLIK_LITE_BANDPOWERS = PLIK_LITE_EXT / "cl_cmb_plik_v22.dat"                # binned C_ℓ, ℓ ≥ 30
BESTFIT_THEORY = PLIK_LITE_EXT / "bf_lite_plikTTTEEE_v22b_lowl_simall.minimum.theory_cl"

FIGDIR     = REPO / "figures"
RESULTSDIR = REPO / "results"


def import_clik():
    """Import the clik Python bindings built by setup_planck.sh."""
    if not PLC.is_dir():
        sys.exit(f"Planck likelihood code not found at {PLC}\n"
                 "Run ./setup_planck.sh or set PLANCK_DIR.")
    sys.path.insert(0, str(PLC / "lib/python/site-packages"))
    os.environ.setdefault("CLIK_DATA", str(PLC / "share/clik"))
    import clik
    return clik


if __name__ == "__main__":
    import numpy as np
    clik = import_clik()
    lkl_hi  = clik.clik(str(PLIK_LITE_TT))   # clik runs its own self-check on load
    lkl_low = clik.clik(str(COMMANDER_TT))

    bf = np.loadtxt(BESTFIT_THEORY)          # columns: ℓ, D_TT, ...
    Cl = np.zeros(lkl_hi.get_lmax()[0] + 1)
    ell = bf[:, 0].astype(int)
    keep = ell < len(Cl)
    Cl[ell[keep]] = bf[keep, 1] * 2 * np.pi / (ell[keep] * (ell[keep] + 1.0))

    print(f"plik_lite TT  logL(best fit) = {lkl_hi(np.append(Cl, 1.0))[0]:.2f}")
    print(f"Commander TT  logL(best fit) = {lkl_low(np.append(Cl[:lkl_low.get_lmax()[0] + 1], 1.0))[0]:.2f}")
