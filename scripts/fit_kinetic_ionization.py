#!/usr/bin/env python3
"""
The refit of the kinetic ionization cross section with the factor 1/β² in its relativistic
Bethe term: writes the data file data/kinetic_ionization_bethe_fits.toml, which
src/physics/ionization_bethe_fits.jl reads.

DREAM's fit (tools/fit_kineticionization.py with tools/ADAS/fit.py and helpers.py, method
"single", MIT License, Copyright (c) 2020 Mathias Hoppe and Ola Embreus), run unchanged:
per charge state, a least-squares fit of the logarithm of the Maxwell-Jüttner average of
v σ to the ADAS SCD coefficients at the lowest density, over the temperature window of
tools/kineticCrossSectionParams.json (2 to 100 eV where it has none), with C and β* free
and I the NIST ionization energy. Only the cross section differs: its relativistic term
C π a0² α² (Ry/I) (ln(p²/2I) − β²) is divided by β², so that it joins the large-U limit of
the Burgess-Chidichimo term (Garland et al., Phys. Plasmas 27, 040702 (2020)), and the
constants are those of scipy.constants (CODATA 2022 from SciPy 1.15 on), within 1e-9 of those with
which Runes2P's `ionization_cross_section` evaluates it, where DREAM's tool takes a0 = 5.29e-11 m,
Ry = 13.6 eV and α = 1/137. The blend is floored at zero, which Runes2P's form with 1/β² does
as well.

Usage (from reference/dream, whose pixi environment has DREAM's tools; the DREAM checkout of
the boltzmann-eqterm set, reference/dream/dream-boltzmann-eqterm-93d27e17/DREAM (build_all.sh),
with its ADAS cache in tools/cache):
    pixi run python ../../scripts/fit_kinetic_ionization.py out.toml [element ...]
With `--dream-form` the cross section and the constants are DREAM's own, which reproduces
the values of DREAM's src/Atomics/kineticionizationdata.cpp, held in
data/kinetic_ionization_fits.toml (the check of the method). That check compares numbers,
not bytes: the data file spells its floats as Julia does (1.0e-15), the script as Python
does (1e-15), and the script writes a header of its own.
The script writes the fits as TOML in the form of data/kinetic_ionization_bethe_fits.toml
(see data/README.md): the table [metadata] (METADATA, or DREAM_FORM_METADATA with
`--dream-form`), then in [data] one array per element with one row [C, I, β*] per charge
state Z0 = 0, …, Z − 1. Each value is rounded to 15 decimals and written in its shortest
round-trip form. The script prints the ratio of the fitted average to the SCD at the ends
and the middle of each window.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.constants as sc

# DREAM's tools and ADAS cache from the checkout of the boltzmann-eqterm set, the DREAM commit that the code cites
TOOLS = (Path(__file__).resolve().parents[1] / "reference" / "dream" / "dream-boltzmann-eqterm-93d27e17"
         / "DREAM" / "tools")
sys.path.insert(0, str(TOOLS))
import ADAS  # noqa: E402
import ADAS.helpers as helpers  # noqa: E402
import ADAS.fit  # noqa: E402

MC2 = sc.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
A0 = sc.physical_constants["Bohr radius"][0]
RY = sc.physical_constants["Rydberg constant times hc in eV"][0]
ALPHA = sc.fine_structure

# the elements of Runes2P's KINETIC_IONIZATION_FITS, in its order
ELEMENTS = ["H", "D", "T", "He", "Li", "Be", "B", "C", "O", "Ne", "Ar", "Fe", "W"]


def bethe_cross_section(p, C, DI_eV, betaStar):
    """DREAM's helpers.kineticIonizationContribution with 1/β² on the relativistic term."""
    p = np.asarray(p, dtype=float)
    gamma = np.hypot(1, p)
    Ek = p * (p / (gamma + 1))        # p²/(γ + 1) without the overflow of p² at the momenta of quad
    DI = DI_eV / MC2
    U = Ek / DI
    above = U > 1
    Us = np.where(above, U, 2.0)
    nonrel = np.pi * A0**2 * C * (RY / DI_eV)**2 * np.log(Us)**(1 + betaStar / Us) / Us
    beta2 = np.where(above, (p / gamma)**2, 1.0)
    rel = np.pi * A0**2 * ALPHA**2 * C * (RY / DI_eV) * (2 * np.log(np.where(above, p, 1.0)) - np.log(2 * DI) - beta2) / beta2
    S = 1 / (1 + np.exp(1 - Ek * MC2 * 1e-5))
    # floored at zero: just above the threshold of a state with I of several keV the Bethe
    # logarithm ln(p²/2I) lies below β², and the weight S of the absolute energy is already
    # 0.3 at 15 keV, so the blend is negative there (W⁶⁴⁺, I = 15.6 keV, at T = 450 eV)
    out = np.where(above, np.maximum((1 - S) * nonrel + S * rel, 0.0), 0.0)
    return float(out) if out.ndim == 0 else out


def maxwell_juttner(p, n, T):
    """DREAM's helpers.maxwellJuttnerDistribution with the CODATA rest energy."""
    from scipy.special import kve
    theta = T / MC2
    gamma = np.hypot(1, p)
    return n / (4 * np.pi * theta * kve(2, 1 / theta)) * np.exp(-(p * (p / (gamma + 1))) / theta)


def dream_table():
    """The fits of DREAM's src/Atomics/kineticionizationdata.cpp, per element (C, I, β*)."""
    text = (TOOLS.parent / "src" / "Atomics" / "kineticionizationdata.cpp").read_text()
    table, element = {}, None
    for line in text.splitlines():
        if line.startswith("const real_t ") and "_params[" in line:
            element = line.split()[2].split("_params")[0]
            table[element] = []
        elif element is not None and line.startswith("/* Z0 ="):
            table[element].append([float(x) for x in line.split("*/")[1].replace(",", " ").split()])
        elif line.startswith("};"):
            element = None
    return table


DREAM_TABLE = dream_table()


def nist_energies(elements, datatype, cache=True, cachedir=None):
    """The NIST ionization energies that DREAM's fit took, from the I column of its table, in
    place of the download of tools/get_nist.py."""
    assert datatype == "ionization"
    return None, None, [np.array([row[1] for row in DREAM_TABLE[e]]) for e in elements]


ADAS.data.get_nist.load_elements = nist_energies
DREAM_FORM = "--dream-form" in sys.argv
if DREAM_FORM:
    sys.argv.remove("--dream-form")
else:
    helpers.kineticIonizationContribution = bethe_cross_section
if not DREAM_FORM:
    helpers.maxwellJuttnerDistribution = maxwell_juttner
    helpers.mc2 = MC2


def windows():
    return json.loads((TOOLS / "kineticCrossSectionParams.json").read_text())


def nearest(arr, val):
    return arr[np.abs(arr - val).argmin()]


def fit_element(element, limits):
    scd, Z, _, T, E_ion = ADAS.data.getIonizationData(element)
    rows, checks = [], []
    for Z0 in range(Z):
        window = limits.get(element, {}).get(str(Z0))
        if window is None:
            Tl, Tu = 2, 100
        else:
            Tl, Tu = nearest(T, window["Tlow"]), nearest(T, window["Tupp"])
        _, _, out = ADAS.fit.fitKineticIonizationForSpecies(element, Z0, "single", T_lower=Tl, T_upper=Tu)
        rows.append((out["C1"], out["DI1"], out["betaStar"]))
        inside = T[(T > Tl) & (T <= Tu)] if np.any((T > Tl) & (T <= Tu)) else T[:1]
        probe = np.array([inside[0], inside[len(inside) // 2], inside[-1]])
        average = helpers.averagedIonizationCrossSection(probe, out["C1"], out["DI1"], out["betaStar"])
        reference = np.array([scd[Z0, np.argmin(np.abs(T - t)), 0] for t in probe])
        checks.append((Z0, probe, average / reference))
        print(f"{element} Z0={Z0}: C={out['C1']:.6g} I={out['DI1']:.6g} beta*={out['betaStar']:.6g} "
              f"avg/SCD at T={probe} eV: {np.round(average / reference, 4)}", flush=True)
    return rows, checks


# The header of the data file. VERSION_OR_DATE names the inputs of the fit and the date it
# was run; it changes when the fit is run again on other inputs.
METADATA = {
    "source": (
        "fit of this package with DREAM's tools/fit_kineticionization.py and tools/ADAS/fit.py "
        "(method \"single\", scripts/fit_kinetic_ionization.py), the cross section with 1/β² in "
        "its relativistic Bethe term, to the ADAS SCD coefficients (H. P. Summers, The ADAS User "
        "Manual, version 2.6 (2004), https://www.adas.ac.uk); I is the NIST ionization energy of "
        "DREAM's src/Atomics/kineticionizationdata.cpp"
    ),
    "version_or_date": (
        "fitted on 2026-10-12 with DREAM 93d27e17 (the boltzmann-eqterm set) and the ADAS files "
        "of its tools/cache"
    ),
    "licence": (
        "DREAM's tools: MIT, see THIRD_PARTY_NOTICES.md; derived by least-squares fits to ADAS "
        "SCD coefficients, no tabulated ADAS value is reproduced"
    ),
    "generator": "scripts/fit_kinetic_ionization.py",
    "description": "per element, one row per charge state Z0 = 0, …, Z − 1: C, I [eV], β*",
}

# The header with `--dream-form`: the fit with DREAM's own cross section and constants, the
# check against data/kinetic_ionization_fits.toml.
DREAM_FORM_METADATA = {
    **METADATA,
    "source": (
        "fit with DREAM's tools/fit_kineticionization.py and tools/ADAS/fit.py (method \"single\", "
        "scripts/fit_kinetic_ionization.py --dream-form), DREAM's own cross section and constants, "
        "to the ADAS SCD coefficients (H. P. Summers, The ADAS User Manual, version 2.6 (2004), "
        "https://www.adas.ac.uk); I is the NIST ionization energy of DREAM's "
        "src/Atomics/kineticionizationdata.cpp"
    ),
    "version_or_date": "DREAM 93d27e17 (the boltzmann-eqterm set) and the ADAS files of its tools/cache",
    "generator": "scripts/fit_kinetic_ionization.py --dream-form",
}


def toml_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def toml_float(x):
    """The value rounded to 15 decimals, in the shortest form that parses back to it."""
    x = float(f"{x:.15f}")
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(x)


def toml_table(table):
    """The data file: the header, then one array of rows [C, I, β*] per element."""
    lines = ["[metadata]"]
    metadata = DREAM_FORM_METADATA if DREAM_FORM else METADATA
    lines += [f"{key} = {toml_string(value)}" for key, value in metadata.items()]
    lines += ["", "[data]"]
    for element, rows in table.items():
        lines.append(f"{element} = [")
        lines += ["    [" + ", ".join(toml_float(x) for x in row) + "]," for row in rows]
        lines.append("]")
    return "\n".join(lines) + "\n"


def main(outfile, elements):
    limits = windows()
    table = {}
    for element in elements:
        table[element], _ = fit_element(element, limits)
    Path(outfile).write_text(toml_table(table), encoding="utf-8")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ELEMENTS)
