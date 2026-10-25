#!/usr/bin/env python3
"""
Write the data files data/mean_excitation_energies.toml and data/ion_sizes.toml
from the DREAM C++ sources, so that the tabulated atomic data is copied mechanically.

Usage: python3 scripts/port_atomic_data.py <DREAM source dir> [<output dir>]

The output directory defaults to data/ of the repository.
"""
import re
import sys
from pathlib import Path

# The DREAM checkout that the data files of the repository are generated from.
DREAM_VERSION = ("DREAM commit 93d27e1763827e19dafb1fe790e99c2688ad6d62 (branch boltzmann-eqterm); "
                 "the three source files are identical in DREAM v26.5")
LICENCE = "MIT, see THIRD_PARTY_NOTICES.md"
GENERATOR = "scripts/port_atomic_data.py"

dream = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "data"
sdf = (dream / "src/Equations/SlowingDownFrequency.cpp").read_text()
psf = (dream / "src/Equations/PitchScatterFrequency.cpp").read_text()
ext = (dream / "src/Equations/MeanExcitationEnergy_Extended.cpp").read_text()


def strip_comments(s):
    return re.sub(r"/\*.*?\*/", "", s, flags=re.S)


def array_body(src, name):
    m = re.search(re.escape(name) + r"\s*(\[[^\]]*\])+\s*=\s*\{(.*?)\};", src, flags=re.S)
    if m is None:
        raise SystemExit(f"array {name} not found")
    return strip_comments(m.group(2))


def numbers(body):
    return [t.strip() for t in body.replace("{", ",").replace("}", ",").split(",") if t.strip()]


def toml_float(v):
    """Shortest round-trip form of the C literal; NAN is the TOML literal nan."""
    return "nan" if v == "NAN" else repr(float(v))


def toml_int(v):
    if re.fullmatch(r"\d+", v) is None:
        raise SystemExit(f"not an integer: {v}")
    return v


def rows(body):
    return [[t.strip() for t in r.split(",") if t.strip()] for r in re.findall(r"\{([^{}]*)\}", body)]


def toml_matrix(key, rs):
    lines = [f"{key} = ["]
    for r in rs:
        lines.append("    [" + ", ".join(toml_float(v) for v in r) + "],")
    lines.append("]")
    return "\n".join(lines)


def toml_vector(key, vs, fmt):
    return f"{key} = [" + ", ".join(fmt(v) for v in vs) + "]"


def toml_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write(name, metadata, data):
    lines = ["[metadata]"]
    lines += [f"{key} = {toml_string(value)}" for key, value in metadata]
    lines += ["", "[data]"] + data
    with open(out / name, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


mee = rows(array_body(sdf, "SlowingDownFrequency::MEAN_EXCITATION_ENERGY_DATA"))
d = numbers(array_body(sdf, "SlowingDownFrequency::MEAN_EXCITATION_ENERGY_FUNCTION_D"))
s0 = numbers(array_body(sdf, "SlowingDownFrequency::MEAN_EXCITATION_ENERGY_FUNCTION_S_0"))
extrows = rows(array_body(ext, "MEAN_EXCITATION_ENERGY_EXTENDED"))

write("mean_excitation_energies.toml",
      [("source", "DREAM, src/Equations/SlowingDownFrequency.cpp (MEAN_EXCITATION_ENERGY_DATA, "
                  "MEAN_EXCITATION_ENERGY_FUNCTION_D, MEAN_EXCITATION_ENERGY_FUNCTION_S_0) and "
                  "src/Equations/MeanExcitationEnergy_Extended.cpp (MEAN_EXCITATION_ENERGY_EXTENDED, "
                  "written by DREAM's tools/Generate_MeanExcitationEnergy.py); the tabulated values, D_N and "
                  "S_N0 of S. P. A. Sauer, J. R. Sabin and J. Oddershede, J. Chem. Phys. 148, 174307 (2018)"),
       ("version_or_date", DREAM_VERSION),
       ("licence", LICENCE),
       ("generator", GENERATOR),
       ("description", "mean excitation energy [eV]: mean_excitation_energy row Z = 1, ..., 18, column Z0 + 1; "
                       "extended row Z = 1, ..., 86 (values in rows 19 to 86), column N_bound = Z - Z0; "
                       "d and s0 the parameters D_N and S_N0 of the analytic formula, index N_bound")],
      [toml_matrix("mean_excitation_energy", mee),
       toml_vector("d", d, toml_float),
       toml_vector("s0", s0, toml_float),
       toml_matrix("extended", extrows)])


def labelled_rows(src, name):
    """Rows of a 1D C array written one element group per line as /*Xx*/ a, b, ..."""
    m = re.search(re.escape(name) + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, flags=re.S)
    out = []
    for label, vals in re.findall(r"/\*\s*(\w+)\s*\*/([^/]*)", m.group(1)):
        out.append((label, [v.strip() for v in vals.split(",") if v.strip()]))
    return out


zs_rows = labelled_rows(psf, "PitchScatterFrequency::ionSizeAj_Zs")
z0_rows = labelled_rows(psf, "PitchScatterFrequency::ionSizeAj_Z0s")
a_rows = labelled_rows(psf, "PitchScatterFrequency::ionSizeAj_data")
length = int(re.search(r"ionSizeAj_len\s*=\s*(\d+)", psf).group(1))
element_Z = {label: vals[0] for label, vals in zs_rows}
z_intended = [element_Z[label] for label, vals in z0_rows for _ in vals]
z_dream = [v for _, vals in zs_rows for v in vals]
z_dream += ["0"] * (length - len(z_dream))  # C zero-initializes missing elements
z0 = [v for _, vals in z0_rows for v in vals]
a = [v for _, vals in a_rows for v in vals]
assert len(z_intended) == len(z0) == len(a) == length

write("ion_sizes.toml",
      [("source", "DREAM, src/Equations/PitchScatterFrequency.cpp (ionSizeAj_Zs, ionSizeAj_Z0s, ionSizeAj_data); "
                  "the values of Table 1 of L. Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), "
                  "doi:10.1017/S0022377818001113"),
       ("version_or_date", DREAM_VERSION),
       ("licence", LICENCE),
       ("generator", GENERATOR),
       ("description", "one entry per ion: z the nuclear charge, z0 the charge state, a the normalized "
                       "effective ion size a_j (multiply by 2/alpha); z_dream the nuclear charges as stored "
                       "in DREAM, where the Ar row has one entry too few, which shifts all later entries "
                       "(zero-padded to the declared length)")],
      [toml_vector("z", z_intended, toml_int),
       toml_vector("z0", z0, toml_int),
       toml_vector("a", a, toml_float),
       toml_vector("z_dream", z_dream, toml_int)])
