#!/usr/bin/env python3
"""Fill the cache of ADF11 files of OpenADAS that Runes2P reads.

    python3 scripts/fetch_adas.py [--cache DIR] [--dream-cache DIR] [ELEMENT ...]

Run on a machine with internet access; a cluster whose compute nodes have none fills the
cache from a login node. The cache is ENV RUNES2P_ADAS_CACHE or ~/.cache/Runes2P/adas, as
`adas_cache_directory()` of Runes2P. The OpenADAS terms do not allow redistributing the data with a modelling code, so
the cache is never part of the repository.

A file present in DREAM's cache, given with --dream-cache, is copied from there. The default
is reference/dream/dream-boltzmann-eqterm-93d27e17/DREAM/tools/cache, the checkout of the set
whose DREAM commit the code cites (reference/dream/build_all.sh makes it). A copied file is
the file DREAM's comparisons read, and the high-density files of hydrogen are only there
(OpenADAS does not serve them). Any other file is downloaded from
https://open.adas.ac.uk/download/adf11/<class><yy>/<class><yy>_<element>.dat.

The table below is that of DREAM's tools/elements.json and of ADAS_FILES, read from
data/adas_files.toml; the test of the cache checks that the two agree.
"""
import argparse
import os
import shutil
import sys
import urllib.request

CLASSES = ("acd", "scd", "ccd", "plt", "prb")
HYDROGEN = {"acd": "acd12_h.high_dens.dat", "scd": "scd12_h.high_dens.dat", "ccd": "96",
            "plt": "plt12_h.high_dens.dat", "prb": "prb12_h.high_dens.dat"}
MIXED_96_89 = {"acd": "96", "scd": "96", "ccd": "89", "plt": "96", "prb": "96"}
FILES = {
    "H": HYDROGEN, "D": HYDROGEN, "T": HYDROGEN,
    "He": dict.fromkeys(CLASSES, "96"), "Li": MIXED_96_89, "Be": MIXED_96_89,
    "B": dict.fromkeys(CLASSES, "89"), "C": dict.fromkeys(CLASSES, "96"),
    "O": MIXED_96_89, "Ne": MIXED_96_89, "Ar": dict.fromkeys(CLASSES, "89"),
    "Fe": {"acd": "85", "scd": "85", "ccd": "89", "plt": "89", "prb": "89"},
    "W": {"acd": "50", "scd": "50", "ccd": "89", "plt": "42", "prb": "50"},
}


def file_name(element, cls):
    entry = FILES[element][cls]
    if entry.endswith(".dat"):
        return entry
    file_element = "h" if cls != "ccd" and element in ("D", "T") else element.lower()
    return f"{cls}{entry}_{file_element}.dat"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=os.environ.get("RUNES2P_ADAS_CACHE",
                                                          os.path.expanduser("~/.cache/Runes2P/adas")))
    # the ADAS cache of the boltzmann-eqterm set, the DREAM commit that the code cites
    parser.add_argument("--dream-cache", default=os.path.join(root, "reference", "dream", "dream-boltzmann-eqterm-93d27e17",
                                                              "DREAM", "tools", "cache"))
    parser.add_argument("elements", nargs="*", default=sorted(FILES))
    args = parser.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    missing = []
    for element in args.elements:
        if element not in FILES:
            sys.exit(f"no ADF11 files listed for the element {element}")
        for cls in CLASSES:
            name = file_name(element, cls)
            target = os.path.join(args.cache, name)
            if os.path.isfile(target):
                continue
            local = os.path.join(args.dream_cache, name)
            if os.path.isfile(local):
                shutil.copyfile(local, target)
                print(f"{name}: copied from DREAM's cache")
                continue
            if name.endswith(".high_dens.dat"):
                missing.append(name)
                continue
            url = f"https://open.adas.ac.uk/download/adf11/{name.split('_')[0]}/{name}"
            data = urllib.request.urlopen(url, timeout=60).read().decode()
            if data.startswith("<!DOCTYPE"):
                missing.append(name)
                continue
            with open(target, "w") as f:
                f.write(data)
            print(f"{name}: downloaded")
    if missing:
        sys.exit("not found (the high-density hydrogen files come from DREAM's cache only): " + ", ".join(missing))
    print(f"cache {args.cache} complete")


if __name__ == "__main__":
    main()
