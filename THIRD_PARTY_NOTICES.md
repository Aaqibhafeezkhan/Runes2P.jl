# Third-party notices

## DREAM

Parts of this package are ported from or closely follow DREAM (https://github.com/chalmersplasmatheory/DREAM): among them the knock-on (Møller) collision operator of the `boltzmann-eqterm` branch, the collision frequencies and their tables of mean excitation energies and ion sizes, the Braams–Karney conductivity table, the runaway rates and the effective critical field, the kinetic ionization fits, the way DREAM interpolates the ADAS rate coefficients, and the flux limiters.
Ported functions name their DREAM source file in the docstring or in the file header.
The bicubic interpolation of the ADAS coefficients follows the scheme that DREAM uses through GSL; it is written here from the textbook algorithm and contains no GSL code.
DREAM is distributed under the following license:

```
MIT License

Copyright (c) 2020 Mathias Hoppe and Ola Embreus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CHIANTI

The fit parameters of the electron-impact ionization cross sections for H, Ne and Ar (`data/chianti_ionization.toml`) are copied unchanged from the `<ion>.diparams` and `<ion>.easplom` files (direct ionization and excitation-autoionization) of the CHIANTI atomic database, version 11.0.2.
CHIANTI is a collaborative project involving George Mason University, the University of Michigan (USA), the University of Cambridge (UK) and NASA Goddard Space Flight Center (USA).
Work that uses these data should cite Dere et al., Astron. Astrophys. Suppl. Ser. 125, 149 (1997), Dufresne et al., Astrophys. J. 974, 71 (2024), and for the ionization data Dere, Astron. Astrophys. 466, 771 (2007).

## ADAS

No ADAS data file is distributed with this package.
The rate coefficients of the ADF11 class are read at run time from a cache that the user fills: from OPEN-ADAS (https://open.adas.ac.uk) under its terms and conditions, and for H, D and T, except charge exchange, from the high-density hydrogen files that DREAM's repository provides.
The fit parameters in `data/kinetic_ionization_bethe_fits.toml`, and DREAM's fits in `data/kinetic_ionization_fits.toml`, are derived by least-squares fits to ADAS SCD coefficients; no tabulated ADAS value is reproduced.
ADAS: H. P. Summers, The ADAS User Manual, version 2.6 (2004), https://www.adas.ac.uk.

## NIST

Ionization energies are taken from the NIST Atomic Spectra Database (Kramida, Ralchenko, Reader and the NIST ASD Team, https://doi.org/10.18434/T4W30F), through the table that DREAM generates from it (`src/Atomics/nistdata_ionization.cpp`, retrieved on 2023-04-28 by DREAM's `tools/get_nist.py`).
The NIST Atomic Spectra Database is NIST Standard Reference Database 78, copyright by the U.S. Secretary of Commerce on behalf of the United States of America.
The package holds a small extract of its ionization energies in `data/ionization_energies.toml`, as DREAM does.

## GSL

`reference/gsl/bicubic_reference.c` is a separate generator of reference values, not part of the package; it links the GNU Scientific Library 2.8 (GPL-3.0), and its output, a table of numbers, is the reference set `gsl-2.8`.
