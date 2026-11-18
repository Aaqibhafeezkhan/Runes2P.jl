"""
Physical constants in SI units (electron rest energy in eV), with the values used by
DREAM (`src/Constants.cpp`) so that coefficients can be compared with DREAM to round-off.
These values are not of one CODATA set:

- `CLASSICAL_ELECTRON_RADIUS` and `ELECTRON_MASS` equal CODATA 2014 (P. J. Mohr, D. B.
  Newell and B. N. Taylor, Rev. Mod. Phys. 88, 035009 (2016),
  doi:10.1103/RevModPhys.88.035009).
- `ELEMENTARY_CHARGE` and `VACUUM_PERMITTIVITY` are CODATA 2014 rounded to 9 digits.
- `VACUUM_PERMEABILITY` is 4π × 10⁻⁷ H/m rounded to 9 digits. CODATA 2018 rounded to 9
  digits gives the same literal.
- `ELECTRON_REST_ENERGY` and `FINE_STRUCTURE_CONSTANT` equal CODATA 2018 (E. Tiesinga, P. J.
  Mohr, D. B. Newell, B. N. Taylor, Rev. Mod. Phys. 93, 025010 (2021),
  doi:10.1103/RevModPhys.93.025010).
- `SPEED_OF_LIGHT` is exact.

So `ELECTRON_MASS * SPEED_OF_LIGHT^2 / ELEMENTARY_CHARGE` lies 6.8e-9 below
`ELECTRON_REST_ENERGY`, relative.
"""
module Constants

const SPEED_OF_LIGHT = 299792458.0              # [m/s]
const CLASSICAL_ELECTRON_RADIUS = 2.8179403227e-15  # [m]
const ELEMENTARY_CHARGE = 1.60217662e-19        # [C]
const ELECTRON_MASS = 9.10938356e-31            # [kg]
const ELECTRON_REST_ENERGY = 0.51099895000e6    # m_e c² [eV]
const VACUUM_PERMITTIVITY = 8.85418782e-12      # [F/m]
const VACUUM_PERMEABILITY = 1.25663706e-6       # [H/m]
const FINE_STRUCTURE_CONSTANT = 0.0072973525693
"""
    PROTON_MASS

The proton mass [kg], 1.67262192369e-27, of CODATA 2018 (E. Tiesinga et al., Rev. Mod. Phys.
93, 025010 (2021), doi:10.1103/RevModPhys.93.025010), DREAM's `Constants::mH`
(`src/Constants.cpp`): the unit of the ion masses of the NRL Plasma Formulary.
"""
const PROTON_MASS = 1.67262192369e-27

"""
    BOHR_RADIUS

The Bohr radius [m], 5.29177210903e-11, of CODATA 2018 (E. Tiesinga et al., Rev. Mod. Phys.
93, 025010 (2021), doi:10.1103/RevModPhys.93.025010), DREAM's `Constants::a0`
(`src/Constants.cpp`).
"""
const BOHR_RADIUS = 5.29177210903e-11

"""
    RYDBERG_ENERGY

The Rydberg energy [eV], 13.605693123, of CODATA 2018 (E. Tiesinga et al., Rev. Mod. Phys.
93, 025010 (2021), doi:10.1103/RevModPhys.93.025010) to the digits of DREAM's `Constants::Ry`
(`src/Constants.cpp`).
"""
const RYDBERG_ENERGY = 13.605693123

end # module Constants
