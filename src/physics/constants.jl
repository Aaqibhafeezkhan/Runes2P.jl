"""
Physical constants in SI units (electron rest energy in eV), with the values used by
DREAM (`src/Constants.cpp`) so that coefficients can be compared with DREAM to round-off.
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
