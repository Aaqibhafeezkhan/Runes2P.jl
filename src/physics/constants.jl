"""
Physical constants in SI units, with the values used by DREAM (`src/Constants.cpp`)
so that coefficients can be compared with DREAM to round-off.
"""
module Constants

const c = 299792458.0              # speed of light [m/s]
const r0 = 2.8179403227e-15        # classical electron radius [m]
const ec = 1.60217662e-19          # elementary charge [C]
const me = 9.10938356e-31          # electron mass [kg]
const mc2inEV = 0.51099895000e6    # electron rest energy [eV]
const eps0 = 8.85418782e-12        # vacuum permittivity [F/m]
const mu0 = 1.25663706e-6          # vacuum permeability [H/m]
const alpha = 0.0072973525693      # fine-structure constant

end # module Constants
