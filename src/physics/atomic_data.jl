# Tabulated atomic data of DREAM (SlowingDownFrequency.cpp, PitchScatterFrequency.cpp,
# MeanExcitationEnergy_Extended.cpp), read from data/mean_excitation_energies.toml and
# data/ion_sizes.toml. scripts/port_atomic_data.py writes both files from DREAM's sources.
# DREAM is MIT licensed, see THIRD_PARTY_NOTICES.md.

# the parsed tables of the two data files, each read once
const _MEAN_EXCITATION_ENERGY_DATA = read_data_file("mean_excitation_energies")["data"]
const _ION_SIZE_DATA = read_data_file("ion_sizes")["data"]

"""
    MEAN_EXCITATION_ENERGY

Mean excitation energy [eV], row Z, column Z0 + 1, for Z = 1 to 18 (Sauer, Sabin and
Oddershede, J. Chem. Phys. 148, 174307 (2018), through DREAM's
`SlowingDownFrequency.cpp`). Read from `data/mean_excitation_energies.toml`.
"""
const MEAN_EXCITATION_ENERGY = float_matrix(_MEAN_EXCITATION_ENERGY_DATA["mean_excitation_energy"])

"""
    MEAN_EXCITATION_ENERGY_D

Parameter D_N of the analytic mean excitation energy formula, index N_bound (Sauer, Sabin
and Oddershede, J. Chem. Phys. 148, 174307 (2018), through DREAM's
`SlowingDownFrequency.cpp`). Read from `data/mean_excitation_energies.toml`.
"""
const MEAN_EXCITATION_ENERGY_D = float_vector(_MEAN_EXCITATION_ENERGY_DATA["d"])

"""
    MEAN_EXCITATION_ENERGY_S0

Parameter S_N0 of the analytic mean excitation energy formula, index N_bound (Sauer, Sabin
and Oddershede, J. Chem. Phys. 148, 174307 (2018), through DREAM's
`SlowingDownFrequency.cpp`). Read from `data/mean_excitation_energies.toml`.
"""
const MEAN_EXCITATION_ENERGY_S0 = float_vector(_MEAN_EXCITATION_ENERGY_DATA["s0"])

"""
    MEAN_EXCITATION_ENERGY_EXTENDED

Extended mean excitation energy [eV], row Z, column N_bound = Z - Z0 (semi-empirical model
of DREAM's `tools/Generate_MeanExcitationEnergy.py`, rows 19 to 86, through DREAM's
`MeanExcitationEnergy_Extended.cpp`). Read from `data/mean_excitation_energies.toml`.
"""
const MEAN_EXCITATION_ENERGY_EXTENDED = float_matrix(_MEAN_EXCITATION_ENERGY_DATA["extended"])

"""
    ION_SIZE_Z

Effective ion size table (Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), Table 1,
through DREAM's `PitchScatterFrequency.cpp`): nuclear charge. Read from
`data/ion_sizes.toml`.
"""
const ION_SIZE_Z = int_vector(_ION_SIZE_DATA["z"])

"""
    ION_SIZE_Z0

Charge state for each entry of the effective ion size table (Hesslow et al., J. Plasma
Phys. 84, 905840605 (2018), Table 1, through DREAM's `PitchScatterFrequency.cpp`). Read
from `data/ion_sizes.toml`.
"""
const ION_SIZE_Z0 = int_vector(_ION_SIZE_DATA["z0"])

"""
    ION_SIZE_A

Normalized effective ion size a_j (multiply by 2/α) of the effective ion size table
(Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), Table 1, through DREAM's
`PitchScatterFrequency.cpp`). Read from `data/ion_sizes.toml`.
"""
const ION_SIZE_A = float_vector(_ION_SIZE_DATA["a"])

"""
    ION_SIZE_Z_DREAM

Nuclear charges as stored in DREAM's `PitchScatterFrequency.cpp` (`ionSizeAj_Zs`), where
the Ar row has one entry too few, shifting all later entries (zero-padded to the declared
length). Read from `data/ion_sizes.toml`.
"""
const ION_SIZE_Z_DREAM = int_vector(_ION_SIZE_DATA["z_dream"])
