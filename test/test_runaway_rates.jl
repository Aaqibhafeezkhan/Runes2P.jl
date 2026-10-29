# Fluid runaway rate formulas against DREAM's own values
# (reference/dream/cases/runaway_rates.py): critical and Dreicer fields, thermal collision
# time, Connor-Hastie rate with and without corrections, and the avalanche growth rate at
# DREAM's effective critical momentum. The kinetic measurements are in
# test/test_runaway_physics.jl.
using HDF5

const DREAM_RUNAWAY_RATE_DATA = reference_file("dream_runaway_rates.h5")

@testset "fluid runaway rates vs DREAM" begin
    h5open(DREAM_RUNAWAY_RATE_DATA) do h5
        temperatures = read(h5["temperature"])
        corrected = h5["connor_hastie_corrected"]
        fields = read(corrected["E_field"])
        # DREAM's own densities of the step: the runaways generated in it lower n_cold,
        # which the fields and rates are evaluated at, so both codes use the same inputs
        densities = read(corrected["n_cold"])
        total_densities = read(corrected["n_tot"])
        # DREAM ran with lnlambda = CONSTANT: E_c from the relativistic (constant) Coulomb
        # logarithm, E_D and τ_ee from the thermal one
        relativistic = [relativistic_coulomb_logarithm(n, T)
                        for (T, n) in zip(temperatures, densities)]
        thermal = [thermal_coulomb_logarithm(n, T)
                   for (T, n) in zip(temperatures, densities)]

        @testset "fields and collision time" begin
            # the two codes use slightly different values of the physical constants, which
            # shows up at 3e-9 here and, amplified by the exponential, at 1e-7 in the rate
            # the strongest-field case generates enough runaways within the step that n_cold
            # moves while DREAM evaluates the field, which shows up at 4e-8
            @test critical_field.(densities, relativistic) ≈ read(corrected["Ecfree"]) rtol = 1e-7
            @test dreicer_field.(densities, temperatures, thermal) ≈
                  read(corrected["EDreic"]) rtol = 1e-8
            @test thermal_collision_time.(densities, temperatures, thermal) ≈
                  read(corrected["tauEETh"]) rtol = 1e-8
        end

        @testset "Connor-Hastie rate ($variant)" for (variant, corrections) in ((
            "connor_hastie", false),
            ("connor_hastie_corrected", true))
            reference = read(h5[variant]["gammaDreicer"])
            rates = [connor_hastie_rate(densities[k], temperatures[k], fields[k], 1.0;
                         thermal_coulomb_logarithm = thermal[k],
                         relativistic_coulomb_logarithm = relativistic[k], corrections)
                     for k in eachindex(fields)]
            # the rate spans 24 orders of magnitude over these cases
            @test rates ≈ reference rtol = 1e-6
        end

        @testset "avalanche growth rate at DREAM's critical momentum" begin
            # Γ = 4π r₀² c n_tot / p_c²; with DREAM's effective p_c (Hesslow et al. 2019,
            # `fluid/pCrit`) this must reproduce DREAM's fluid growth rate exactly. The
            # plasma is fully ionized deuterium, so n_tot is the ion density.
            p_crit = read(corrected["pCrit"])
            @test avalanche_growth_rate.(total_densities, p_crit) ≈
                  read(corrected["GammaAva"]) rtol = 1e-8
            # the test-particle critical momentum of Rosenbluth-Putvinski is smaller than
            # DREAM's effective one (which includes screening and pitch-angle scattering),
            # so it overestimates the growth rate
            rp = critical_momentum.(fields, read(corrected["Ecfree"]))
            @test all(rp .< p_crit)
            @test all(avalanche_growth_rate.(total_densities, rp) .>
                      read(corrected["GammaAva"]))
        end

        @testset "the neural network differs from the asymptotic formula" begin
            # both are published kinetic references; Connor-Hastie is asymptotic in E/E_D,
            # so the two differ most in the weak-field and low-temperature cases
            network = read(h5["neural_network"]["gammaDreicer"])
            reference = read(corrected["gammaDreicer"])
            ratio = network ./ reference
            @test all(isfinite, ratio)
            over_dreicer = read(corrected["E_field"]) ./ read(corrected["EDreic"])
            # where Connor-Hastie's asymptotic expansion applies (E/E_D ≈ 0.1) the two agree
            # within 20%; at E/E_D below 0.011 it overestimates by orders of magnitude
            # (measured 1.5e-7 to 2e-4 of the network rate), which is why the kinetic
            # comparison of test_runaway_physics.jl uses the stronger fields
            strong = over_dreicer .> 0.09
            @test all(0.8 .< ratio[strong] .< 1.3)
            @test all(ratio[over_dreicer .< 0.011] .< 1e-3)
        end
    end
end
