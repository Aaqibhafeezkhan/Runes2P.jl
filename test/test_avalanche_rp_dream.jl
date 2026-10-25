# The Rosenbluth-Putvinski avalanche against DREAM (reference/dream/cases/rp_avalanche.py):
# the kinetic source on the hot-tail grid of DREAM's disruption example, the effective
# critical field, p⋆ and the fluid growth rate of the external avalanche, at the state of
# one step of 1e-12 s in a prescribed plasma of D and Ar at 5 and 3 eV, with a field of
# +30 V/m (above the effective critical field, knock-ons at ξ* > 0) and -0.05 V/m (below it,
# ξ* < 0, decay). The source is the same formula on the same cells; the effective critical
# field and p⋆ are solved to 1e-13 and 1e-12 in Runes2P, to 2e-3 and 1e-3 in DREAM, which
# interpolates the pitch averages as well. The matched rate of the external avalanche,
# (Γ_ext⁻² + Γ_fluid⁻²)^(-1/2) (ExternalAvalancheTerm.cpp:51-56), is not in DREAM's output and
# is not compared here on its own; it enters the runaway density of the coupled comparison
# of test_disruption_dream.jl, which compares n_re per radius and so sees the positive
# branch at its second radius in the quench. Needs no ADF11 cache.
using HDF5

@testset "Rosenbluth-Putvinski avalanche against DREAM" begin
    data = h5open(reference_file("dream_rp_avalanche.h5"), "r") do h5
        Dict(name => read(h5[name]) for name in keys(h5))
    end
    nr = length(data["n_re"])
    radial = RadialGrid(uniform_r_faces(nr, data["minor_radius"]))
    @test radial.r_centers ≈ data["r_grid"] rtol = 1e-12
    grid = cylindrical_grid(MomentumGrid(data["p1_f"], data["p2_f"]), radial)
    layout = UnknownLayout(radial,
        [RadialUnknown(:runaway_density), RadialUnknown(:total_electron_density),
            RadialUnknown(:electric_field), KineticUnknown(:f, grid)])
    u = zeros(length(layout))
    radial_values(u, layout, 1) .= data["n_re"]
    radial_values(u, layout, 2) .= data["n_tot"]
    radial_values(u, layout, 3) .= data["E_field"]
    term = RosenbluthPutvinskiSourceTerm(:f, grid; p_cutoff = data["p_cut"])
    source = Runes2P.rhs!(zeros(length(u)), term, u, layout, 4, 0.0)

    @testset "the kinetic source per cell" begin
        # DREAM's S_ava, (p, ξ, radius) as HDF5 gives it to Julia: every cell 1 + 3.0e-9 times
        # DREAM's, the difference of DREAM's constants of the prefactor e⁴/((4πε₀ mₑc)² c)
        # against Runes2P's r₀² c of CODATA 2018. The sign is the one of the set: DREAM v26.5
        # stores the source with the sign of the operator.
        S = DREAM_REFERENCE_SETS[reference_set("dream_rp_avalanche.h5")].S_ava_sign .*
            data["S_ava"]
        for ir in 1:nr
            ours = kinetic_view(source, layout, 4, ir)
            @test findall(!iszero, ours) == findall(!iszero, S[:, :, ir])
            @test ours ≈ S[:, :, ir] rtol = 1e-8
        end
    end

    @testset "the effective critical field and the fluid growth rate" begin
        settings = CollisionSettings(;
            screening = :partial, coulomb_logarithm = :energy_dependent,
            screening_window = :dream,
            ion_size_table = :dream, bremsstrahlung_constant = :dream)
        # (radius, charge state) as HDF5 gives the arrays of (charge state, radius)
        deuterium, argon = permutedims(data["deuterium"]), permutedims(data["argon"])
        for ir in 1:nr
            plasma = PlasmaState(; cold_electron_temperature = [data["T_cold"][ir]],
                cold_electron_density = [data["n_cold"][ir]],
                electric_field = [data["E_field"][ir]], magnetic_field = [data["B0"]],
                ions = [IonSpecies("D", 1, deuterium[:, ir:ir]),
                    IonSpecies("Ar", 18, argon[:, ir:ir])])
            rate, field, p_star = fluid_avalanche_growth_rate(plasma, 1, data["n_tot"][ir]; settings)
            @test field ≈ data["Eceff"][ir] rtol = 3e-3
            @test p_star ≈ data["pStar"][ir] rtol = 3e-3
            @test rate ≈ data["GammaAva"][ir] rtol = 1e-2
        end
        @test data["GammaAva"][1] > 0 && data["GammaAva"][2] < 0
    end
end
