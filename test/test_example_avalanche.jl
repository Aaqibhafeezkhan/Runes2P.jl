# DREAM's Boltzmann knock-on example, examples/avalanche/generateCylindrical.py (Ola
# Embréus, 2026-03-05), run without overrides by reference/dream/cases/example_avalanche.py.
# The knock-on operator is compared with DREAM's C_boltz on the example's hot-tail grid and
# distribution at the first, middle and last output times. The test checks the example's
# values in the recorded settings and rebuilds its grid from them.
using HDF5

const DREAM_EXAMPLE_AVALANCHE_DATA = reference_file("dream_example_avalanche.h5")

@testset "DREAM example avalanche/generateCylindrical.py" begin
    h5open(DREAM_EXAMPLE_AVALANCHE_DATA) do h5
        attr = attributes(h5)
        @test read(attr["example"]) == "examples/avalanche/generateCylindrical.py"
        @test read(attr["example_author"]) == "Ola Embreus"
        setting(key) = only(read(h5["settings/$key"]))

        # the example's grid and knock-on settings
        @test (setting("hottailgrid/np"), setting("hottailgrid/pmax")) == (200, 30.0)
        @test (setting("hottailgrid/npsep"), setting("hottailgrid/psep")) == (40, 0.7)
        @test setting("eqsys/n_re/avalanche") == 5            # AVALANCHE_MODE_BOLTZMANN_KNOCK_ON
        @test setting("eqsys/n_re/pCutAvalanche") == 0.1
        @test setting("eqsys/T_cold/data/x") == 10.0
        @test setting("collisions/collfreq_mode") == 1        # SUPERTHERMAL
        @test setting("collisions/collfreq_type") == 3        # PARTIALLY_SCREENED
        @test setting("collisions/lnlambda") == 2             # ENERGY_DEPENDENT
        # pitch faces with geometric growth, nxiexp = 1.5: 30 faces, 29 cells
        xi_faces = read(h5["settings/hottailgrid/xi_f"])
        @test xi_faces ≈ reverse(1 .- 2 .* range(0, 1; length = 30) .^ 1.5) rtol = 1e-14
        p_faces = read(h5["grid/p1_f"])
        @test p_faces ≈ biuniform_p_faces(200, 40, 0.0, 0.7, 30.0) rtol = 1e-14
        @test read(h5["grid/p2_f"]) ≈ xi_faces rtol = 1e-14

        mom = MomentumGrid(p_faces, read(h5["grid/p2_f"]))
        g = cylindrical_grid(mom, RadialGrid(read(h5["grid/r_f"])))
        jacobian = stack(phase_space_jacobian(g, ir) for ir in 1:radial_cell_count(g))
        @test all(r -> isapprox(r, 4π^2; rtol = 1e-14), read(h5["grid/Vprime"]) ./ jacobian)
        p_cut = setting("eqsys/n_re/pCutAvalanche")
        f = read(h5["f_hot"])          # (np, nξ, nr, stored times)
        C_dream = read(h5["C_boltz"])  # C_boltz[n] from f_hot[n]
        n_tot = read(h5["n_tot"])      # (nr, stored times)

        dream = KnockOnOperator(
            g, g, p_cut; dream_compatible = true, primary_pitch_points = 1, table_size = 80)
        default = KnockOnOperator(g, g, p_cut)
        for n in axes(C_dream, 4)
            C = zeros(size(g))
            apply!(C, dream, f[:, :, :, n], n_tot[:, n])
            @test max_rel_diff(C_dream[:, :, :, n], C) < 1e-12  # observed 3.5e-13
            # Runes2P's default kernel (ξ* over the receiving interval, primary averaged over its
            # pitch cell) produces the same knock-ons and places them differently
            C_default = zeros(size(g))
            apply!(C_default, default, f[:, :, :, n], n_tot[:, n])
            @test density(g, C_default) ≈ density(g, C) rtol = 1e-12
        end
    end
end
