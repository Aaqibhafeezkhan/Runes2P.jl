# p_max boundary conditions vs DREAM, DREAM's own examples/basic/basic.py
# (Embreus, 2020-11-23), run by reference/dream/cases/pmax_boundary.py with the settings
# the example writes and the overrides test_example_runaway.jl needs (n_cold prescribed,
# no particle source). The example's own initial 1 keV Maxwellian on p_max = 5 underflows
# to exactly zero on the upper part of the grid, and its own 4 steps of 0.25 microseconds
# cannot carry anything out to p = 5 either, so it cannot tell BC_F_0, BC_PHI_CONST and
# BC_DPHI_CONST apart; the case script overrides the initial f_hot with the example's own
# commented-out linear ramp (pmax - p) / pmax instead, non-negligible at p_max. Two more
# variants rerun BC_PHI_CONST and BC_DPHI_CONST with QUICK interpolation (the boundary
# with the scheme's face coefficients, not only centred). All five reproduce DREAM to
# round-off (observed 7.0e-12 to 7.7e-12), including :extrapolated_flux (DREAM's
# BC_DPHI_CONST), ported literally with its apparent Aᵖ index inconsistency; a throwaway
# check that used the "obviously correct" Aᵖ (the second term's own face) instead failed
# at 1.0e-3, so the round-off match here does discriminate the index, not just happen to
# pass regardless of it.
using HDF5

const DREAM_PMAX_BOUNDARY_DATA = joinpath(@__DIR__, "data", "dream_pmax_boundary.h5")

@testset "p_max boundary vs DREAM" begin
    h5open(DREAM_PMAX_BOUNDARY_DATA) do h5
        attr = attributes(h5)
        @test read(attr["example"]) == "examples/basic/basic.py"

        variants = ("f_zero" => (:f_zero, :central), "phi_const" => (:constant_flux, :central),
                    "dphi_const" => (:extrapolated_flux, :central),
                    "phi_const_quick" => (:constant_flux, :quick),
                    "dphi_const_quick" => (:extrapolated_flux, :quick))
        results = Dict{String,Any}()

        # the example's ions (D fully ionized, Ar neutral), built from the file rather than
        # restated: neutral Ar's bound electrons matter for partial screening even though it
        # contributes no free density
        nr = 10
        ion_names = read(h5["ions/name"])
        ion_z = read(h5["ions/Z"])
        ion_states = read(h5["ions/states"])
        ions = IonSpecies[]
        idx = 1
        for (z, iname) in zip(ion_z, ion_names)
            densities = ion_states[idx:idx+z]
            idx += z + 1
            push!(ions, IonSpecies(iname, z, repeat(densities, 1, nr)))
        end

        for (name, (p_upper_boundary, scheme)) in variants
            @testset "$name" begin
                grp = h5[name]
                setting(key) = only(read(grp["settings/$key"]))
                # E and T are prescribed uniform over the example's 10 radii and 1 time
                setting_uniform(key) = (v = read(grp["settings/$key"]); @assert allequal(v); first(v))

                # the example's values (DREAM defaults where the example sets nothing),
                # unchanged by the overrides
                @test (setting("hottailgrid/np"), setting("hottailgrid/pmax"), setting("hottailgrid/nxi")) ==
                      (100, 5.0, 30)
                @test setting_uniform("eqsys/E_field/data/x") == 50.0
                @test setting_uniform("eqsys/T_cold/data/x") == 10.0
                @test setting("eqsys/n_cold/data/x") == 1e20   # override: prescribed at the free D density
                @test (setting("radialgrid/B0"), setting("radialgrid/a"), setting("radialgrid/nr")) == (5.0, 0.22, 10)
                @test setting("collisions/collfreq_mode") == 2 # COLLFREQ_MODE_FULL (default)
                @test setting("collisions/collfreq_type") == 3 # PARTIALLY_SCREENED
                @test setting("collisions/lnlambda") == 3      # LNLAMBDA_THERMAL (default)
                @test setting("eqsys/f_hot/adv_interp/p1") == (scheme === :central ? 1 : 5)  # AD_INTERP_QUICK
                @test setting("eqsys/f_hot/adv_interp/p2") == (scheme === :central ? 1 : 5)
                @test setting("eqsys/f_hot/particleSource") == 1  # override: PARTICLE_SOURCE_ZERO
                @test setting("solver/type") == 1               # LINEAR_IMPLICIT (the example's own)

                density = setting("eqsys/n_cold/data/x")
                mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
                rad = RadialGrid(read(h5["grid/r_f"]))
                g = cylindrical_grid(mom, rad)
                E = fill(setting_uniform("eqsys/E_field/data/x"), nr)
                T = fill(setting_uniform("eqsys/T_cold/data/x"), nr)
                plasma = PlasmaState(cold_electron_temperature=T, cold_electron_density=fill(density, nr),
                                     ions=ions, electric_field=E,
                                     magnetic_field=fill(setting("radialgrid/B0"), nr))
                settings = CollisionSettings(mode=:full, screening=:partial, coulomb_logarithm=:thermal,
                                             ion_size_table=:dream, screening_window=:dream, bremsstrahlung=:neglect)
                op = fokker_planck_operator(g, plasma, settings; scheme, p_upper_boundary, synchrotron=false)
                f_initial = read(grp["f_hot_initial"])
                # f itself is O(1) (the raw ramp), not O(n_cold): scale the unknown to it,
                # not to the density
                layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g;
                                                            scale=maximum(abs, f_initial))])
                u0 = zeros(length(layout))
                for ir in 1:nr
                    kinetic_view(u0, layout, 1, ir) .= f_initial[:, :, ir]
                end
                times = read(grp["t"])
                system = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                          FokkerPlanckTerm(:hot_tail_distribution, op))])
                solution = integrate(system, u0, times)
                @test all(solution.statistics.accepted)
                # a linear problem: DREAM's LINEAR_IMPLICIT solver takes one linear solve
                # per step (`solver/iterations` is 1 throughout); Runes2P's exact Newton
                # Jacobian takes 2 (one update, one convergence check), as in
                # test_time_stepping_dream.jl
                @test all(==(2), solution.statistics.newton_iterations)

                error = 0.0
                f_final = read(grp["f_hot_final"])
                for ir in 1:nr
                    a = f_final[:, :, ir]
                    b = kinetic_view(solution.states[:, end], layout, 1, ir)
                    cut = 1e-12 * maximum(abs, a)
                    error = max(error, maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut))
                end

                # all five variants reproduce DREAM to round-off (observed 7.0e-12 to
                # 7.7e-12 over the 8 steps)
                @test error < 1e-8

                # n_re collects the flux through p_max: each increment is dt times the
                # flux at the end of the step (backward Euler in both codes), a check that
                # momentum_flux at p_max is consistent
                n_re = read(grp["n_re"])   # (nr, nt + 1)
                for ir in 1:nr
                    fluxes = [momentum_flux(op, reshape(solution.states[:, k], size(g)), length(mom.p_faces), ir)
                              for k in 2:length(times)]
                    @test all(isapprox.(diff(n_re[ir, :]), fluxes .* diff(times); rtol=1e-8))
                end

                results[name] = kinetic_view(solution.states[:, end], layout, 1, 1)
            end
        end

        @testset "negative control: the boundary conditions differ" begin
            relerr(a, b) = maximum(abs, a - b) / maximum(abs, a)
            @test relerr(results["f_zero"], results["phi_const"]) > 1e-5
            @test relerr(results["phi_const"], results["dphi_const"]) > 1e-5
            @test relerr(results["phi_const"], results["phi_const_quick"]) > 1e-5
            @test relerr(results["dphi_const"], results["dphi_const_quick"]) > 1e-5
            # each Runes2P boundary must match only its own DREAM variant, not the others
            for (name, other) in (("f_zero", "phi_const"), ("phi_const", "dphi_const"), ("dphi_const", "f_zero"))
                a = read(h5["$other/f_hot_final"])[:, :, 1]
                @test relerr(a, results[name]) > 1e-5
            end
        end
    end
end
