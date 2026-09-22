# Multi-step backward Euler compared with a DREAM run (reference/dream/cases/time_stepping.py):
# Fokker-Planck operator with prescribed, constant plasma plus DREAM's explicit knock-on
# term, advanced over DREAM's time points by Newton with the sparse direct solver and with
# GMRES preconditioned by the per-radius LU. With the knock-on term explicit, the
# preconditioner is the whole Newton matrix, so GMRES must take one iteration per Newton
# iteration and reproduce the direct solve.
using HDF5

const DREAM_TIME_STEPPING_DATA = joinpath(@__DIR__, "data", "dream_time_stepping.h5")

@testset "time stepping vs DREAM" begin
    h5open(DREAM_TIME_STEPPING_DATA) do h5
        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        g = cylindrical_grid(mom, RadialGrid(read(h5["grid/r_f"])))
        nr = radial_cell_count(g)
        names = read(h5["ions/name"])
        Zs = read(h5["ions/Z"])
        Z0s = read(h5["ions/Z0"])
        n = read(h5["ions/n"])
        ions = [IonSpecies(names[k], Zs[k], Z0s[k], n[:, k]) for k in eachindex(Zs)]
        p_cut = read(h5["p_cut"])
        variants = (
            # observed largest differences over the 20 steps: 2.0e-14 and 3.8e-13, at
            # round-off level from the first step on
            "superthermal" => (CollisionSettings(mode=:superthermal, screening=:partial,
                                                 coulomb_logarithm=:energy_dependent, ion_size_table=:dream), 1e-13),
            "full" => (CollisionSettings(ion_size_table=:dream), 1e-11),
        )
        for (variant, (settings, rtol)) in variants
            @testset "$variant" begin
                grp = h5[variant]
                times = read(grp["t"])
                f_dream = read(grp["f_hot"])      # (np, nξ, nr, nt + 1)
                n_tot = read(grp["n_tot"])        # (nr, nt + 1), prescribed and constant
                @test all(==(2), read(grp["newton_iterations"]))
                # n_tot comes from the prescribed ions, so one target density serves all steps
                @test n_tot == repeat(n_tot[:, 1], 1, size(n_tot, 2))
                # DREAM's Rosenbluth-Putvinski source from n_re (hot-tail grid only, not
                # modelled here) must be negligible: n_re stays near 1e-21 of the hot density
                @test maximum(read(grp["n_re"])) < 1e-18 * minimum(density(g, f_dream[:, :, :, 1]))

                plasma = PlasmaState(cold_electron_temperature=read(grp["T_cold"])[:, 1],
                                     cold_electron_density=read(grp["n_cold"])[:, 1], ions=ions,
                                     electric_field=read(h5["E"]),
                                     magnetic_field=fill(read(h5["B"]), nr))
                fokker_planck = fokker_planck_operator(g, plasma, settings; synchrotron=true)
                knockon = KnockOnOperator(g, g, p_cut; dream_compatible=true, primary_pitch_points=1,
                                          table_size=80)
                layout = UnknownLayout(g.radial_grid, [KineticUnknown(:hot_tail_distribution, g; scale=1e16)])
                equation = Equation(:hot_tail_distribution, TimeDerivative(),
                                    FokkerPlanckTerm(:hot_tail_distribution, fokker_planck),
                                    KnockOnTerm(:hot_tail_distribution, knockon; target_density=n_tot[:, 1],
                                                treatment=:explicit))
                system = EquationSystem(layout, [equation])
                u0 = zeros(length(layout))
                for ir in 1:nr
                    kinetic_view(u0, layout, 1, ir) .= f_dream[:, :, ir, 1]
                end
                for (solver_name, solver) in ("direct" => NewtonSolver(),
                                              "GMRES" => NewtonSolver(linear_solver=GMRES()))
                    @testset "$solver_name" begin
                        solution = integrate(system, u0, times; solver)

                        stats = solution.statistics
                        @test all(stats.accepted)
                        @test stats.time == times[2:end]
                        # a linear problem: the solve, then the check
                        @test all(==(2), stats.newton_iterations)
                        if solver_name == "GMRES"
                            @test stats.krylov_iterations == stats.newton_iterations
                            @test stats.preconditioner_updates == stats.newton_iterations
                        end
                        errors = map(eachindex(times)) do m
                            worst = 0.0
                            for ir in 1:nr
                                a = f_dream[:, :, ir, m]
                                b = kinetic_view(solution.states[:, m], layout, 1, ir)
                                cut = 1e-12 * maximum(abs, a)
                                worst = max(worst, maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a)
                                                           if abs(a[k]) > cut))
                            end
                            worst
                        end
                        @test errors[1] == 0
                        @test maximum(errors) < rtol
                    end
                end
            end
        end
    end
end
