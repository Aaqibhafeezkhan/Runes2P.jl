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
                                                 coulomb_logarithm=:energy_dependent, ion_size_table=:dream,
                                                 screening_window=:dream), 1e-13),
            "full" => (CollisionSettings(ion_size_table=:dream, screening_window=:dream), 1e-11),
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
                fokker_planck = fokker_planck_operator(g, plasma, settings; synchrotron=true,
                                                       scheme=:central)  # DREAM's default
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
                            # one factorization per step: the second iteration has the same matrix
                            @test all(==(1), stats.preconditioner_updates)
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

# DREAM's adaptive stepper on the superthermal variant of the same setup
# (reference/dream/cases/time_stepping_adaptive.py): two half steps and one full step
# from the same state, the error per non-trivial unknown |x_full - x_half|₂ over
# (abstol + reltol |x_full|₂), the next step 0.95 dt / max error. Runes2P's StepDoubling
# with the same unknowns (f_hot, the hot density and current as algebraic moments, DREAM's
# n_hot and j_hot, and the runaway density fed by the flux through p_max) must take the
# same steps and reject the same attempts. DREAM logs every check with its error, which
# the file keeps (`stepper_log`). The comparison covers the first 23 checks, where f_hot
# binds in both codes: from check 24 on DREAM's step follows the tolerances of the wrong
# unknowns (its GetErrorScale reads them by the position in the non-trivial list, its
# acceptance by the unknown ID), which accepts steps with a reported error above 1 and no
# single set of tolerances reproduces.
const DREAM_ADAPTIVE_DATA = joinpath(@__DIR__, "data", "dream_time_stepping_adaptive.h5")

# the errors DREAM logged, one per check, in order
function _dream_logged_errors(log::AbstractString)
    errors = Float64[]
    for line in split(log, '\n')
        m = match(r"max error:\s+([0-9.eE+-]+)", line)
        m === nothing || push!(errors, parse(Float64, m.captures[1]))
    end
    return errors
end

@testset "adaptive stepper vs DREAM" begin
    h5open(DREAM_ADAPTIVE_DATA) do h5
        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        g = cylindrical_grid(mom, RadialGrid(read(h5["grid/r_f"])))
        nr = radial_cell_count(g)
        names = read(h5["ions/name"])
        ions = [IonSpecies(names[k], read(h5["ions/Z"])[k], read(h5["ions/Z0"])[k], read(h5["ions/n"])[:, k])
                for k in eachindex(names)]
        dt0, t_max, reltol_step = read(h5["dt0"]), read(h5["t_max"]), read(h5["reltol_step"])
        settings = CollisionSettings(mode=:superthermal, screening=:partial, coulomb_logarithm=:energy_dependent,
                                     ion_size_table=:dream, screening_window=:dream)
        for variant in ("check_every_step",)
            @testset "$variant" begin
                grp = h5[variant]
                check_every = Int(read(attributes(grp)["check_every"]))
                t_dream = read(grp["t"])
                f_initial, f_final = read(grp["f_hot_initial"]), read(grp["f_hot_final"])
                n_hot_dream, j_hot_dream = read(grp["n_hot"]), read(grp["j_hot"])
                n_tot = read(grp["n_tot"])
                logged = _dream_logged_errors(read(grp["stepper_log"]))
                plasma = PlasmaState(cold_electron_temperature=read(grp["T_cold"])[:, 1],
                                     cold_electron_density=read(grp["n_cold"])[:, 1], ions=ions,
                                     electric_field=read(h5["E"]), magnetic_field=fill(read(h5["B"]), nr))
                fokker_planck = fokker_planck_operator(g, plasma, settings; synchrotron=true, scheme=:central)
                knockon = KnockOnOperator(g, g, read(h5["p_cut"]); dream_compatible=true, primary_pitch_points=1,
                                          table_size=80)
                # DREAM checks the error of every non-trivial unknown. The moments n_hot
                # and j_hot follow f_hot; the runaway density n_re, fed by the flux
                # through p_max, gets DREAM's intended tolerance (1e-10 absolute);
                # j_re = e c n_re has the same error and is left out.
                unknowns = [KineticUnknown(:hot_tail_distribution, g; scale=1e16),
                            RadialUnknown(:hot_electron_density; scale=1e14),
                            RadialUnknown(:hot_electron_current_density; scale=1e3),
                            RadialUnknown(:runaway_density; scale=1e-10)]
                layout = UnknownLayout(g.radial_grid, unknowns)
                hot_term = FokkerPlanckTerm(:hot_tail_distribution, fokker_planck)
                equations = [Equation(:hot_tail_distribution, TimeDerivative(), hot_term,
                                      KnockOnTerm(:hot_tail_distribution, knockon; target_density=n_tot[:, 1],
                                                  treatment=:explicit)),
                             Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:hot_tail_distribution),
                                      LinearTerm(:hot_electron_density; coefficient=-1.0)),
                             Equation(:hot_electron_current_density, Algebraic(),
                                      CurrentMomentTerm(:hot_tail_distribution, g; pitch_rule=:cell_center),
                                      LinearTerm(:hot_electron_current_density; coefficient=-1.0)),
                             Equation(:runaway_density, TimeDerivative(),
                                      MomentumBoundaryFluxTerm(:hot_tail_distribution, hot_term))]
                system = EquationSystem(layout, equations;
                                        time_step_tolerances=Dict(:runaway_density => Tolerance(1e-10, reltol_step)))
                u0 = zeros(length(layout))
                f0 = f_initial
                n_re_dream = read(grp["n_re"])
                for ir in 1:nr
                    kinetic_view(u0, layout, 1, ir) .= f0[:, :, ir]
                    unknown_view(u0, layout, :hot_electron_density, ir) .= density(g, f0)[ir]
                    unknown_view(u0, layout, :hot_electron_current_density, ir) .= current_density(g, f0)[ir]
                    unknown_view(u0, layout, :runaway_density, ir) .= n_re_dream[ir, 1]
                end
                # the stretch where f_hot binds in both codes: DREAM's checks 1 to 23,
                # its accepted steps 1 to 22. The errors of the two codes agree to 1e-8
                # at the first check and to 4e-6 at the 22nd, since every step feeds
                # the next, and the accepted times drift to 1e-6 by then. The run ends
                # beyond the stretch, so that no ceiling cuts a step inside it.
                common = 23
                stepper = StepDoubling(initial_step=dt0, relative_tolerance=reltol_step, check_every=check_every)
                solution = integrate(system, u0, [0.0, 1.5 * t_dream[common]]; stepper, save_times=t_dream[1:common])
                stats = solution.statistics
                accepted = stats.time[stats.accepted]
                # the same accepted steps, the same rejections with the same errors
                @test length(accepted) >= common - 1
                @test accepted[1:common-1] ≈ t_dream[2:common] rtol = 1e-5
                checked = findall(k -> stats.kind[k] === :step && !isnan(stats.error[k]), eachindex(stats.kind))
                @test length(checked) >= common
                @test stats.error[checked[1:common]] ≈ logged[1:common] rtol = 1e-4
                @test stats.error[checked[1:3]] ≈ logged[1:3] rtol = 1e-6      # from the same state, DREAM logs 7 digits
                @test stats.accepted[checked[1:common]] == (logged[1:common] .<= 1)
                @test all(==(2), stats.newton_iterations)
                @test count(==(:half_step), stats.kind) == 2 * length(checked)
                # the moments DREAM checks at every accepted time (the first column holds
                # the initial guess of the algebraic unknowns, not their solution)
                @test solution.times == t_dream[1:common]
                n_hot = [only(unknown_view(solution.states[:, m], layout, :hot_electron_density, ir))
                         for ir in 1:nr, m in 2:common]
                j_hot = [only(unknown_view(solution.states[:, m], layout, :hot_electron_current_density, ir))
                         for ir in 1:nr, m in 2:common]
                n_re = [only(unknown_view(solution.states[:, m], layout, :runaway_density, ir))
                        for ir in 1:nr, m in 2:common]
                @test n_hot ≈ n_hot_dream[:, 2:common] rtol = 1e-8
                @test j_hot ≈ j_hot_dream[:, 2:common] rtol = 1e-6              # the drift of the times, 1e-6, on its slope
                @test n_re ≈ n_re_dream[:, 2:common] rtol = 1e-6
                # beyond the common stretch DREAM's own log binds on j_re with an error
                # Runes2P's n_re at its intended tolerance does not reach
                @test count(e -> e > 1, logged[common+1:end]) > 50
                # a differential run: with the checks of n_re and j_re
                # disabled, DREAM's step update still binds on unknown 7 (j_re) with the
                # same errors, and DREAM accepts steps whose reported error is above 1,
                # since its acceptance reads the tolerances by the unknown ID and its
                # step update by the position in the non-trivial list
                other = h5["no_runaway_check"]
                other_log = read(other["stepper_log"])
                @test read(attributes(other)["runaway_check"]) == 0
                binding = [parse(Int, m.captures[1]) for m in eachmatch(r"for unknown #(\d+)", other_log)]
                other_errors = _dream_logged_errors(other_log)
                @test other_errors[1:common] ≈ logged[1:common]
                @test count(==(7), binding) > 50
                rejected = length(other_errors) - (length(read(other["t"])) - 1)
                @test rejected == 3
                @test count(e -> e > 1, other_errors) > rejected
                @test read(other["j_re"])[:, 2:end] ≈ Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT *
                                                    read(other["n_re"])[:, 2:end] rtol = 1e-6
            end
        end
    end
end
