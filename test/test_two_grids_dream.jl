# The hot-tail and the runaway grid with the flux between them compared with a DREAM run
# (reference/dream/cases/two_grids.py, after DREAM's examples/2kinetic/generate.py): the
# Fokker-Planck operator on both grids, DREAM's flux through the interface between
# different pitch grids (`pitch_mapping = :dream`), and in the second variant the knock-on
# pairings hot-tail from hot-tail, hot-tail from runaway and runaway from runaway, explicit
# as in DREAM.
using HDF5

const DREAM_TWO_GRIDS_DATA = joinpath(@__DIR__, "data", "dream_two_grids.h5")

@testset "two kinetic grids vs DREAM" begin
    h5open(DREAM_TWO_GRIDS_DATA) do h5
        radial = RadialGrid(read(h5["grid/r_f"]))
        hot = cylindrical_grid(MomentumGrid(read(h5["grid/hottail/p1_f"]), read(h5["grid/hottail/p2_f"])), radial)
        runaway_faces = read(h5["grid/runaway/p1_f"])
        runaway = runaway_grid(hot, runaway_faces[2:end], read(h5["grid/runaway/p2_f"]))
        # DREAM starts the runaway grid at the last face of the hot-tail grid
        @test runaway_faces[1] == last(momentum_grid(hot).p_faces)
        @test momentum_grid(runaway).p_faces == runaway_faces
        nr = radial_cell_count(radial)
        # DREAM's cylindrical Vprime is 4π² times the phase-space Jacobian of Runes2P, on both grids
        for (name, grid) in (("hottail", hot), ("runaway", runaway)), ir in 1:nr
            @test read(h5["grid/$name/Vprime"])[:, :, ir] ≈ 4pi^2 .* phase_space_jacobian(grid, ir) rtol = 1e-13
        end
        names = read(h5["ions/name"])
        Zs = read(h5["ions/Z"])
        Z0s = read(h5["ions/Z0"])
        n = read(h5["ions/n"])
        ions = [IonSpecies(names[k], Zs[k], Z0s[k], n[:, k]) for k in eachindex(Zs)]
        p_cut = read(h5["p_cut"])
        settings = CollisionSettings(ion_size_table=:dream)

        # largest relative difference to DREAM over the cells above 1e-12 of the largest value
        function difference(a, b)
            cut = 1e-12 * maximum(abs, a)
            return maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
        end

        @testset "$variant" for variant in ("neglect", "boltzmann")
            grp = h5[variant]
            times = read(grp["t"])
            f_hot = read(grp["f_hot"])   # (np, nξ, nr, nt + 1)
            f_re = read(grp["f_re"])
            n_tot = read(grp["n_tot"])
            @test n_tot == repeat(n_tot[:, 1], 1, size(n_tot, 2))
            # a linear problem with explicit knock-on: the solve, then the check
            @test all(==(2), read(grp["newton_iterations"]))
            plasma = PlasmaState(cold_electron_temperature=read(grp["T_cold"])[:, 1],
                                 cold_electron_density=read(grp["n_cold"])[:, 1], ions=ions,
                                 electric_field=read(h5["E"]), magnetic_field=fill(read(h5["B"]), nr))
            hot_operator = fokker_planck_operator(hot, plasma, settings; synchrotron=true, scheme=:central,
                                                  p_upper_boundary=:interface)
            # DREAM's default at the p_max of the runaway grid: BC_PHI_CONST
            runaway_operator = fokker_planck_operator(runaway, plasma, settings; synchrotron=true, scheme=:central,
                                                      p_lower_boundary=:interface, p_upper_boundary=:constant_flux)
            hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator)
            interface = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, hot_term,
                                              runaway_operator; pitch_mapping=:dream)
            hot_terms = EquationTerm[hot_term, interface]
            runaway_terms = EquationTerm[FokkerPlanckTerm(:runaway_distribution, runaway_operator), interface]
            # without avalanche the knock-on terms stay in the system with a target density
            # of zero, so that both variants are one type of system and compile once
            let
                kernel = (dream_compatible=true, primary_pitch_points=1, table_size=80)
                target = variant == "boltzmann" ? n_tot[:, 1] : zeros(nr)
                push!(hot_terms,
                      KnockOnTerm(:hot_tail_distribution, KnockOnOperator(hot, hot, p_cut; kernel...);
                                  target_density=target, treatment=:explicit),
                      KnockOnTerm(:runaway_distribution, KnockOnOperator(hot, runaway, p_cut; kernel...);
                                  target_density=target, treatment=:explicit))
                push!(runaway_terms,
                      KnockOnTerm(:runaway_distribution, KnockOnOperator(runaway, runaway, p_cut; kernel...);
                                  target_density=target, treatment=:explicit))
            end
            layout = UnknownLayout(radial, [KineticUnknown(:hot_tail_distribution, hot; scale=1e16),
                                            KineticUnknown(:runaway_distribution, runaway; scale=1e16)])
            system = EquationSystem(layout, [
                Equation(:hot_tail_distribution, TimeDerivative(), hot_terms...),
                Equation(:runaway_distribution, TimeDerivative(), runaway_terms...)])
            hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :runaway_distribution)
            u0 = zeros(length(layout))
            for ir in 1:nr
                kinetic_view(u0, layout, hot_id, ir) .= f_hot[:, :, ir, 1]
                kinetic_view(u0, layout, runaway_id, ir) .= f_re[:, :, ir, 1]
            end
            solution = integrate(system, u0, times; solver=NewtonSolver())
            @test all(solution.statistics.accepted)
            hot_errors = [maximum(difference(f_hot[:, :, ir, m], kinetic_view(solution.states[:, m], layout, hot_id, ir))
                                  for ir in 1:nr) for m in eachindex(times)]
            runaway_errors = [maximum(difference(f_re[:, :, ir, m],
                                                 kinetic_view(solution.states[:, m], layout, runaway_id, ir))
                                      for ir in 1:nr) for m in eachindex(times)]
            @test hot_errors[1] == 0 && runaway_errors[1] == 0
            # observed largest differences over the 20 steps, relative to DREAM's values:
            # 2.0e-13 on the hot-tail grid and 7e-15 on the runaway grid, in both variants
            @test maximum(hot_errors) < 1e-11
            @test maximum(runaway_errors) < 1e-12

            # the flux through the interface is DREAM's gammaFhot
            flux_dream = read(grp["gammaFhot"])   # (nr, nt)
            flux_errors = [abs(interface_flux(interface, solution.states[:, m+1], layout, ir) - flux_dream[ir, m]) /
                           abs(flux_dream[ir, m]) for ir in 1:nr, m in 1:length(times)-1]
            @test maximum(flux_errors) < 1e-10

            # DREAM's runaway density with a runaway grid: it starts
            # as the density of f_re, grows by the flux through the interface and by the
            # knock-on production on the runaway grid, and is not lowered by the flux
            # through the p_max of the runaway grid. At the end it lies above the density of
            # f_re by what has left: 2.3e-6 and 1.5e-5 of its value at the two radii.
            n_re = read(grp["n_re"])   # (nr, nt + 1)
            steps = diff(times)
            runaway_array(m) = stack(kinetic_view(solution.states[:, m], layout, runaway_id, ir) for ir in 1:nr)
            runaway_density = [density(runaway, runaway_array(m)) for m in eachindex(times)]
            @test n_re[:, 1] ≈ runaway_density[1] rtol = 1e-12
            produced = zeros(nr)
            if variant == "boltzmann"
                operator = runaway_terms[3].operator
                for m in 1:length(steps), ir in 1:nr
                    source = zeros(size(runaway)[1:2])
                    apply_at_radius!(source, operator, kinetic_view(solution.states[:, m], layout, runaway_id, ir),
                                     n_tot[ir, 1], ir)
                    produced[ir] += steps[m] * density(runaway, stack(source for _ in 1:nr))[ir]
                end
            end
            for ir in 1:nr
                through_interface = sum(steps[m] * flux_dream[ir, m] for m in eachindex(steps))
                @test n_re[ir, end] - n_re[ir, 1] ≈ through_interface + produced[ir] rtol = 1e-9
                left = sum(steps[m] * momentum_flux(runaway_operator, runaway_array(m + 1),
                                                    momentum_cell_count(runaway) + 1, ir) for m in eachindex(steps))
                @test left > 0
                @test n_re[ir, end] - runaway_density[end][ir] ≈ left rtol = 1e-5
            end

            # the other mapping and the scheme's own flux change the result at the interface
            other = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, hot_term,
                                          runaway_operator; pitch_mapping=:overlap)
            other_system = EquationSystem(layout, [
                Equation(:hot_tail_distribution, TimeDerivative(), hot_term, other, hot_terms[3:end]...),
                Equation(:runaway_distribution, TimeDerivative(), runaway_terms[1], other, runaway_terms[3:end]...)])
            other_solution = integrate(other_system, u0, times; solver=NewtonSolver())
            last_hot = maximum(difference(f_hot[:, :, ir, end],
                                          kinetic_view(other_solution.states[:, end], layout, hot_id, ir))
                               for ir in 1:nr)
            @test last_hot > 1e-8
        end
    end
end
