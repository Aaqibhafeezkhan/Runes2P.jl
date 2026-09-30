# Radial transport of the hot-tail and the runaway distribution compared with a DREAM run
# (reference/dream/cases/radial_transport.py): Rechester-Rosenbluth diffusion with a level
# δB/B = 1e-3 that is uniform in radius and constant in time, at three radii, with DREAM's
# condition f = 0 outside the plasma (`r_boundary = :absorbing_dream`) and without a flux
# through the edge. DREAM sets R₀ = 1 in a cylinder and gets the level times √R₀.
using HDF5

const DREAM_RADIAL_TRANSPORT_DATA = joinpath(@__DIR__, "data", "dream_radial_transport.h5")

@testset "radial transport vs DREAM" begin
    h5open(DREAM_RADIAL_TRANSPORT_DATA) do h5
        radial_grid = RadialGrid(read(h5["grid/r_f"]))
        nr = radial_cell_count(radial_grid)
        hot = cylindrical_grid(MomentumGrid(read(h5["grid/hottail/p1_f"]), read(h5["grid/hottail/p2_f"])),
                               radial_grid)
        runaway = runaway_grid(hot, read(h5["grid/runaway/p1_f"])[2:end], read(h5["grid/runaway/p2_f"]))
        names, Zs, Z0s, n = read(h5["ions/name"]), read(h5["ions/Z"]), read(h5["ions/Z0"]), read(h5["ions/n"])
        ions = [IonSpecies(names[k], Zs[k], Z0s[k], n[:, k]) for k in eachindex(Zs)]
        plasma = PlasmaState(cold_electron_temperature=read(h5["T_cold"]), cold_electron_density=read(h5["n_cold"]),
                             ions=ions, electric_field=read(h5["E"]), magnetic_field=fill(read(h5["B"]), nr))
        settings = CollisionSettings(ion_size_table=:dream, screening_window=:dream)
        level, major_radius = read(h5["delta_b_over_b"]), read(h5["major_radius"])

        function difference(a, b)
            cut = 1e-12 * maximum(abs, a)
            return maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
        end

        edges = ("f_0" => :absorbing_dream, "conservative" => :zero_flux)
        @testset "$variant" for (variant, r_boundary) in edges
            grp = h5[variant]
            times = read(grp["t"])
            saved = read(grp["saved_steps"]) .+ 1
            f_hot, f_re = read(grp["f_hot"]), read(grp["f_re"])   # (np, nξ, nr, saved states)
            @test all(==(2), read(grp["newton_iterations"]))
            hot_operator = fokker_planck_operator(hot, plasma, settings; synchrotron=true, scheme=:central,
                                                  p_upper_boundary=:interface)
            runaway_operator = fokker_planck_operator(runaway, plasma, settings; synchrotron=true, scheme=:central,
                                                      p_lower_boundary=:interface, p_upper_boundary=:constant_flux)
            radial_operators = map((hot, runaway)) do grid
                op = AdvectionDiffusion(grid; scheme=:central, r_boundary)
                add_rechester_rosenbluth_diffusion!(op, level; major_radius)
            end
            hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator)
            interface = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, hot_term,
                                              runaway_operator; pitch_mapping=:dream)
            layout = UnknownLayout(radial_grid, [KineticUnknown(:hot_tail_distribution, hot; scale=1e16),
                                                 KineticUnknown(:runaway_distribution, runaway; scale=1e16)])
            system = EquationSystem(layout, [
                Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface,
                         FokkerPlanckTerm(:hot_tail_distribution, radial_operators[1])),
                Equation(:runaway_distribution, TimeDerivative(),
                         FokkerPlanckTerm(:runaway_distribution, runaway_operator), interface,
                         FokkerPlanckTerm(:runaway_distribution, radial_operators[2]))])
            hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :runaway_distribution)
            u0 = zeros(length(layout))
            for ir in 1:nr
                kinetic_view(u0, layout, hot_id, ir) .= f_hot[:, :, ir, 1]
                kinetic_view(u0, layout, runaway_id, ir) .= f_re[:, :, ir, 1]
            end
            solution = integrate(system, u0, times; save_times=times[saved])
            @test all(solution.statistics.accepted)
            errors = map(eachindex(saved)) do m
                state = solution.states[:, m]
                (maximum(difference(f_hot[:, :, ir, m], kinetic_view(state, layout, hot_id, ir)) for ir in 1:nr),
                 maximum(difference(f_re[:, :, ir, m], kinetic_view(state, layout, runaway_id, ir)) for ir in 1:nr))
            end
            @test errors[1] == (0.0, 0.0)
            # observed largest differences, relative to DREAM's values: 7.7e-13 on the hot-tail
            # grid and 4.6e-15 on the runaway grid with f = 0 outside the plasma, 2.0e-13 and
            # 3.9e-15 without a flux through the edge
            @test maximum(first, errors) < 1e-11
            @test maximum(last, errors) < 1e-12

            # the radial transport changes the result: without it (a level of zero, in a system
            # of the same terms) the states differ from DREAM's
            idle = map(grid -> AdvectionDiffusion(grid; scheme=:central, r_boundary), (hot, runaway))
            without = EquationSystem(layout, [
                Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface,
                         FokkerPlanckTerm(:hot_tail_distribution, idle[1])),
                Equation(:runaway_distribution, TimeDerivative(),
                         FokkerPlanckTerm(:runaway_distribution, runaway_operator), interface,
                         FokkerPlanckTerm(:runaway_distribution, idle[2]))])
            other = integrate(without, u0, times; save_times=times[saved])
            @test maximum(difference(f_hot[:, :, ir, end], kinetic_view(other.states[:, end], layout, hot_id, ir))
                          for ir in 1:nr) > 1e-3
            # Runes2P's distance at the edge gives another result than DREAM's
            if r_boundary === :absorbing_dream
                radial_operators_runes2p = map((hot, runaway)) do grid
                    op = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
                    add_rechester_rosenbluth_diffusion!(op, level; major_radius)
                end
                runes2p = EquationSystem(layout, [
                    Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface,
                             FokkerPlanckTerm(:hot_tail_distribution, radial_operators_runes2p[1])),
                    Equation(:runaway_distribution, TimeDerivative(),
                             FokkerPlanckTerm(:runaway_distribution, runaway_operator), interface,
                             FokkerPlanckTerm(:runaway_distribution, radial_operators_runes2p[2]))])
                closer_edge = integrate(runes2p, u0, times; save_times=times[saved])
                @test maximum(difference(f_hot[:, :, ir, end],
                                         kinetic_view(closer_edge.states[:, end], layout, hot_id, ir))
                              for ir in 1:nr) > 1e-3
            end
        end
    end
end
