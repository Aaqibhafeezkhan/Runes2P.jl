# The self-consistent electric field with both kinetic grids compared with a DREAM run
# (reference/dream/cases/self_consistent_field.py, after DREAM's
# examples/transport-rr/generate_consistent.py): the field as an unknown of both kinetic
# equations, Faraday's and Ampère's law with the wall, the currents and densities as moments
# with the threshold of the hot region, and the particle source with its magnitude as an
# unknown. DREAM's mode throughout: the cell-centre pitch rule of the moments, the flux
# between the grids with DREAM's pitch mapping, centred interpolation. The variant
# "implicit_source" has the particle source and the wall at 1.1 minor radii, and
# "corrected_conductivity" the same with DREAM's corrected conductivity: one type of
# system, the term of the correction has the coefficient zero in the first. The variant
# "no_source" has no particle source, the wall at the edge of the plasma and no loop
# voltage.
using HDF5

const DREAM_SELF_CONSISTENT_DATA = reference_file("dream_self_consistent_field.h5")

@testset "self-consistent field vs DREAM" begin
    h5open(DREAM_SELF_CONSISTENT_DATA) do h5
        radial = RadialGrid(read(h5["grid/r_f"]))
        hot = cylindrical_grid(
            MomentumGrid(read(h5["grid/hottail/p1_f"]), read(h5["grid/hottail/p2_f"])),
            radial)
        runaway = runaway_grid(hot, read(h5["grid/runaway/p1_f"])[2:end], read(h5["grid/runaway/p2_f"]))
        nr = radial_cell_count(radial)
        @test last(radial.r_faces) ≈ read(h5["minor_radius"]) rtol = 1e-14
        names = read(h5["ions/name"])
        Zs = read(h5["ions/Z"])
        Z0s = read(h5["ions/Z0"])
        n = read(h5["ions/n"])
        ions = [IonSpecies(names[k], Zs[k], Z0s[k], n[:, k]) for k in eachindex(Zs)]
        settings = CollisionSettings(
            ion_size_table = :dream, screening_window = :dream, bremsstrahlung = :neglect)
        fluid = (electric_field = "E_field", poloidal_flux = "psi_p",
            total_current_density = "j_tot",
            ohmic_current_density = "j_ohm", hot_electron_current_density = "j_hot",
            runaway_current_density = "j_re", hot_electron_density = "n_hot", runaway_density = "n_re",
            cold_electron_density = "n_cold")
        scalars = (plasma_current = "I_p", edge_poloidal_flux = "psi_edge",
            wall_poloidal_flux = "psi_wall")

        # largest relative difference to DREAM over the cells above 1e-12 of the largest value
        function difference(a, b)
            cut = 1e-12 * maximum(abs, a)
            return maximum(abs(a[k] - b[k]) / abs(a[k])
            for k in eachindex(a) if abs(a[k]) > cut)
        end

        @testset "$variant" for variant in ("implicit_source", "corrected_conductivity", "no_source")
            grp = h5[variant]
            corrected = variant == "corrected_conductivity"
            with_source = variant != "no_source"
            # DREAM's PARTICLE_SOURCE_ZERO = 1, PARTICLE_SOURCE_IMPLICIT = 2
            @test read(attributes(grp)["particle_source"]) == (with_source ? 2 : 1)
            @test read(attributes(grp)["corrected_conductivity"]) == Int(corrected)
            times = read(grp["t"])
            saved = read(grp["saved_steps"]) .+ 1
            f_hot = read(grp["f_hot"])   # (np, nξ, nr, saved steps)
            f_re = read(grp["f_re"])
            data = Dict{Symbol, Array{Float64}}(name => read(grp[dream])
            for (name, dream) in pairs(fluid))   # (nr, nt + 1)
            for (name, dream) in pairs(scalars)
                data[name] = vec(read(grp[dream]))   # (nt + 1)
            end
            temperature = read(grp["T_cold"])[:, 1]
            cold_density = data[:cold_electron_density][:, 1]
            wall_radius = read(grp["wall_radius"])
            loop_voltage = read(grp["loop_voltage_at_wall"])
            @test all(<=(5), read(grp["newton_iterations"]))

            # DREAM starts from the flux of its discrete Ampère's law for the current of the
            # initial distributions, with the flux at the wall zero
            initial = initial_poloidal_flux(radial, data[:total_current_density][:, 1]; wall_radius)
            @test initial.poloidal_flux ≈ data[:poloidal_flux][:, 1] rtol = 1e-12
            @test initial.plasma_current ≈ data[:plasma_current][1] rtol = 1e-12
            @test initial.edge_poloidal_flux≈data[:edge_poloidal_flux][1] rtol=1e-12 atol=1e-18
            @test data[:wall_poloidal_flux][1] == 0
            # and from the moments of the initial distributions
            region = MomentumRegion(ThermalMomentum(temperature))
            moment(f, grid; kwargs...) = current_density(grid, f; pitch_rule = :cell_center, kwargs...)
            @test moment(f_hot[:, :, :, 1], hot; region) ≈
                  data[:hot_electron_current_density][:, 1] rtol = 1e-11
            @test moment(f_re[:, :, :, 1], runaway) ≈ data[:runaway_current_density][:, 1] rtol = 1e-11
            @test density(hot, f_hot[:, :, :, 1]; region) ≈
                  data[:hot_electron_density][:, 1] rtol = 1e-11
            @test density(runaway, f_re[:, :, :, 1]) ≈ data[:runaway_density][:, 1] rtol = 1e-11

            plasma = PlasmaState(cold_electron_temperature = temperature,
                cold_electron_density = cold_density,
                ions = ions, electric_field = zeros(nr), magnetic_field = fill(read(h5["B"]), nr))
            # DREAM's conductivity of Sauter and Redl is that of Braams and Karney in a cylinder
            conductivity = read(grp["conductivity"])   # (nr, nt)
            # (Z_eff = 1 and n_cold = n_free in this case: the check covers the table, the
            # interpolation, the Coulomb logarithm and the constants, test_conductivity.jl the
            # effective charge)
            expected_conductivity = [braams_karney_conductivity(plasma, ir) for ir in 1:nr]
            @test all(isapprox(conductivity[:, m], expected_conductivity; rtol = 1e-12)
            for m in axes(conductivity, 2))
            correction = ConductivityTerm(
                :electric_field, ConductivityModel(plasma; kinetic = :dream_fit);
                coefficient = corrected ? 1 : 0)
            # the ohmic current of the start holds the correction with the field of the start
            @test moment(f_hot[:, :, :, 1], hot) .- moment(f_hot[:, :, :, 1], hot; region) .+
                  (corrected ?
                   conductivity_at(correction, 0.0) .* data[:electric_field][:, 1] :
                   zeros(nr)) ≈
                  data[:ohmic_current_density][:, 1] rtol = 1e-11
            hot_operator = fokker_planck_operator(
                hot, plasma, settings; synchrotron = true, scheme = :central,
                p_upper_boundary = :interface, field_in_coefficients = false)
            # DREAM's default at the p_max of the runaway grid: BC_PHI_CONST
            runaway_operator = fokker_planck_operator(
                runaway, plasma, settings; synchrotron = true, scheme = :central,
                p_lower_boundary = :interface, p_upper_boundary = :constant_flux,
                field_in_coefficients = false)
            hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_operator; electric_field = :electric_field)
            runaway_term = FokkerPlanckTerm(
                :runaway_distribution, runaway_operator; electric_field = :electric_field)
            interface = MomentumInterfaceTerm(
                :hot_tail_distribution, :runaway_distribution, hot_term,
                runaway_operator; pitch_mapping = :dream)
            minus(name) = LinearTerm(name; coefficient = -1)
            current(name, grid; kwargs...) = CurrentMomentTerm(name, grid; pitch_rule = :cell_center, kwargs...)
            unknowns = vcat(
                induction_unknowns(radial; electric_field_scale = 1.0, current_density_scale = 1e6),
                [
                    RadialUnknown(:ohmic_current_density; scale = 1e6),
                    RadialUnknown(:hot_electron_current_density; scale = 1e2),
                    RadialUnknown(:runaway_current_density; scale = 1e5),
                    RadialUnknown(:hot_electron_density; scale = 1e11),
                    RadialUnknown(:runaway_density; scale = 1e15),
                    RadialUnknown(:runaway_density_above_grid; scale = 1e10),
                    RadialUnknown(:cold_electron_density; scale = 1e19)],
                with_source ? [RadialUnknown(:particle_source; scale = 1e20)] :
                RadialUnknown[], [
                    KineticUnknown(:hot_tail_distribution, hot; scale = 1e24),
                    KineticUnknown(:runaway_distribution, runaway; scale = 1e13)])
            layout = UnknownLayout(radial, unknowns)
            equations = vcat(
                induction_equations(radial; wall_radius, loop_voltage_at_wall = loop_voltage),
                [
                    Equation(:total_current_density, Algebraic(),
                        LinearTerm(:ohmic_current_density),
                        LinearTerm(:hot_electron_current_density), LinearTerm(:runaway_current_density),
                        minus(:total_current_density)),
                    # DREAM's ohmic current of the fully kinetic mode without the corrected
                    # conductivity: the current of all momenta minus the hot current
                    Equation(:ohmic_current_density, Algebraic(),
                        current(:hot_tail_distribution, hot),
                        minus(:hot_electron_current_density), correction, minus(:ohmic_current_density)),
                    Equation(:hot_electron_current_density, Algebraic(),
                        current(:hot_tail_distribution, hot; region),
                        minus(:hot_electron_current_density)),
                    Equation(:runaway_current_density, Algebraic(),
                        current(:runaway_distribution, runaway),
                        minus(:runaway_current_density)),
                    Equation(:hot_electron_density, Algebraic(),
                        DensityMomentTerm(:hot_tail_distribution, hot; region),
                        minus(:hot_electron_density)),
                    Equation(:runaway_density, Algebraic(),
                        DensityMomentTerm(:runaway_distribution),
                        LinearTerm(:runaway_density_above_grid), minus(:runaway_density)),
                    Equation(:runaway_density_above_grid, TimeDerivative(),
                        MomentumBoundaryFluxTerm(:runaway_distribution, runaway_term)),
                    Equation(:cold_electron_density, Algebraic(),
                        PrescribedValueTerm(cold_density),
                        minus(:cold_electron_density)),
                    Equation(:runaway_distribution, TimeDerivative(), runaway_term, interface)],
                with_source ?
                [
                    Equation(:particle_source, Constraint(),
                        DensityMomentTerm(:hot_tail_distribution),
                        minus(:cold_electron_density), minus(:hot_electron_density); scale = 1e19),
                    Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface,
                        ParticleSourceTerm(:particle_source, hot, temperature))] :
                [Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface)])
            system = EquationSystem(layout, equations)
            hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution),
            unknown_id(layout, :runaway_distribution)
            u0 = zeros(length(layout))
            for ir in 1:nr
                kinetic_view(u0, layout, hot_id, ir) .= f_hot[:, :, ir, 1]
                kinetic_view(u0, layout, runaway_id, ir) .= f_re[:, :, ir, 1]
                for name in keys(fluid)
                    radial_view(u0, layout, unknown_id(layout, name), ir) .= data[name][ir, 1]
                end
            end
            for name in keys(scalars)
                u0[only(unknown_range(layout, unknown_id(layout, name)))] = data[name][1]
            end
            solution = integrate(system, u0, times; solver = NewtonSolver(residual_tolerance = 1e-12))
            @test all(solution.statistics.accepted)

            # the distributions at the saved steps
            hot_errors = [maximum(difference(f_hot[:, :, ir, k],
                                      kinetic_view(solution.states[:, m], layout, hot_id, ir))
                          for ir in 1:nr) for (k, m) in enumerate(saved)]
            runaway_errors = [maximum(difference(f_re[:, :, ir, k],
                                          kinetic_view(solution.states[:, m], layout, runaway_id, ir))
                              for ir in 1:nr) for (k, m) in enumerate(saved)]
            @test hot_errors[1] == 0 && runaway_errors[1] == 0
            @test maximum(hot_errors) < 1e-10
            @test maximum(runaway_errors) < 1e-9
            # the fluid unknowns at all time levels after the start
            value(name, ir, m) = only(unknown_view(view(solution.states, :, m), layout, name, ir))
            errors = Dict{Symbol, Float64}()
            for name in keys(fluid)
                largest = maximum(abs, data[name])
                errors[name] = maximum(abs(value(name, ir, m) - data[name][ir, m]) /
                                       largest
                for ir in 1:nr, m in 2:length(times))
            end
            for name in keys(scalars)
                largest = maximum(abs, data[name])
                position = only(unknown_range(layout, unknown_id(layout, name)))
                errors[name] = largest == 0 ? maximum(abs, solution.states[position, :]) :
                               maximum(abs(solution.states[position, m] - data[name][m]) /
                                       largest
                for m in 2:length(times))
            end
            if with_source
                source = read(grp["S_particle"])
                errors[:particle_source] = maximum(abs(value(:particle_source, ir, m) -
                                                       source[ir, m])
                for ir in 1:nr, m in 2:length(times)) / maximum(abs, source)
            end
            # observed largest differences, relative to the largest value of DREAM: 2e-13 in the
            # field, 1e-14 in the flux, 1e-13 in the currents and the hot density, 2e-15 in the
            # runaway density, and 9e-8 in the magnitude of the source, which follows from the
            # change of the density over a step. 1.5e-12 in f_hot and 2.0e-11 in f_re (above).
            # 99 Newton iterations in the 30 steps (DREAM: 3 per step and 5 in the first),
            # 21 colors for 3756 unknowns
            for (name, error) in errors
                @test error < (name === :particle_source ? 1e-6 : 1e-10)
            end
            @test sum(solution.statistics.newton_iterations) <= 4 * (length(times) - 1)
            @test jacobian_colors(system) <= 21
            # observed with the corrected conductivity: 1.1e-13 of the field, 1.6e-12 of f_hot and
            # f_re, 1.9e-15 of the plasma current, 5.8e-8 of the source. The correction raises
            # the plasma current at the end from 138 kA to 198 kA and lowers the field at the
            # axis from 1.10 to 0.89 V/m
            # the field changes during the run, and nothing leaves the runaway grid
            field = data[:electric_field]
            @test maximum(abs.(field[:, end] .- field[:, 2]) ./ abs.(field[:, 2])) > 0.02
            # DREAM's runaway density is not lowered by the flux through the upper boundary of
            # the runaway grid: it is the density of f_re with the electrons above the grid.
            # Observed: 7e-6 to 9e-6 of the runaway density has left the grid at the end
            for ir in 1:nr
                above = value(:runaway_density_above_grid, ir, length(times))
                left = data[:runaway_density][ir, end] -
                       density(runaway, f_re[:, :, :, end])[ir]
                @test above > 1e-6 * data[:runaway_density][ir, end]
                @test above ≈ left rtol = 1e-6
            end

            # The field of the fully kinetic system is fixed through the kinetic equations
            # (index 2), and the magnitude of the source through the density: short steps
            # divide changes at round-off by the step. One step from the start at three
            # lengths, with the direct solver and with GMRES. The field of a short step is
            # that which holds the current against the friction on the drifting Maxwellian of
            # the start (1.85, 2.02 and 1.81 V/m). It falls to 1.10, 1.21 and 1.11 V/m in the
            # first step of the run, as the distribution takes the shape of the ohmic one.
            if variant == "implicit_source"
                short = map((1e-5, 1e-9, 1e-11)) do dt
                    states = map((NewtonSolver(residual_tolerance = 1e-12),
                        NewtonSolver(linear_solver = GMRES(), residual_tolerance = 1e-12))) do solver
                        step = integrate(system, u0, [0.0, dt]; solver)
                        @test all(step.statistics.accepted)
                        @test sum(step.statistics.newton_iterations) <= 8
                        step.states[:, end]
                    end
                    fields = [collect(unknown_values(state, layout, unknown_id(layout, :electric_field)))
                              for state in states]
                    @test fields[1] ≈ fields[2] rtol = 1e-8
                    fields[1]
                end
                @test short[1] ≈ data[:electric_field][:, 2] rtol = 0.1
                @test short[2] ≈ short[3] rtol = 0.03
                @test all(short[3] .> 1.5 .* data[:electric_field][:, 2])
            end
        end
    end
end
