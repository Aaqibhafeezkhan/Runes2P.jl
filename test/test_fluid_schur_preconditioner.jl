# The preconditioner of the coupled system: a small system whose radii are
# coupled through Ampère's law and the scalars of the induction equations and through the
# radial diffusion of the distribution, with a fluid Ohm's law and a kinetic current. The
# preconditioner is a linear map, holds the fluid coupling exactly, and GMRES with it
# reaches the solution of the direct solver in fewer iterations than with RadiusBlockLU.
using LinearAlgebra: I as IdentityMatrix, Diagonal, ldiv!
using SparseArrays: sparse, findnz
using Random: Xoshiro
# GMRES of Krylov.jl, a dependency of Runes2P, also where the test project does not list it
# an assertion on allocations holds for optimised code alone: it is skipped, and reported
# as broken, in a development run at -O0 or -O1
const OPTIMIZED = Base.JLOptions().opt_level >= 2

const SCHUR_TEST_GMRES = Base.require(Runes2P, :Krylov).gmres

@testset "FluidSchurLU" begin
    rng = Xoshiro(551)
    T, ne, radii = 20.0, 5e19, 5
    theta = T / Constants.ELECTRON_REST_ENERGY
    radial = RadialGrid(uniform_r_faces(radii, 0.22))
    grid = cylindrical_grid(
        MomentumGrid(uniform_p_faces(24, 0.0, 7 * sqrt(2theta)), uniform_xi_faces(4)),
        radial)
    plasma = PlasmaState(; cold_electron_temperature = fill(T, radii),
        ions = [IonSpecies("D", 1, 1, fill(ne, radii))],
        electric_field = zeros(radii), magnetic_field = fill(5.0, radii))
    settings = CollisionSettings(mode = :full, screening = :complete, coulomb_logarithm = :thermal)
    operator = fokker_planck_operator(grid, plasma, settings; electric_field = false)
    minus(name) = LinearTerm(name; coefficient = -1)
    collision_time = thermal_collision_time(ne, T, thermal_coulomb_logarithm(T, ne))

    function coupled_system(delta_b_over_b; field_particle = false)
        kinetic_terms = Any[
            FokkerPlanckTerm(:hot_tail_distribution, operator; electric_field = :electric_field),
            ParticleSourceTerm(:particle_source, grid, fill(T, radii))]
        # the momentum-conserving field-particle term takes the place of the conductivity correction
        field_particle &&
            push!(kinetic_terms,
                FieldParticleTerm(:hot_tail_distribution, FieldParticleOperator(grid, plasma, settings)))
        current_terms = field_particle ? () :
                        (ConductivityTerm(:electric_field,
            ConductivityModel(plasma; kinetic = :discrete_operator, operator)),)
        if delta_b_over_b > 0
            radial_operator = AdvectionDiffusion(grid; scheme = :central, r_boundary = :absorbing)
            add_rechester_rosenbluth_diffusion!(radial_operator, delta_b_over_b; major_radius = 1.0)
            push!(kinetic_terms, FokkerPlanckTerm(:hot_tail_distribution, radial_operator))
        end
        unknowns = vcat(induction_unknowns(radial; current_density_scale = 1e4),
            [RadialUnknown(:particle_source; scale = ne / collision_time),
                KineticUnknown(:hot_tail_distribution, grid; scale = 1e20)])
        equations = vcat(
            induction_equations(radial; wall_radius = 0.3, current_density_scale = 1e4),
            [
                Equation(:total_current_density, Algebraic(),
                    CurrentMomentTerm(:hot_tail_distribution, grid),
                    current_terms..., minus(:total_current_density)),
                Equation(:particle_source, Constraint(),
                    DensityMomentTerm(:hot_tail_distribution),
                    PrescribedValueTerm(-ne); scale = ne),
                Equation(:hot_tail_distribution, TimeDerivative(), kinetic_terms...)])
        layout = UnknownLayout(radial, unknowns)
        system = EquationSystem(layout, equations)
        u0 = zeros(length(layout))
        shape = source_shape(system.equations[end].terms[2], 1)
        profile = [1 - (r / 0.22)^2 for r in radial.r_centers]
        field = 1e-2 .* profile
        sigma = braams_karney_conductivity(plasma, 1)
        initial = initial_poloidal_flux(radial, sigma .* field; wall_radius = 0.3)
        unknown_values(u0, layout, unknown_id(layout, :electric_field)) .= field
        unknown_values(u0, layout, unknown_id(layout, :total_current_density)) .= sigma .*
                                                                                  field
        unknown_values(u0, layout, unknown_id(layout, :poloidal_flux)) .= initial.poloidal_flux
        for name in (:plasma_current, :edge_poloidal_flux, :wall_poloidal_flux)
            u0[only(unknown_range(layout, unknown_id(layout, name)))] = getproperty(initial, name)
        end
        for ir in 1:radii
            kinetic_view(u0, layout, unknown_id(layout, :hot_tail_distribution), ir) .= ne .*
                                                                                        shape
        end
        return system, layout, u0
    end

    # the scaled Newton matrix of a step of `dt` at u0, as NewtonSolver builds it
    function newton_matrix(system, u, dt)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, dt)
        weights = [(system.mass[i] == 0 ? 1.0 : dt) / system.row_scales[i]
                   for i in eachindex(system.mass)]
        return sparse(Diagonal(weights) * (sparse(mass_matrix(system)) ./ dt - J) *
                      Diagonal(unknown_scales(system.layout)))
    end

    for (delta_b_over_b, label) in ((0.0, "without radial transport"), (
        3e-4, "with radial transport"))
        @testset "$label" begin
            system, layout, u0 = coupled_system(delta_b_over_b)
            dt = 10 * collision_time
            A = newton_matrix(system, u0, dt)
            factors = Runes2P.preconditioner_factors(FluidSchurLU(), A, layout)
            @test Runes2P.factorize!(factors, FluidSchurLU(), A)
            n = size(A, 1)
            x, y = randn(rng, n), randn(rng, n)
            px, py, pxy = zeros(n), zeros(n), zeros(n)
            ldiv!(px, factors, x)
            ldiv!(py, factors, y)
            ldiv!(pxy, factors, 2 .* x .- 3 .* y)
            @test pxy ≈ 2 .* px .- 3 .* py rtol = 1e-10
            @test (@allocated ldiv!(px, factors, x)) == 0 skip = !OPTIMIZED
            # the Schur stage alone is the exact inverse of A without the kinetic couplings
            # between radii
            without_lines = Runes2P.preconditioner_factors(
                FluidSchurLU(;
                    radial_lines = false), A, layout)
            Runes2P.factorize!(without_lines, FluidSchurLU(; radial_lines = false), A)
            kinetic = falses(n)
            for ir in 1:radii
                kinetic[unknown_range(layout, unknown_id(layout, :hot_tail_distribution), ir)] .= true
            end
            radius = zeros(Int, n)
            for ir in 1:radii
                radius[radius_block_range(layout, ir)] .= ir
            end
            rows, columns, values = findnz(A)
            keep = [!(kinetic[r] && kinetic[c] && radius[r] != radius[c])
                    for (r, c) in zip(rows, columns)]
            reduced = sparse(rows[keep], columns[keep], values[keep], n, n)
            @test ldiv!(zeros(n), without_lines, x) ≈ reduced \ x rtol = 1e-9
            if delta_b_over_b == 0
                @test reduced == A
            end
            # GMRES with each preconditioner on the first Newton system
            b = randn(rng, n)
            exact = A \ b
            counts = Dict{Symbol, Int}()
            for (name, preconditioner) in (:block => RadiusBlockLU(), :schur =>
                FluidSchurLU(),
                :schur_without_lines => FluidSchurLU(; radial_lines = false))
                cache = Runes2P.preconditioner_factors(preconditioner, A, layout)
                @test Runes2P.factorize!(cache, preconditioner, A)
                solution, stats = SCHUR_TEST_GMRES(
                    A, b; N = cache, ldiv = true, restart = true, memory = 200, itmax = 400,
                    rtol = 1e-12, atol = 0.0)
                @test stats.solved
                @test solution ≈ exact rtol = 1e-8
                counts[name] = stats.niter
            end
            @test counts[:schur] < counts[:block]
            if delta_b_over_b == 0
                @test counts[:schur_without_lines] <= 2
            else
                @test counts[:schur] <= counts[:schur_without_lines]
            end
            # a run: GMRES with FluidSchurLU against the direct solver
            times = collect(range(0, 20 * dt; length = 6))
            direct = integrate(system, u0, times)
            for preconditioner in (FluidSchurLU(), FluidSchurLU(; radial_lines = false))
                krylov = integrate(system, u0, times;
                    solver = NewtonSolver(linear_solver = GMRES(; preconditioner)))
                @test all(krylov.statistics.accepted)
                @test krylov.states[:, end] ≈ direct.states[:, end] rtol = 1e-7
            end
        end
    end

    # the plasma as unknowns, the columns of the temperature, the cold density
    # and the charge states in the kinetic rows (the implicit Fokker-Planck term) and the block
    # of the kinetic ionization in the rows of the charge states. Without kinetic couplings
    # between radii the preconditioner is the exact inverse, and GMRES needs one iteration.
    @testset "with the plasma unknowns and the kinetic ionization" begin
        composition = IonComposition(["D" => 1, "Ar" => 18])
        nr = 3
        temperatures = [20.0, 15.0, 10.0]
        ion_radial = RadialGrid(uniform_r_faces(nr, 0.5))
        ion_grid = cylindrical_grid(
            MomentumGrid(
                uniform_p_faces(30, 0.0, 10 *
                                         sqrt(2 * 20.0 / Constants.ELECTRON_REST_ENERGY)),
                uniform_xi_faces(3)),
            ion_radial)
        background = [
            IonSpecies("D", 1, 1, fill(1e19, nr)), IonSpecies("Ar", 18, zeros(19, nr))]
        prescribed = PlasmaState(; cold_electron_temperature = temperatures,
            ions = background, electric_field = zeros(nr),
            magnetic_field = fill(2.5, nr))
        layout = UnknownLayout(ion_radial,
            [ion_densities_unknown(composition),
                RadialUnknown(:cold_electron_density; scale = 1e19),
                RadialUnknown(:cold_electron_temperature; scale = 10.0),
                KineticUnknown(:f, ion_grid; scale = 1e19)])
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        kinetic = KineticIonizationTerm(composition, :f, ion_grid; region)
        fp = FokkerPlanckTerm(:f,
            AdvectionDiffusion(ion_grid; scheme = :exponential_fitting, p_upper_boundary = :f_zero);
            plasma = PlasmaFromUnknowns(prescribed; ions = :ion_densities),
            settings = CollisionSettings(mode = :full, screening = :partial), lagged = ())
        equations = [Equation(:ion_densities, TimeDerivative(), kinetic),
            population_equations(composition)...,
            Equation(:cold_electron_temperature, TimeDerivative(), SecondaryHeatingTerm(kinetic)),
            Equation(:f, TimeDerivative(), fp)]
        u0 = zeros(length(layout))
        set_ion_densities!(u0, layout, background)
        deposit_neutrals!(u0, layout, composition, "D" => 1e17, "Ar" => 1e17)
        unknown_values(u0, layout, 3) .= temperatures
        n_free = [free_electron_density(ion_densities(u0, layout, composition), ir)
                  for ir in 1:nr]
        unknown_values(u0, layout, 2) .= n_free
        for ir in 1:nr
            m = momentum_grid(ion_grid, ir)
            shape = [exp(-p^2 / (sqrt(1 + p^2) + 1) /
                         (temperatures[ir] / Constants.ELECTRON_REST_ENERGY))
                     for p in m.p_centers, _ in 1:3]
            kinetic_view(u0, layout, 4, ir) .= n_free[ir] .* shape ./
                                               sum(kinetic.weights[ir] .* shape)
        end
        system = EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u0)
        A = newton_matrix(system, u0, 1e-7)
        n = size(A, 1)
        # the couplings this test is about are in the matrix
        f_rows = unknown_range(layout, 4, 2)
        ion_rows = unknown_range(layout, 1, 2)
        @test count(!iszero, A[f_rows, unknown_range(layout, 1, 2)]) > 0      # f rows, charge-state columns
        @test count(!iszero, A[f_rows, unknown_range(layout, 3, 2)]) > 0      # f rows, temperature column
        @test count(!iszero, A[ion_rows, f_rows]) > 0                         # kinetic ionization block
        factors = Runes2P.preconditioner_factors(FluidSchurLU(), A, layout)
        @test Runes2P.factorize!(factors, FluidSchurLU(), A)
        x = randn(rng, n)
        @test ldiv!(zeros(n), factors, x) ≈ A \ x rtol = 1e-8
        b = randn(rng, n)
        solution, stats = SCHUR_TEST_GMRES(
            A, b; N = factors, ldiv = true, restart = true, memory = 50, itmax = 50,
            rtol = 1e-10, atol = 0.0)
        @test stats.solved
        @test stats.niter <= 2
        @test solution ≈ A \ b rtol = 1e-7
    end

    # the low-rank terms of the blocks in the residuals between the stages and
    # in the columns of the Schur stage, with the fluid block and the lines both present
    @testset "with the field-particle term" begin
        system, layout, u0 = coupled_system(3e-4; field_particle = true)
        dt = 10 * collision_time
        times = collect(range(0, 20 * dt; length = 6))
        direct = integrate(system, u0, times; solver = NewtonSolver(linear_solver = SparseLU()))
        @test all(direct.statistics.accepted)
        for preconditioner in (FluidSchurLU(), FluidSchurLU(; radial_lines = false))
            krylov = integrate(system, u0, times;
                solver = NewtonSolver(linear_solver = GMRES(; preconditioner)))
            @test all(krylov.statistics.accepted)
            @test krylov.states[:, end] ≈ direct.states[:, end] rtol = 1e-7
            @test sum(krylov.statistics.krylov_iterations) <=
                  20 * sum(krylov.statistics.newton_iterations)
        end
        # the incomplete factorization meets the zero pivot of the constraint row
        @test_throws "RadiusBlockILU" integrate(system, u0, times;
            solver = NewtonSolver(linear_solver = GMRES(;
                preconditioner = RadiusBlockILU())))
    end

    @testset "checks and record" begin
        system, layout, u0 = coupled_system(0.0)
        A = newton_matrix(system, u0, 1e-6)
        # pitch grids that differ between radii: the Schur stage works, the lines are refused
        # when asked for, and the default leaves them out with a warning
        grids = [MomentumGrid(uniform_p_faces(24, 0.0, 7 * sqrt(2theta)),
                     ir == 1 ? uniform_xi_faces(4) :
                     biuniform_theta_xi_faces(4, 3, 0.9))
                 for ir in 1:radii]
        other = cylindrical_grid(grids, radial)
        other_layout = UnknownLayout(radial,
            [RadialUnknown(:electric_field), KineticUnknown(:hot_tail_distribution, other)])
        other_matrix = sparse(1.0IdentityMatrix, length(other_layout), length(other_layout))
        @test_throws "one pitch grid" Runes2P.preconditioner_factors(
            FluidSchurLU(; radial_lines = true), other_matrix,
            other_layout)
        # pitch grids over different intervals have no overlaps at all
        apart = cylindrical_grid(
            [MomentumGrid(uniform_p_faces(24, 0.0, 7 * sqrt(2theta)),
                 collect(range(-1.0, ir == 1 ? 1.0 : 0.9; length = 5))) for ir in 1:radii],
            radial)
        apart_layout = UnknownLayout(radial,
            [RadialUnknown(:electric_field), KineticUnknown(:hot_tail_distribution, apart)])
        @test_throws "one pitch grid" Runes2P.preconditioner_factors(
            FluidSchurLU(; radial_lines = true),
            sparse(1.0IdentityMatrix, length(apart_layout),
                length(apart_layout)), apart_layout)
        @test Runes2P.preconditioner_factors(
            FluidSchurLU(; radial_lines = false), other_matrix, other_layout) isa
              Runes2P.FluidSchurFactors
        fallback = @test_logs (:warn, r"without radial lines") Runes2P.preconditioner_factors(
            FluidSchurLU(), other_matrix,
            other_layout)
        @test fallback.lines === nothing
        transported, transported_layout, transported_u0 = coupled_system(1e-3)
        @test Runes2P.preconditioner_factors(
            FluidSchurLU(), newton_matrix(transported, transported_u0, 1e-6),
            transported_layout).lines !== nothing
        @test_throws "true, false or :auto" FluidSchurLU(; radial_lines = :always)
        @test GMRES().preconditioner == FluidSchurLU(; radial_lines = :auto)
        @test GMRES().restart_length == 50   # the default restart length
        # a kinetic row that couples to a radial unknown of another radius
        coupled = copy(A)
        row = first(unknown_range(layout, unknown_id(layout, :hot_tail_distribution), 1))
        column = only(unknown_range(layout, unknown_id(layout, :electric_field), 2))
        coupled[row, column] = 1.0
        @test_throws "couples to a radial unknown of radius 2" Runes2P.preconditioner_factors(
            FluidSchurLU(), coupled, layout)
        record = describe(NewtonSolver(linear_solver = GMRES(preconditioner = FluidSchurLU(;
            radial_lines = false))))
        @test record["linear_solver"]["preconditioner"] ==
              Dict{String, Any}("type" => "FluidSchurLU", "radial_lines" => false)
    end
end
