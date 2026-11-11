# MKL PARDISO as the linear solver of the Newton iterations, MKLPardiso of the extension
# Runes2PMKLPardisoExt: the error without Pardiso.jl and the record of the solver; with MKL,
# a system linear in the state factorized once per step, the field-particle term (the case
# of test_field_particle.jl) through the Woodbury update against SparseLU, a checkpoint and
# a resume that repeat the uninterrupted run of a system linear in the state within
# round-off, and the window of test_disruption_runaway_dream.jl against SparseLU within the
# Newton tolerance and against DREAM within the tolerances of that file, with the relative
# residual of the solves below the floor_linear_tolerance of NewtonSolver; the texts of the
# errors of MKL that the extension reads from the exceptions of Pardiso.jl; and a solve
# whose residual exceeds MAX_RELATIVE_RESIDUAL of the extension as a singular matrix, with
# a new analysis after it, a Newton iteration that ends at its first solve and a step
# retried shorter, as with SparseLU. The tests that need MKL are skipped when Pardiso.jl
# cannot load it; the window needs the ADF11 cache.
using Pardiso
using HDF5
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, nnz, spdiagm, getcolptr, rowvals, nonzeros
include(joinpath(@__DIR__, "example_disruption.jl"))

# the relative residual of a solve that the Newton iterations need for the waivers of the
# residual test (`floor_linear_tolerance` of NewtonSolver)
const FLOOR_LINEAR_TOLERANCE = NewtonSolver().floor_linear_tolerance

# the largest difference between the states `a` and `b`, in units of the Newton tolerance of
# each entry of the workspace `ws` at `b` for steps of length dt
function tolerance_units(a, b, ws, dt)
    bound = ws.absolute .+ ws.relative .* abs.(b) .+ ws.rate_tolerance ./ dt
    return maximum(abs.(a .- b) ./ bound)
end

# the field-particle case of test_field_particle.jl ("steady state with a prescribed field"):
# a Fokker-Planck term and the field-particle term at two radii with a prescribed field E
function field_particle_case(E)
    T, Z, np_, nr_ = 10.0, 1, 80, 2
    densities = [1e20, 5e19]
    g = _field_particle_grid(; np_, T, nr_)
    ions = [IonSpecies("D", Z, Z, densities)]
    plasma(field) = PlasmaState(;
        cold_electron_temperature = fill(T, nr_), ions, electric_field = fill(field, nr_),
        magnetic_field = zeros(nr_))
    fM = _maxwellian_on(g, T)
    fM .*= reshape(densities ./ density(g, fM), 1, 1, :)
    op = FieldParticleOperator(g, plasma(0.0), FIELD_PARTICLE_SETTINGS)
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1e20)])
    fp = fokker_planck_operator(g, plasma(E), FIELD_PARTICLE_SETTINGS)
    system = EquationSystem(layout,
        [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, fp), FieldParticleTerm(:f, op))];
        time_scale = 1e-6)
    return (; g, system, u0 = vec(fM))
end

# the bound of the relative residual of a solve of MKLPardiso
const MAX_RELATIVE_RESIDUAL = Base.get_extension(Runes2P, :Runes2PMKLPardisoExt).MAX_RELATIVE_RESIDUAL

# a cache of MKLPardiso for `matrix` alone, of a system without matrix-free terms, and one
# solve with it: the status, the solution and the relative residual of the solve
function pardiso_cache(matrix; settings...)
    n = size(matrix, 1)
    return Runes2P.linear_solver_cache(MKLPardiso(; settings...), matrix,
        (; matrix_free_terms = ()), ones(n), ones(n))
end
function cache_solve(cache, matrix, b)
    x = zeros(size(matrix, 1))
    Runes2P.matrix_unchanged!(cache, false)
    status = Runes2P.solve!(x, cache, matrix, b, nothing, 0.0, 1.0, 1)
    return status, x, Runes2P.linear_tolerance(cache)
end

# the solve of PARDISO without the extension, with its settings: the analysis, the
# factorization and the solve of the transposed matrix of the compressed rows; the
# solution, and the relative residual
function plain_pardiso_solve(matrix, b; refinement_steps = 0)
    pardiso = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(pardiso, Pardiso.REAL_NONSYM)
    Pardiso.pardisoinit(pardiso)
    Pardiso.set_iparm!(pardiso, 12, 2)
    Pardiso.set_iparm!(pardiso, 8, refinement_steps)
    Pardiso.set_nprocs!(pardiso, 1)
    n = size(matrix, 1)
    pardiso_matrix = SparseMatrixCSC(n, n, Vector{Pardiso.MklInt}(getcolptr(matrix)),
        Vector{Pardiso.MklInt}(rowvals(matrix)), nonzeros(matrix))
    x = zeros(n)
    for phase in (Pardiso.ANALYSIS, Pardiso.NUM_FACT, Pardiso.SOLVE_ITERATIVE_REFINE)
        Pardiso.set_phase!(pardiso, phase)
        Pardiso.pardiso(pardiso, x, pardiso_matrix, b)
    end
    return x, norm(b - matrix * x) / norm(b)
end

# a matrix of m rows with the diagonal d and the cyclic shift o, the entries (i, i + 1) and
# (m, 1)
cyclic_shift(m, d, o) = spdiagm(0 => fill(d, m), 1 => fill(o, m - 1), -(m - 1) => [o])

@testset "MKL PARDISO" begin
    @testset "the solver without Pardiso.jl, and its record" begin
        # the method of the package, which the extension's method for a sparse matrix hides
        stub = Tuple{MKLPardiso, Any, Any, Any, Any}
        refusal = try
            invoke(Runes2P.linear_solver_cache, stub, MKLPardiso(),
                nothing, nothing, nothing, nothing)
        catch err
            err
        end
        @test refusal isa ArgumentError
        @test occursin("using Pardiso", refusal.msg)
        @test_throws ArgumentError MKLPardiso(threads = 0)
        @test_throws ArgumentError MKLPardiso(refinement_steps = -1)
        default_record = Dict{String, Any}(
            "type" => "MKLPardiso", "threads" => 1, "refinement_steps" => 0)
        @test describe(NewtonSolver(linear_solver = MKLPardiso()))["linear_solver"] ==
              default_record
        threaded_record = Dict{String, Any}(
            "type" => "MKLPardiso", "threads" => 8, "refinement_steps" => 10)
        @test describe(MKLPardiso(threads = 8, refinement_steps = 10)) == threaded_record
        @test Base.get_extension(Runes2P, :Runes2PMKLPardisoExt) !== nothing
    end

    if !Pardiso.mkl_is_available()
        @info "Pardiso.jl did not load MKL (Pardiso.mkl_is_available() is false): the solves with MKLPardiso are skipped"
        @test_skip Pardiso.mkl_is_available()
    else
        @testset "a linear system is factorized once per step" begin
            # the field-particle case is linear in f: the second Newton iteration of a step
            # keeps the matrix and the factorization of the first, with the Woodbury update
            case = field_particle_case(0.05)
            @test case.system.linear
            dt = 1e-6
            solver = MKLPardiso()
            kept = NewtonWorkspace(case.system, NewtonSolver(linear_solver = solver))
            fresh = NewtonWorkspace(case.system,
                NewtonSolver(linear_solver = solver, reuse_linear_matrix = false))
            u_kept, u_fresh = copy(case.u0), copy(case.u0)
            result_kept = newton_solve!(u_kept, case.u0, kept, dt, dt)
            result_fresh = newton_solve!(u_fresh, case.u0, fresh, dt, dt)
            @test result_kept.status === :converged && result_fresh.status === :converged
            @test result_kept.iterations == result_fresh.iterations == 2
            @test kept.cache.factorizations == 1
            @test fresh.cache.factorizations == 2
            @test kept.cache.low_rank !== nothing
            @test u_kept == u_fresh
            @test Runes2P.linear_tolerance(kept.cache) <= FLOOR_LINEAR_TOLERANCE
            # the next step has another start state but the same matrix of a linear system with
            # constant coefficients: the key of the reuse is the end time, so one more factorization
            v = copy(u_kept)
            newton_solve!(v, u_kept, kept, 2dt, dt)
            @test kept.cache.factorizations == 2
            # the ordering and the matching of the pattern exist after the first solve, and a
            # solve that factorizes counts its time
            @test kept.cache.analyzed
            @test Runes2P.factorization_time(fresh.cache) > 0
            # a matrix with another count of nonzeros than the pattern of the cache is refused
            n = size(kept.matrix, 1)
            other = spdiagm(0 => ones(n))
            @test nnz(other) != length(kept.cache.row_indices)
            x = zeros(n)
            @test_throws DimensionMismatch Runes2P.solve!(
                x, kept.cache, other, ones(n), u_kept, 2dt, dt, 1)
        end

        @testset "the field-particle term against SparseLU" begin
            # the run of test_field_particle.jl to the ohmic steady state: j/E with both solvers
            E = 0.05
            case = field_particle_case(E)
            times = range(0.0, 2e-5; length = 21)
            function conductivity(linear_solver)
                solution = integrate(case.system, case.u0, times;
                    solver = NewtonSolver(; linear_solver))
                f = reshape(solution.states[:, end], size(case.g))
                return current_density(case.g, f) ./ E, solution
            end
            sigma_lu, lu_run = conductivity(SparseLU())
            sigma_pardiso, pardiso_run = conductivity(MKLPardiso())
            @test all(pardiso_run.statistics.accepted)
            @test sigma_pardiso ≈ sigma_lu rtol = 1e-8
            @info "MKL PARDISO, field-particle case: Newton iterations $(sum(pardiso_run.statistics.newton_iterations)) (SparseLU $(sum(lu_run.statistics.newton_iterations))), largest relative difference of j/E $(maximum(abs.(sigma_pardiso ./ sigma_lu .- 1)))"
            @test pardiso_run.settings["solver"]["linear_solver"]["type"] == "MKLPardiso"
            # the keywords: two threads of PARDISO, and ten refinement steps, whose bound is
            # still the measured relative residual, since the refinement stops at its cap and
            # promises none; it stays at or below floor_linear_tolerance
            sigma_threads, _ = conductivity(MKLPardiso(threads = 2))
            @test sigma_threads ≈ sigma_lu rtol = 1e-8
            sigma_refined, _ = conductivity(MKLPardiso(refinement_steps = 10))
            @test sigma_refined ≈ sigma_lu rtol = 1e-8
            refined = NewtonWorkspace(case.system,
                NewtonSolver(linear_solver = MKLPardiso(refinement_steps = 10)))
            u = copy(case.u0)
            @test newton_solve!(u, case.u0, refined, 1e-6, 1e-6).status === :converged
            @test 0 < Runes2P.linear_tolerance(refined.cache) <= FLOOR_LINEAR_TOLERANCE
            @test Pardiso.get_iparm(refined.cache.pardiso, 8) == 10
            # PARDISO's output iparm(7): the refinement steps of the last solve
            performed_steps = Pardiso.get_iparm(refined.cache.pardiso, 7)
            @info "MKL PARDISO, field-particle case: $(performed_steps) refinement steps in the last solve"
            @test 0 < performed_steps <= 10
        end

        @testset "the errors of MKL that the extension tells apart" begin
            # the extension reads a singular matrix from the exceptions of Pardiso.jl, which
            # do not carry MKL's error number: -4 and -7 alone mean a singular matrix
            pardiso = Pardiso.MKLPardisoSolver()
            for code in -1:-1:-12
                err = try
                    Pardiso.check_error(pardiso, code)
                catch caught
                    caught
                end
                if code == -4
                    @test err isa Pardiso.PardisoPosDefException
                else
                    @test err isa Pardiso.PardisoException
                    @test occursin("singular", err.info) == (code == -7)
                    @test occursin("32-bit integer overflow", err.info) == (code == -8)
                end
            end
        end

        @testset "a solve with a large residual is a singular matrix" begin
            m = 100
            b = [1.0 + sin(i) / 2 for i in 1:m]
            # a zero row and column; two equal rows but for 1e-15; the Laplacian of zero flux,
            # singular with the constants in its null space
            zero_row = cyclic_shift(m, 1.0, 0.1)
            zero_row[1, :] .= 0.0
            zero_row[:, 1] .= 0.0
            near_equal_rows = SparseMatrixCSC([1.0 1.0; 1.0 1.0+1e-15])
            neumann = spdiagm(0 => [1.0; fill(2.0, m - 2); 1.0], 1 => fill(-1.0, m - 1),
                -1 => fill(-1.0, m - 1))
            for (name, matrix) in (("a zero row and column", zero_row),
                ("two nearly equal rows", near_equal_rows), ("the Laplacian", neumann))
                rhs = b[1:size(matrix, 1)]
                for refinement_steps in (0, 10)
                    # PARDISO perturbs the small pivots and returns a finite solution
                    x_plain, residual_plain = plain_pardiso_solve(
                        matrix, rhs; refinement_steps)
                    @info "MKL PARDISO, $name, $refinement_steps refinement steps: relative residual $(residual_plain)"
                    @test all(isfinite, x_plain)
                    @test residual_plain > MAX_RELATIVE_RESIDUAL
                    cache = pardiso_cache(matrix; refinement_steps)
                    status, x, residual = cache_solve(cache, matrix, rhs)
                    @test status === :singular_matrix
                    # the residual of the failed solve is kept, and so is its solution
                    @test residual == residual_plain
                    @test x == x_plain
                    @test !cache.factorized && !cache.analyzed
                end
            end
            # a well-conditioned solve passes, with the solution of PARDISO bit for bit
            for matrix in (cyclic_shift(m, 1.0, 0.1), neumann + spdiagm(0 => fill(0.5, m)))
                x_plain, residual_plain = plain_pardiso_solve(matrix, b)
                cache = pardiso_cache(matrix)
                status, x, residual = cache_solve(cache, matrix, b)
                @test status === :ok
                @test residual <= MAX_RELATIVE_RESIDUAL
                @test x == x_plain
                @test cache.factorized && cache.analyzed
            end
        end

        @testset "a new analysis after a failed solve" begin
            # the ordering and the matching of a matrix dominant on the cyclic shift pivot
            # on the shift, which in a matrix dominant on its diagonal is 1e-4: the solve
            # with them fails, and the retry of the same matrix analyses it anew and gives
            # the solution of a fresh cache bit for bit
            m = 100
            b = [1.0 + sin(i) / 2 for i in 1:m]
            shift_dominant = cyclic_shift(m, 1e-3, 1.0)
            diagonal_dominant = cyclic_shift(m, 1.0, 1e-4)
            _, x_fresh, residual_fresh = cache_solve(
                pardiso_cache(diagonal_dominant), diagonal_dominant, b)
            @test residual_fresh <= MAX_RELATIVE_RESIDUAL
            cache = pardiso_cache(shift_dominant)
            status, _, residual = cache_solve(cache, shift_dominant, b)
            @test status === :ok
            @test residual <= MAX_RELATIVE_RESIDUAL
            status, _, stale_residual = cache_solve(cache, diagonal_dominant, b)
            @info "MKL PARDISO, the analysis of another matrix: relative residual $(stale_residual), a new analysis $(residual_fresh)"
            @test status === :singular_matrix
            @test !(stale_residual <= MAX_RELATIVE_RESIDUAL)
            @test !cache.analyzed
            factorizations = cache.factorizations
            status, x, residual = cache_solve(cache, diagonal_dominant, b)
            @test status === :ok
            @test cache.analyzed
            @test cache.factorizations == factorizations + 1
            @test residual == residual_fresh
            @test x == x_fresh
            # with the shift 1e-8 the solve with the analysis of the other matrix gives a
            # residual that is not within the bound (not a number on the machine of this
            # test's first run), which fails too, and the retry recovers
            near_diagonal = cyclic_shift(m, 1.0, 1e-8)
            cache = pardiso_cache(shift_dominant)
            @test cache_solve(cache, shift_dominant, b)[1] === :ok
            status, _, stale_residual = cache_solve(cache, near_diagonal, b)
            @info "MKL PARDISO, the analysis of another matrix, the shift 1e-8: relative residual $(stale_residual)"
            @test status === :singular_matrix
            @test !(stale_residual <= MAX_RELATIVE_RESIDUAL)
            status, _, residual = cache_solve(cache, near_diagonal, b)
            @test status === :ok
            @test residual <= MAX_RELATIVE_RESIDUAL
        end

        @testset "a singular step is retried shorter at once, as with SparseLU" begin
            # dx/dt = c x with c = 1 and 0.5 at two radii: the Newton matrix 1/dt - c is
            # singular at the first radius for dt = 1, and regular for the half step
            system = single_unknown_system(
                TimeDerivative(), LinearTerm(:x; coefficient = [1.0, 0.5]); nr = 2)
            u_old = [1.0, 1.0]
            results = Dict()
            runs = Dict()
            for linear_solver in (SparseLU(), MKLPardiso())
                solver = NewtonSolver(; linear_solver)
                u = copy(u_old)
                ws = NewtonWorkspace(system, solver)
                results[linear_solver] = newton_solve!(u, u_old, ws, 1.0, 1.0)
                runs[linear_solver] = integrate(system, u_old, [0.0, 1.0]; solver)
            end
            lu_result, pardiso_result = results[SparseLU()], results[MKLPardiso()]
            # the Newton iteration ends at its first solve
            @test pardiso_result.status === lu_result.status === :singular_matrix
            @test pardiso_result.iterations == lu_result.iterations == 0
            lu_run, pardiso_run = runs[SparseLU()], runs[MKLPardiso()]
            for statistics in (lu_run.statistics, pardiso_run.statistics)
                @test statistics.status == [:singular_matrix, :converged, :converged]
                @test statistics.newton_iterations == [0, 2, 2]
                @test statistics.time_step == [1.0, 0.5, 0.5]
            end
            # two half steps of the implicit Euler method: x = 1/(1 - c/2)^2
            @test pardiso_run.states[:, end] ≈ [4.0, 16 / 9] rtol = 1e-12
            @test pardiso_run.states[:, end] ≈ lu_run.states[:, end] rtol = 1e-12
        end

        @testset "a resumed run repeats the uninterrupted one" begin
            solver = NewtonSolver(linear_solver = MKLPardiso())
            case = _checkpoint_case()
            whole = integrate(case.system, case.u0, case.times; solver,
                save_times = case.save_times, updates = (case.update(),))
            @test all(whole.statistics.accepted)
            case = _checkpoint_case()
            checkpoint = CheckpointSettings("run"; directory = _checkpoint_directory(), interval = 0)
            stop = 5
            first_saves = filter(<=(case.times[stop]), case.save_times)
            first_segment = integrate(case.system, case.u0, case.times[1:stop]; solver,
                save_times = first_saves, updates = (case.update(),), checkpoint)
            @test first_segment.states == whole.states[:, 1:length(first_saves)]
            stored = read_checkpoint(checkpoint)
            expected = Dict{String, Any}(
                "type" => "MKLPardiso", "threads" => 1, "refinement_steps" => 0)
            @test stored.settings["solver"]["linear_solver"] == expected
            case = _checkpoint_case()
            second_segment = @test_logs (:warn, r"changed settings") integrate(
                case.system, case.u0, case.times; solver, save_times = case.save_times,
                updates = (case.update(),), checkpoint, resume = true)
            later = (length(first_saves) + 1):length(case.save_times)
            # the resumed run computes the ordering and the matching of PARDISO from its own
            # first matrix, the whole run from that of the start; on this case, linear in the
            # state, the same within round-off (measured 1.4e-16)
            difference = norm(second_segment.states .- whole.states[:, later]) /
                         norm(whole.states[:, later])
            @info "MKL PARDISO, resumed run: relative difference from the whole run $(difference)"
            @test difference < 1e-12
            @test read_output(checkpoint).states ≈ whole.states rtol = 1e-12
        end

        @testset "the window of the disruption case with a runaway grid" begin
            if !isdir(adas_cache_directory())
                @info "no ADF11 cache at $(adas_cache_directory()): the window with MKLPardiso is skipped"
                @test_skip isdir(adas_cache_directory())
            else
                file = reference_file("dream_disruption_runaway_small.h5")
                start = h5open(h5 -> read_dream_state(h5["window"]), file, "r")
                times = collect(start["t"])
                dt = times[2] - times[1]
                solver = NewtonSolver(linear_solver = MKLPardiso())
                _, _, _, lu_run = run_window(start, :dream)
                case, system, u0, pardiso_run = run_window(start, :dream; solver)
                @test all(pardiso_run.statistics.accepted)
                @test pardiso_run.settings["solver"]["linear_solver"]["type"] ==
                      "MKLPardiso"
                lu_iterations = lu_run.statistics.newton_iterations
                pardiso_iterations = pardiso_run.statistics.newton_iterations
                @info "MKL PARDISO, window: Newton iterations per step $(pardiso_iterations), SparseLU $(lu_iterations)"
                # the two runs agree within the Newton tolerance of every entry
                ws = NewtonWorkspace(system, solver)
                units = maximum(tolerance_units(pardiso_run.states[:, k], lu_run.states[:, k], ws, dt)
                for k in axes(lu_run.states, 2))
                @info "MKL PARDISO, window: largest difference from SparseLU $(units) Newton tolerances"
                @test units < 1
                # and with DREAM within the tolerances of test_disruption_runaway_dream.jl
                errors = phase_differences(case, pardiso_run, start)
                for name in sort!(collect(keys(errors)))
                    @info "MKL PARDISO, window against DREAM: $name $(errors[name])"
                    @test errors[name] < WINDOW_TOLERANCES[name]
                end
                # the relative residual of the last solve of each step
                u_old = copy(u0)
                residuals = Float64[]
                for k in 2:length(times)
                    u = copy(u_old)
                    result = newton_solve!(u, u_old, ws, times[k], times[k] - times[k - 1])
                    @test result.status === :converged
                    push!(residuals, Runes2P.linear_tolerance(ws.cache))
                    u_old = u
                end
                @info "MKL PARDISO, window: relative residuals of the last solve of each step $(residuals)"
                @test all(<=(FLOOR_LINEAR_TOLERANCE), residuals)
            end
        end
    end
end
