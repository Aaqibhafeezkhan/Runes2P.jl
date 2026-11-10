# MKL PARDISO as the linear solver of the Newton iterations, MKLPardiso of the extension
# Runes2PMKLPardisoExt: the error without Pardiso.jl and the record of the solver; with MKL,
# a system linear in the state factorized once per step, the field-particle term (the case of
# test_field_particle.jl) through the Woodbury update against SparseLU, a checkpoint and a
# resume that repeat the uninterrupted run of a system linear in the state within round-off,
# and the window of
# test_disruption_runaway_dream.jl against SparseLU within the Newton tolerance and against
# DREAM within the tolerances of that file, with the relative residual of the solves below the
# floor_linear_tolerance of NewtonSolver; and the texts of the errors of MKL that the
# extension reads from the exceptions of Pardiso.jl. The tests that need MKL are skipped when
# Pardiso.jl cannot load it; the window needs the ADF11 cache.
using Pardiso
using HDF5
using LinearAlgebra: norm
using SparseArrays: nnz, spdiagm
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
