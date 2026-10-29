# Shared helpers of the example scripts: command-line switches (`name=value`), the
# plasma, grid, scheme, collision and solver builders from them, checkpoints and stop,
# the run report and HDF5 output. Every switch has a default, so each script runs without
# arguments.
using Runes2P, Printf, Statistics
using HDF5   # loads Runes2P's HDF5 extension, which defines write_output

"Value of the switch `name=value` on the command line, or `default`."
function argument(name, default)
    for a in ARGS
        startswith(a, name * "=") && return String(a[(length(name) + 2):end])
    end
    return default
end
has_argument(name) = any(a -> startswith(a, name * "="), ARGS)
integer(name, default) = parse(Int, argument(name, string(default)))
number(name, default) = parse(Float64, argument(name, string(default)))
symbol(name, default) = Symbol(argument(name, string(default)))
boolean(name, default) = argument(name, string(default)) in ("true", "1", "yes")

"""
Momentum grid from the switches `grid=uniform|biuniform|tanh|biuniform_270`, `np=`,
`p_max=`, `np_low=` and `p_sep=` (cells below and position of the split of the biuniform
and tanh grids), and the pitch grid from `pitch=uniform|theta`, `nxi=`, `nxi_low=`,
`theta_sep=` (cells below and position, in radians from ξ = 1, of the split of the
biuniform-θ grid). `grid=biuniform_270` is a fully kinetic hot-tail grid of 270 cells to
p = 2.5 with 70 below 0.07 and 15 pitch cells with 10 within 0.5 rad of ξ = 1 (a choice of
this package; see the docs page *Choices of this package*), and ignores the other switches.
"""
function momentum_grid_from_switches(; np = 200, nxi = 15, p_max = 2.0)
    kind = symbol("grid", :uniform)
    kind === :biuniform_270 &&
        return MomentumGrid(biuniform_p_faces(270, 70, 0.0, 0.07, 2.5), biuniform_theta_xi_faces(15, 10, 0.5))
    np, nxi, p_max = integer("np", np), integer("nxi", nxi), number("p_max", p_max)
    np_low, p_sep = integer("np_low", np ÷ 4), number("p_sep", 0.1 * p_max)
    p_faces = kind === :uniform ? uniform_p_faces(np, 0.0, p_max) :
              kind === :biuniform ? biuniform_p_faces(np, np_low, 0.0, p_sep, p_max) :
              kind === :tanh ? tanh_p_faces(np, np_low, 0.0, p_sep, p_max) :
              error("grid must be uniform, biuniform, tanh or biuniform_270, not $kind")
    pitch = symbol("pitch", :uniform)
    xi_faces = pitch === :uniform ? uniform_xi_faces(nxi) :
               pitch === :theta ?
               biuniform_theta_xi_faces(nxi, integer("nxi_low", 2nxi ÷ 3), number("theta_sep", 0.5)) :
               error("pitch must be uniform or theta, not $pitch")
    return MomentumGrid(p_faces, xi_faces)
end

"""
Pure deuterium plasma at `nr` radii from `T=` [eV], `n=` [m⁻³], `B=` [T] and the field
`E=` [V/m] or `e_over_ec=` (multiples of the critical field at `T` and `n`); `n_ar=` adds
neutral argon (for the screening switches).
"""
function plasma_from_switches(nr; T = 1e3, n = 1e20, e_over_ec = 5.0, B = 5.0)
    T, n, B = number("T", T), number("n", n), number("B", B)
    E = has_argument("E") ? number("E", 0.0) :
        number("e_over_ec", e_over_ec) * critical_field(n, thermal_coulomb_logarithm(n, T))
    ions = [IonSpecies("D", 1, 1, fill(n, nr))]
    n_ar = number("n_ar", 0.0)
    if n_ar > 0
        argon = zeros(19, nr)
        argon[1, :] .= n_ar   # neutral
        push!(ions, IonSpecies("Ar", 18, argon))
    end
    return PlasmaState(cold_electron_temperature = fill(T, nr),
        cold_electron_density = fill(n, nr), ions = ions,
        electric_field = fill(E, nr), magnetic_field = fill(B, nr))
end

"""
`mode=full|superthermal|ultra_relativistic`, `screening=partial|complete|none`,
`lnlambda=thermal|energy_dependent|relativistic`, `ion_size_table=hesslow|dream`: the
switches of `CollisionSettings` (its own defaults), validated by its constructor.
"""
function collision_settings_from_switches(; mode = :full)
    CollisionSettings(
        mode = symbol("mode", mode), screening = symbol("screening", :partial),
        coulomb_logarithm = symbol("lnlambda", :thermal), ion_size_table = symbol("ion_size_table", :hesslow))
end

"""
Scheme switches: `scheme=` sets both directions (`exponential_fitting`, the default;
`central`, `upwind`, `quick`, `upwind_second_order`, or a limiter `van_albada`, `tcdf`,
`muscl`, `smart`, `ospre`), `p_scheme=` and `xi_scheme=` override one direction;
`pmax_boundary=f_zero|constant_flux|extrapolated_flux|extrapolated_flux_consistent|outflow`.
"""
function scheme_switches()
    scheme = symbol("scheme", :exponential_fitting)
    return (p_scheme = symbol("p_scheme", scheme), xi_scheme = symbol("xi_scheme", scheme),
        p_upper_boundary = symbol("pmax_boundary", :f_zero))
end

"`jacobian=exact|frozen_limiter|upwind|central|exponential_fitting`: the stored Jacobian of a limiter."
jacobian_switch() = symbol("jacobian", :exact)

"""
Newton and linear solver from `solver=lu|gmres`, `preconditioner=lu|ilu`, `ilu_drop=`,
`refresh=every_iteration|every_step|on_demand`, `operator=exact|stored` (the Newton
operator over an approximate stored Jacobian), `restart=`, `max_iterations=`,
`residual_tolerance=`, `line_search=`.
"""
function solver_from_switches(; solver = :lu, refresh = :every_step)
    kind = symbol("solver", solver)
    linear = if kind === :lu
        SparseLU()
    elseif kind === :gmres
        preconditioner = symbol("preconditioner", :lu) === :ilu ?
                         RadiusBlockILU(drop_tolerance = number("ilu_drop", 1e-4)) :
                         RadiusBlockLU()
        GMRES(; preconditioner, preconditioner_refresh = symbol("refresh", refresh),
            operator = symbol("operator", :exact),
            restart_length = integer("restart", 30))
    else
        error("solver must be lu or gmres, not $kind")
    end
    return NewtonSolver(
        linear_solver = linear, max_iterations = integer("max_iterations", 30),
        residual_tolerance = number("residual_tolerance", 1e-8), line_search = boolean("line_search", false))
end

"Maxwell-Jüttner distribution of temperature `T` [eV] and density `n` per radius on `grid`."
function maxwellian(grid, T, n, nr)
    mom = momentum_grid(grid)
    theta = T / Constants.ELECTRON_REST_ENERGY
    f = [exp(-(sqrt(1 + p^2) - 1) / theta)
         for p in mom.p_centers, _ in 1:pitch_cell_count(mom), _ in 1:nr]
    f .*= n ./ reshape(density(grid, f), 1, 1, nr)
    return f
end

"Time points from `dt=` and `t_end=`, and the `saves=` save times over the run."
function times_from_switches(; dt = 1e-5, t_end = 1e-3, saves = 11)
    dt, t_end = number("dt", dt), number("t_end", t_end)
    return collect(0:dt:t_end), collect(range(0, t_end; length = integer("saves", saves)))
end

"""
Checkpoints, resume and stop of a run in segments, the keywords of `integrate`:
`checkpoint=` is the name of the run (no checkpoints without it), `checkpoint_dir=` the
directory of its files, `checkpoint_interval=` the wall time between checkpoints [s],
`overwrite=true` lets a run replace the segment files of an earlier one, `resume=true`
continues from the last checkpoint when there is one. `stop_file=` and `wall_budget=` [s]
stop the segment at a time point, with `slack=` [s] in the margin. In a Slurm job with
checkpoints the segment stops before the end of the job without further switches.
`log=stdout` or `log=<file>` writes the progress log, and `request_file=` names the file
whose appearance asks for an entry.
"""
function run_options()
    name = argument("checkpoint", "")
    checkpoint = isempty(name) ? nothing :
                 CheckpointSettings(name; directory = argument("checkpoint_dir", "."),
        interval = number("checkpoint_interval", 1800.0), overwrite = boolean("overwrite", false))
    resume = checkpoint !== nothing && boolean("resume", false)
    asked = has_argument("stop_file") || has_argument("wall_budget")
    stop = asked || (checkpoint !== nothing && haskey(ENV, "SLURM_JOB_END_TIME")) ?
           StopCondition(
        stop_file = has_argument("stop_file") ? argument("stop_file", "") : nothing,
        wall_budget = has_argument("wall_budget") ? number("wall_budget", 0.0) : nothing,
        slack = number("slack", 300.0)) : nothing
    target = argument("log", "")
    request_file = has_argument("request_file") ? argument("request_file", "") : nothing
    log = isempty(target) ? nothing :
          target == "stdout" ? ProgressLog(; request_file) :
          ProgressLog(; file = target, request_file)
    return (; checkpoint, resume, stop, log)
end

"End the script with the exit status of the run, which the batch script of a segment reads."
finish(solution) = exit(exit_status(solution))

"Solver statistics, positivity and density of a run; `output=file.h5` writes the solution."
function report(solution, grid; wall = NaN)
    segment = solution.settings["segment"]
    @printf("segment %d from t = %.3e s to %.3e s: %s%s\n",
        segment["index"], segment["start_time"], segment["end_time"],
        segment["status"],
        haskey(segment, "stop_reason") ? " (" * segment["stop_reason"] * ")" : "")
    st = solution.statistics
    accepted = st.accepted
    steps = count(accepted)
    newton = sum(st.newton_iterations)
    @printf("steps: %d accepted of %d attempted (%d halved); Newton iterations per step %.2f",
        steps, length(accepted),
        length(accepted) - steps, newton / max(steps, 1))
    sum(st.krylov_iterations) > 0 &&
        @printf("; Krylov iterations per Newton iteration %.2f; factorizations per step %.2f",
            sum(st.krylov_iterations) / max(newton, 1),
            sum(st.preconditioner_updates) / max(steps, 1))
    @printf("; steps accepted at the residual floor %d, as a linear system %d",
        count(st.accepted_at_residual_floor .& .!st.accepted_as_linear),
        count(st.accepted_as_linear))
    isnan(wall) || @printf("; wall time %.2f s", wall)
    println()
    @printf("time per step: %.1f ms, of which Jacobian %.1f, residual %.1f, factorization %.1f, linear solve %.1f\n",
        1e3 * mean(st.wall_time), 1e3 * mean(st.jacobian_time), 1e3 *
                                                                mean(st.residual_time),
        1e3 * mean(st.factorization_time), 1e3 * mean(st.linear_solve_time))
    states = solution.states
    if isempty(states)
        println("the segment saved no state")
        return solution
    end
    negative = count(<(0), states)
    @printf("positivity: %d negative entries over the saved states; min f / max f = %.2e\n",
        negative, minimum(states) / maximum(states))
    for (k, t) in enumerate(solution.times)
        f = reshape(states[:, k], size(grid))
        n = density(grid, f)
        @printf("t = %9.3e s: density %s m⁻³\n", t,
            join((@sprintf("%.6e", x) for x in n), " "))
    end
    file = argument("output", "")
    if !isempty(file)
        write_output(file, solution)
        println("solution written to ", file)
    end
    return solution
end
