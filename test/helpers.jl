# Helpers that several test files use. The runner loads this module once and
# every test module imports it: a test file may use these, `example_cases.jl`, Runes2P and
# Test, and nothing that another test file defines. A helper that a second file needs
# moves here, with a comment naming the file it came from.
module TestHelpers

using Runes2P
using LinearAlgebra: I

export write_adf11, SquareRootDecayTerm, ArctangentTerm, ConstantSourceTerm, ExchangeTerm, single_unknown_system,
       _checkpoint_plasma, _checkpoint_case, _same_statistics, _checkpoint_directory, RequestStop,
       ManufacturedSolution, BeamSolution, geometric_xi_faces, uniform_theta_xi_faces, biuniform_xi_faces,
       manufactured_errors, manufactured_step, iter_plasma_operator, _conservation_residual, max_rel_diff,
       BoundedGrowthTerm, _growth_step

# from test_time_stepping.jl
# test terms on a radial unknown with one component: -rate √x (throws a DomainError for
# x < 0), rate atan(x - root) (for an algebraic equation) and a constant source
struct SquareRootDecayTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
end
struct ArctangentTerm <: EquationTerm
    unknown::Symbol
    root::Float64
end
struct ConstantSourceTerm <: EquationTerm
    value::Float64
end
Runes2P.dependencies(term::Union{SquareRootDecayTerm,ArctangentTerm}) = [term.unknown]
Runes2P.dependencies(::ConstantSourceTerm) = Symbol[]

function Runes2P.rhs!(out, term::SquareRootDecayTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
        out[a] -= term.rate * sqrt(u[b])
    end
    return out
end
function Runes2P.rhs!(out, term::ArctangentTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
        out[a] += atan(u[b] - term.root)
    end
    return out
end
function Runes2P.rhs!(out, term::ConstantSourceTerm, u, layout, row, t)
    for ir in 1:radial_cell_count(layout), a in unknown_range(layout, row, ir)
        out[a] += term.value
    end
    return out
end
function Runes2P.sparsity!(entries, term::Union{SquareRootDecayTerm,ArctangentTerm}, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end
Runes2P.sparsity!(entries, ::ConstantSourceTerm, layout, row) = entries

# a system of one radial unknown :x on `nr` radii
function single_unknown_system(kind, terms...; nr=1, scale=1.0, nonnegative=false, extra=())
    rad = RadialGrid(uniform_r_faces(nr, 0.5))
    unknowns = [RadialUnknown(:x; scale, nonnegative), (u for (u, _) in extra)...]
    equations = [Equation(:x, kind, terms...), (eq for (_, eq) in extra)...]
    return EquationSystem(UnknownLayout(rad, unknowns), equations)
end

# fast exchange between the two components of a radial unknown: rate (u2 - u1) and its
# negative, a stiff pair whose residual has a round-off floor above `residual_tolerance`
struct ExchangeTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
end
Runes2P.dependencies(term::ExchangeTerm) = [term.unknown]
function Runes2P.rhs!(out, term::ExchangeTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        values = radial_view(u, layout, id, ir)
        rows = unknown_range(layout, row, ir)
        difference = term.rate * (values[2] - values[1])
        out[rows[1]] += difference
        out[rows[2]] -= difference
    end
    return out
end
function Runes2P.sparsity!(entries, term::ExchangeTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout), r in unknown_range(layout, row, ir),
        c in unknown_range(layout, id, ir)
        push!(entries, (r, c))
    end
    return entries
end

# from test_checkpoint.jl
# a plasma that cools while the field rises, so that the window of the knock-on cutoff
# moves down: hotter and with a weaker field at the second radius
function _checkpoint_plasma(t; nr=2, t_end=8e-4)
    temperature = [1000.0, 2000.0][1:nr] .* (0.05)^(t / t_end)
    field = [1.0, 0.8][1:nr] .* (1 + 5 * t / t_end)
    return PlasmaState(cold_electron_temperature=temperature, cold_electron_density=fill(1e20, nr),
                       ions=[IonSpecies("D", 1, 1, fill(1e20, nr))], electric_field=field,
                       magnetic_field=fill(5.0, nr))
end

# Fokker-Planck term with the prescribed plasma, radial diffusion, a knock-on term whose
# cutoff follows the plasma, and a density moment as a second unknown. The momentum grid
# has faces every 0.025 below 0.2, where the cutoff of the energy-midpoint rule moves in this
# plasma, and every 0.125 above.
function _checkpoint_case(; treatment=:explicit, with_density=true)
    nr = 2
    mom = MomentumGrid(biuniform_p_faces(16, 8, 0.0, 0.2, 1.2), uniform_xi_faces(4))
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 1.0)))
    plasma_at(t) = _checkpoint_plasma(t; nr)
    op = AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero)
    radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
    radial.diffusion_r_r .= 50.0
    knockon = KnockOnOperator(grid, grid, 0.45)
    terms = (PrescribedFokkerPlanckTerm(:f, op, plasma_at; settings=CollisionSettings(mode=:full)),
             FokkerPlanckTerm(:f, radial),
             KnockOnTerm(:f, knockon; target_density=fill(1e20, nr), treatment))
    unknowns = Any[KineticUnknown(:f, grid; scale=1e20)]
    equations = Any[Equation(:f, TimeDerivative(), terms...)]
    if with_density
        push!(unknowns, RadialUnknown(:hot_density; scale=1e20))
        push!(equations, Equation(:hot_density, Algebraic(), DensityMomentTerm(:f),
                                  LinearTerm(:hot_density; coefficient=-1.0)))
    end
    layout = UnknownLayout(grid.radial_grid, unknowns)
    system = EquationSystem(layout, equations)
    theta = 1000.0 / Constants.ELECTRON_REST_ENERGY
    f0 = [1e20 * exp(-(sqrt(1 + p^2) - 1) / theta) for p in mom.p_centers, _ in 1:4, _ in 1:nr]
    u0 = zeros(length(layout))
    for ir in 1:nr
        unknown_view(u0, layout, :f, ir) .= f0[:, :, ir]
        with_density && (unknown_view(u0, layout, :hot_density, ir) .= density(grid, f0)[ir])
    end
    times = [0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4]
    save_times = [0.0, 2e-4, 4e-4, 5e-4, 7e-4, 8e-4]   # time points, every saved state is a computed one
    update() = KnockOnCutoffUpdate(knockon, plasma_at)
    return (; grid, mom, system, layout, u0, times, save_times, plasma_at, knockon, update)
end

# the fields of the statistics that do not hold wall times
const _UNTIMED_STATISTICS = [field for field in fieldnames(SolverStatistics) if !endswith(String(field), "time") ||
                                                                               field === :time]

function _same_statistics(a::SolverStatistics, b::SolverStatistics)
    return all(field -> isequal(getfield(a, field), getfield(b, field)), _UNTIMED_STATISTICS)
end

_checkpoint_directory() = mktempdir()

# from test_stop.jl
# an interval update that creates a file at the time points from `time` on, as the batch
# script does when the warning signal of the batch system arrives
struct RequestStop <: IntervalUpdate
    file::String
    time::Float64
end
Runes2P.update_name(::RequestStop) = "request_stop"
Runes2P.apply_update!(update::RequestStop, u, layout, t) = (t >= update.time && touch(update.file); false)
Runes2P.update_state(::RequestStop) = [0.0]
Runes2P.restore_update!(update::RequestStop, state) = update

# from test_manufactured_solutions.jl
module ManufacturedSolution
# bulk resolved by the fine region below p = 0.7, tail over the coarse region; the tail is
# 1e-15 at p = 30, so the f = 0 ghost at p_max is consistent with it
g(p) = exp(-p^2 / 0.08) + 1e-2 * exp(-p)
dg(p) = -2p / 0.08 * exp(-p^2 / 0.08) - 1e-2 * exp(-p)
h(xi) = 1 + 0.3xi + 0.2 * (3xi^2 - 1) / 2
dh(xi) = 0.3 + 0.6xi
f(p, xi) = g(p) * h(xi)
f_p(p, xi) = dg(p) * h(xi)
f_xi(p, xi) = g(p) * dh(xi)
# coefficients; the ξ fluxes vanish at ξ = ±1, as the operator's boundary condition requires
advection_p(p, xi) = -(1 + 0.1p) * (1 + 0.2xi)
diffusion_p_p(p, xi) = 0.5 + 0.1p
cross(p, xi) = 0.3 * (1 - xi^2)
advection_xi(p, xi) = 0.5 * (1 - xi^2)
diffusion_xi_xi(p, xi) = 0.4 * (1 - xi^2)
flux_p(p, xi) = advection_p(p, xi) * f(p, xi) - diffusion_p_p(p, xi) * f_p(p, xi) - cross(p, xi) * f_xi(p, xi)
flux_xi(p, xi) = advection_xi(p, xi) * f(p, xi) - diffusion_xi_xi(p, xi) * f_xi(p, xi) - cross(p, xi) * f_p(p, xi)
end

# the same coefficients with a beam-like pitch profile, peaked at ξ = 1 where runaways
# gather and where the pitch grids of DREAM's examples are finest; a quadratic profile
# would be treated too well by the centred differences in ξ
module BeamSolution
using ..ManufacturedSolution: g, dg, advection_p, diffusion_p_p, cross, advection_xi, diffusion_xi_xi
h(xi) = 1 + 0.3xi + 0.2 * exp(10 * (xi - 1))
dh(xi) = 0.3 + 2.0 * exp(10 * (xi - 1))
f(p, xi) = g(p) * h(xi)
f_p(p, xi) = dg(p) * h(xi)
f_xi(p, xi) = g(p) * dh(xi)
flux_p(p, xi) = advection_p(p, xi) * f(p, xi) - diffusion_p_p(p, xi) * f_p(p, xi) - cross(p, xi) * f_xi(p, xi)
flux_xi(p, xi) = advection_xi(p, xi) * f(p, xi) - diffusion_xi_xi(p, xi) * f_xi(p, xi) - cross(p, xi) * f_p(p, xi)
end

# pitch faces: those of avalanche/generateCylindrical.py (geometric growth with exponent
# 1.5, finest at ξ = 1), and DREAM's pitch grid types uniform in θ and biuniform in ξ
# (half of the cells above ξ = 0.6, a split without an example)
geometric_xi_faces(cell_count) = reverse(1 .- 2 .* range(0, 1; length=cell_count + 1) .^ 1.5)
uniform_theta_xi_faces(cell_count) = cos.(range(pi, 0; length=cell_count + 1))
biuniform_xi_faces(cell_count) = vcat(collect(range(-1.0, 0.6; length=cell_count - cell_count ÷ 2 + 1)),
                                      collect(range(0.6, 1.0; length=cell_count ÷ 2 + 1))[2:end])

# errors of the step on a grid, and the local truncation error L f_exact - L_exact f_exact
# next to the separation p = 0.7 and in the bulk below it
function manufactured_errors(p_faces, pitch_cells; scheme=:central, xi_faces=uniform_xi_faces(pitch_cells))
    step = manufactured_step(p_faces, xi_faces; scheme)
    error = step.solution .- step.f_exact
    separation = findfirst(>=(0.7 - 1e-12), p_faces)
    return (; max=maximum(abs, error),
            l1=sum(abs.(error) .* step.weights) / sum(abs.(step.f_exact) .* step.weights),
            truncation_at_separation=maximum(abs, step.truncation[separation-2:separation+1, :]),
            truncation_in_bulk=maximum(abs, step.truncation[separation÷2-1:separation÷2+1, :]))
end

# the manufactured step on a grid: discrete solution, exact solution at the cell centres,
# local truncation error and cell volumes
function manufactured_step(p_faces, xi_faces; scheme=:central, ms=ManufacturedSolution)
    mom = MomentumGrid(p_faces, xi_faces)
    pitch_cells = pitch_cell_count(mom)
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    op = AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero)
    pc, pf, xc, xf = mom.p_centers, mom.p_faces, mom.xi_centers, mom.xi_faces
    np = length(pc)
    for j in 1:pitch_cells, k in 1:np+1
        op.advection_p[k, j, 1] = ms.advection_p(pf[k], xc[j])
        op.diffusion_p_p[k, j, 1] = ms.diffusion_p_p(pf[k], xc[j])
        op.diffusion_p_xi[k, j, 1] = ms.cross(pf[k], xc[j])
    end
    for k in 1:pitch_cells+1, i in 1:np
        op.advection_xi[i, k, 1] = ms.advection_xi(pc[i], xf[k])
        op.diffusion_xi_xi[i, k, 1] = ms.diffusion_xi_xi(pc[i], xf[k])
        op.diffusion_xi_p[i, k, 1] = ms.cross(pc[i], xf[k])
    end
    L = operator_matrix(op)
    V = phase_space_jacobian(grid, 1)
    Vp = phase_space_jacobian_p_faces(grid, 1)
    Vxi = phase_space_jacobian_xi_faces(grid, 1)
    # exact operator from the exact fluxes (midpoint rule on the faces, second order)
    exact = [-((Vp[i+1, j] * ms.flux_p(pf[i+1], xc[j]) - Vp[i, j] * ms.flux_p(pf[i], xc[j])) / mom.p_widths[i] +
               (Vxi[i, j+1] * ms.flux_xi(pc[i], xf[j+1]) - Vxi[i, j] * ms.flux_xi(pc[i], xf[j])) / mom.xi_widths[j]) /
             V[i, j] for i in 1:np, j in 1:pitch_cells]
    f_exact = [ms.f(p, xi) for p in pc, xi in xc]
    solution = reshape((I - L) \ vec(f_exact .- exact), np, pitch_cells)
    truncation = reshape(L * vec(f_exact), np, pitch_cells) .- exact
    weights = V .* mom.p_widths .* mom.xi_widths'
    return (; solution, f_exact, truncation, weights, momentum=mom)
end

# from test_exponential_fitting.jl
function iter_plasma_operator(mom, temperature, field; scheme, p_upper_boundary=:f_zero)
    n_deuterium = 1e20
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 2.0)))
    argon = zeros(19, 1)
    argon[1, 1] = 3e19
    plasma = PlasmaState(cold_electron_temperature=[temperature], cold_electron_density=[n_deuterium],
                         ions=[IonSpecies("D", 1, 1, [n_deuterium]), IonSpecies("Ar", 18, argon)],
                         electric_field=[field], magnetic_field=[5.3])
    settings = CollisionSettings(mode=:full, screening=:partial, coulomb_logarithm=:energy_dependent,
                                 ion_size_table=:dream)
    return g, fokker_planck_operator(g, plasma, settings; scheme, p_upper_boundary)
end

# from test_advection_diffusion.jl
function _conservation_residual(g, df)
    s = 0.0
    for ir in 1:radial_cell_count(g), j in 1:pitch_cell_count(g), i in 1:momentum_cell_count(g)
        s += phase_space_jacobian(g, ir)[i, j] * momentum_grid(g).p_widths[i] *
             momentum_grid(g).xi_widths[j] * g.radial_grid.r_widths[ir] * df[i, j, ir]
    end
    return s
end

# from test_knockon_operator.jl
function max_rel_diff(a, b; floor=1e-12)
    cut = floor * maximum(abs, a)
    return maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
end

# from test_step_doubling.jl
# growth k x whose evaluation is refused above a bound (a DomainError, as a coefficient
# outside its table would raise): the backward-Euler step x / (1 - k h) overshoots the
# bound for k h near 1, and the solve fails with :domain_error
struct BoundedGrowthTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
    bound::Float64
end
Runes2P.dependencies(term::BoundedGrowthTerm) = [term.unknown]
function Runes2P.rhs!(out, term::BoundedGrowthTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
        u[b] <= term.bound || throw(DomainError(u[b], "above the bound of the growth term"))
        out[a] += term.rate * u[b]
    end
    return out
end
function Runes2P.sparsity!(entries, term::BoundedGrowthTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end

# the backward-Euler step of the growth k x
_growth_step(u, k, h) = u / (1 - k * h)

# an ADF11 file in the layout of OpenADAS (from test_adas.jl): log₁₀ of the density [cm⁻³], of the temperature
# [eV] and of the coefficients [cm³/s], eight numbers per line, one block per charge state
function write_adf11(path, Z, log_density, log_temperature, blocks; iprt=1, closing_line=false)
    open(path, "w") do io
        println(io, lpad(Z, 5), lpad(length(log_density), 5), lpad(length(log_temperature), 5), lpad(1, 5), lpad(Z, 5),
                "     /SYNTHETIC          /TEST")
        println(io, "-"^80)
        rows(values) = foreach(chunk -> println(io, join((lpad(string(round(v; digits=5)), 10) for v in chunk))),
                               Iterators.partition(values, 8))
        rows(log_density)
        rows(log_temperature)
        for (k, block) in enumerate(blocks)
            println(io, "---------------------/ IGRD= 1  / IPRT= $iprt  /--------/ Z1= $k   / DATE= 03/10/26")
            rows(vec(block))
        end
        closing_line && println(io, "-"^80)    # the dashed line after the last block of some files
        println(io, " ")      # the blank line after the last block of some files
        println(io, "C" * "-"^79)
        println(io, "C synthetic table of the tests of Runes2P")
    end
    return path
end

end
