# The disruption case with DREAM's runaway grid against DREAM: DREAM's examples/Disruptions
# generate.py --kinetic --cylindrical, scenario 1, at the size of the tests (2 radii, a hot-tail
# grid of 30 × 4 cells) with the case script's runaway grid
# (reference/dream/cases/disruption_kinetic.py --runaway-grid: 32 × 8 cells from the hot-tail p_max
# 0.8 to 20). The window of dream_disruption_runaway_small.h5 is DREAM's state at step 170 of the
# quench, 1.31e-4 s after the start of the scenario, where the runaway current is 5.9 % of the
# plasma current, and the 10 steps of 6.53e-7 s after it. The system of example_disruption.jl with
# `runaway = true` starts from that state and runs the 10 steps with DREAM's settings: the flux
# through the interface of the two grids into f_re and n_re, the kinetic avalanche on f_re and the
# fluid one in n_re above the hot-tail p_max, the runaway current as the moment of f_re, the kinetic
# ionization by f_re and the heating of the cold electrons by it, with every term of the case
# without a runaway grid. At the first state, DREAM's j_re is checked against the moment of its f_re
# on the built grid, and n_re against ∫f_re (n_re exceeds it by the runaways beyond the grid's
# p_max). The budgets of the assembled system and of the runaway grid are asserted independently of
# DREAM, and the window runs once more with Runes2P's defaults (compat = runes2p), where only the
# budgets are checked. The allocation of jacobian! at the first state is bounded, and so is that
# of the plasma columns of each term that adds them, of the local Jacobians of the terms that read
# every charge state of a radius and of the residuals of the ion rates and the radiated power. The
# blocks of these terms and of the friction term, which keep buffers and weights between calls,
# are checked bit for bit at the state and at a changed state: against the block of the public
# add_local_jacobian!, built with fresh buffers, for the local Jacobians, and against the block of
# a new term for the friction term. The moments of the density and the current keep the positions
# and weights of their rows for the pattern of J: their blocks on one matrix, at the state, at a
# changed state and at the state again, and on a new matrix are checked bit for bit against those
# of a new term, and the allocation of their Jacobians is bounded. Needs the ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

#! format: off
# the closures of the runaway grid over the window (`runaway_budgets`), at least 44 times the
# larger of the values measured at -O0 and -O2 (compat = dream / runes2p): the particles through
# the interface to round-off, and the balance of n_re − ∫f_re against the avalanche beyond the
# grid and the outflow there
const WINDOW_RUNAWAY_BUDGETS = (interface = 2e-14,   # 2.8e-16 / 4.5e-16
                                avalanche = 1e-11)   # 1.3e-13 / 9.0e-14
#! format: on

# the bytes a call of jacobian! of the window allocates after a first call, and those of the
# plasma columns of each term that adds them: about twice those measured at -O2 in this
# testset (in the comment). A loop that inserts the partials of a dual number whose type is
# picked at run time boxes every entry: it exceeds each of these bounds several times over
# (1.5e5 bytes for the plasma columns of the interface term of f, the fewest boxed entries).
const JACOBIAN_ALLOCATION = 3.5e5           # 1.76e5
const PLASMA_COLUMNS_ALLOCATION = 1.6e4     # 8.0e3 (InterfaceParticleFluxTerm, largest), 7.8e3 (FokkerPlanckTerm)
# the bytes of the Jacobian of FrictionEnergyTransferTerm with the weights it keeps, about
# twice those measured at -O2; its dual collision model is built at every call
const FRICTION_JACOBIAN_ALLOCATION = 1.2e5  # 5.7e4
# the bytes of the Jacobian of each term that assembles it from one evaluation in dual numbers
# on the charge states of a radius (add_local_jacobian!) with the buffers it keeps, about twice
# the largest measured at -O2, and of the residuals of the ion rates and the radiated power, which
# allocate nothing at -O2, a margin of 1e3 bytes
const LOCAL_JACOBIAN_ALLOCATION = 1.6e4     # 7.8e3 (IonEnergyExchangeTerm), 5.1e2 (ConductivityTerm)
const ION_RESIDUAL_ALLOCATION = 1e3         # 0 (IonRateTerm, RadiatedPowerTerm)
# the bytes of the Jacobian of a moment of the density or the current with the positions and
# weights it keeps, a margin of 1e3 bytes: the moments over the hot region allocate the limits of
# the region at the radii, the others nothing
const MOMENT_JACOBIAN_ALLOCATION = 1e3      # 1.6e2 (hot region), 0 (the others)

# the block of the bound term `b` of `system` at the state u, added by `add!` to a zero Jacobian
function term_block(system, b, u, t, add!)
    J = jacobian_prototype(system)
    fill!(J.nzval, 0.0)
    add!(J, b.term, u, system.layout, b.row, t)
    return J.nzval
end

# the block of the bound term `b` of `system` at the state u built afresh: for a friction term
# by a new term with its arguments, for a term with a local Jacobian by the public
# add_local_jacobian!, whose buffers are made at each call
function fresh_block(system, b, u, t)
    if b.term isa FrictionEnergyTransferTerm
        term = b.term
        new_term = FrictionEnergyTransferTerm(term.kinetic, term.grid, term.source;
            settings = term.settings, coefficient = term.coefficient)
        return term_block(
            system, (term = new_term, row = b.row), u, t, Runes2P.add_jacobian!)
    end
    return term_block(system, b, u, t, Runes2P.add_local_jacobian!)
end

# a new moment term of the density or the current with the arguments of `term`
function new_moment(term::DensityMomentTerm, layout)
    if term.envelopes === nothing
        return DensityMomentTerm(term.kinetic; coefficient = term.coefficient)
    end
    grid = kinetic_grid(layout, unknown_id(layout, term.kinetic))
    return DensityMomentTerm(term.kinetic, grid; coefficient = term.coefficient,
        region = term.envelopes.region)
end
function new_moment(term::CurrentMomentTerm, layout)
    region = term.envelopes === nothing ? nothing : term.envelopes.region
    return CurrentMomentTerm(term.kinetic, term.grid; coefficient = term.coefficient,
        pitch_rule = term.pitch_rule, region)
end

# the block of the bound term `b` at the state u, added to the matrix J after its values are zeroed
function kept_block(J, system, b, u, t)
    fill!(J.nzval, 0.0)
    Runes2P.add_jacobian!(J, b.term, u, system.layout, b.row, t)
    return copy(J.nzval)
end

# the largest relative residual of the algebraic rows `names` at the state `u` of `case`
function algebraic_mismatch(case, system, u, t, dt, names)
    residual = zeros(length(u))
    residual!(residual, u, u, system, t, dt)
    return maximum(maximum(abs(residual[i]) / abs(u[i])
                   for i in unknown_range(case.layout, unknown_id(case.layout, name), ir))
    for name in names, ir in 1:case.nr)
end

@testset "the disruption case with a runaway grid against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the disruption case with a runaway grid against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = reference_file("dream_disruption_runaway_small.h5")
        start = h5open(h5 -> read_dream_state(h5["window"]), file, "r")
        times = collect(start["t"])
        @test haskey(start, "f_re")
        case, system, u0, solution = run_window(start, :dream)
        # DREAM's first state: the runaway current is the moment of f_re, and n_re holds ∫f_re
        # and the runaways beyond the grid's p_max, 3e-6 and 8e-6 of n_re at the two radii
        @test algebraic_mismatch(case, system, u0, times[1], times[2] - times[1],
            (:runaway_current_density,)) < 1e-13   # 1.1e-15
        n_re = collect(unknown_values(u0, case.layout, unknown_id(case.layout, :runaway_density)))
        on_grid = density(case.runaway_grid,
            kinetic_array(u0, case.layout, unknown_id(case.layout, :f_re)))
        @test all(0 .< (n_re .- on_grid) ./ n_re .< 1e-4)
        @test all(solution.statistics.accepted)
        errors = phase_differences(case, solution, start)
        @test Set(keys(WINDOW_TOLERANCES)) == Set(keys(errors))
        for name in sort!(collect(keys(errors)))
            @info "the disruption case with a runaway grid against DREAM: $name $(errors[name])"
            @test errors[name] < WINDOW_TOLERANCES[name]
        end
        for (_, value, bound) in budget_checks(case, solution, "window, compat = dream", DRIFT["quench"])
            @test value < bound
        end
        for (_, value, bound) in runaway_budget_checks(
            case, solution, "window, compat = dream",
            WINDOW_RUNAWAY_BUDGETS)
            @test value < bound
        end
        # the plasma columns insert their partials without a box per entry
        J = jacobian_prototype(system)
        jacobian!(J, system, u0, times[1])
        allocated = @allocated jacobian!(J, system, u0, times[1])
        @info "the disruption case with a runaway grid against DREAM: jacobian! $allocated B"
        @test allocated < JACOBIAN_ALLOCATION skip = !OPTIMIZED
        @test sort!([string(nameof(typeof(b.term))) for b in system.column_terms]) ==
              ["FokkerPlanckTerm", "FokkerPlanckTerm", "InterfaceParticleFluxTerm",
            "MomentumInterfaceTerm", "MomentumInterfaceTerm"]
        for b in system.column_terms
            Runes2P.add_plasma_columns!(J, b.term, u0, case.layout, b.row, times[1])
            allocated = @allocated Runes2P.add_plasma_columns!(
                J, b.term, u0, case.layout, b.row, times[1])
            @info "the disruption case with a runaway grid against DREAM: plasma columns of the $(nameof(typeof(b.term))) of $(case.layout.unknowns[b.row].name) $allocated B"
            @test allocated < PLASMA_COLUMNS_ALLOCATION skip = !OPTIMIZED
        end
        # the terms with a local Jacobian keep its buffers, and the friction term its weights:
        # their kept blocks at the state and at another state are the same bit for bit as
        # those built afresh, with the fresh buffers of the public add_local_jacobian! for the
        # local Jacobians and by a new term for the friction term, and the kept blocks at the
        # state are the same again after the other state; RunawayFluidEnergyTransferTerm, the
        # fourth term with kept buffers, is in the cases without a runaway grid only
        other = copy(u0)
        for name in (:cold_electron_temperature, :cold_electron_density, :ion_densities)
            for i in unknown_range(case.layout, unknown_id(case.layout, name), 1)
                other[i] *= 1.01
            end
        end
        local_types = Union{ConductivityTerm, RadiatedPowerTerm, IonEnergyExchangeTerm,
            RunawayFluidEnergyTransferTerm}
        local_terms = [b for b in system.direct_terms if b.term isa local_types]
        @test sort!(unique([string(nameof(typeof(b.term))) for b in local_terms])) ==
              ["ConductivityTerm", "IonEnergyExchangeTerm", "RadiatedPowerTerm"]
        friction_terms = [b
                          for b in system.direct_terms
                          if b.term isa FrictionEnergyTransferTerm]
        @test !isempty(friction_terms)
        for b in [local_terms; friction_terms]
            first_block = term_block(system, b, u0, times[1], Runes2P.add_jacobian!)
            @test isequal(fresh_block(system, b, u0, times[1]), first_block)
            other_block = term_block(system, b, other, times[1], Runes2P.add_jacobian!)
            @test !isequal(other_block, first_block)
            @test isequal(fresh_block(system, b, other, times[1]), other_block)
            @test isequal(term_block(system, b, u0, times[1], Runes2P.add_jacobian!), first_block)
            @test isequal(term_block(system, b, u0, times[1], Runes2P.add_jacobian!), first_block)
        end
        for b in local_terms
            allocated = call_allocation(
                Runes2P.add_jacobian!, J, b.term, u0, case.layout, b.row, times[1])
            @info "the disruption case with a runaway grid against DREAM: local Jacobian of the $(nameof(typeof(b.term))) of $(case.layout.unknowns[b.row].name) $allocated B"
            @test allocated < LOCAL_JACOBIAN_ALLOCATION skip = !OPTIMIZED
        end
        for b in friction_terms
            allocated = call_allocation(
                Runes2P.add_jacobian!, J, b.term, u0, case.layout, b.row, times[1])
            @info "the disruption case with a runaway grid against DREAM: Jacobian of the $(nameof(typeof(b.term))) of $(case.layout.unknowns[b.row].name) $allocated B"
            @test allocated < FRICTION_JACOBIAN_ALLOCATION skip = !OPTIMIZED
        end
        # the moments of the density and the current find the positions of their rows once
        # per pattern of J; the hot-region moments follow the temperature, which differs at the
        # changed state
        moment_terms = [b
                        for b in system.direct_terms
                        if b.term isa Union{DensityMomentTerm, CurrentMomentTerm}]
        @test sort!([string(nameof(typeof(b.term))) for b in moment_terms]) ==
              ["CurrentMomentTerm", "CurrentMomentTerm",
            "CurrentMomentTerm", "DensityMomentTerm"]
        @test count(b -> b.term.envelopes !== nothing, moment_terms) == 2
        kept = jacobian_prototype(system)
        for b in moment_terms
            fresh = (term = new_moment(b.term, case.layout), row = b.row)
            first_block = kept_block(kept, system, b, u0, times[1])
            @test isequal(term_block(system, fresh, u0, times[1], Runes2P.add_jacobian!), first_block)
            other_block = kept_block(kept, system, b, other, times[1])
            @test isequal(other_block, first_block) == (b.term.envelopes === nothing)
            @test isequal(term_block(system, fresh, other, times[1], Runes2P.add_jacobian!), other_block)
            @test isequal(kept_block(kept, system, b, u0, times[1]), first_block)
            # on a new matrix of the pattern, then on the kept one again
            @test isequal(term_block(system, b, other, times[1], Runes2P.add_jacobian!), other_block)
            @test isequal(kept_block(kept, system, b, u0, times[1]), first_block)
            allocated = call_allocation(
                Runes2P.add_jacobian!, kept, b.term, u0, case.layout, b.row, times[1])
            @info "the disruption case with a runaway grid against DREAM: moment Jacobian of the $(nameof(typeof(b.term))) of $(case.layout.unknowns[b.row].name) $allocated B"
            @test allocated < MOMENT_JACOBIAN_ALLOCATION skip = !OPTIMIZED
        end
        ion_terms = [b
                     for b in system.sparse_terms
                     if b.term isa Union{IonRateTerm, RadiatedPowerTerm}]
        @test sort!([string(nameof(typeof(b.term))) for b in ion_terms]) ==
              ["IonRateTerm", "RadiatedPowerTerm"]
        out = zeros(length(u0))
        for b in ion_terms
            allocated = call_allocation(rhs!, out, b.term, u0, case.layout, b.row, times[1])
            @info "the disruption case with a runaway grid against DREAM: residual of the $(nameof(typeof(b.term))) $allocated B"
            @test allocated < ION_RESIDUAL_ALLOCATION skip = !OPTIMIZED
        end
        # Runes2P's defaults: the budgets alone, since the physics departs from DREAM's
        case, _, _, solution = run_window(start, :runes2p)
        @test all(solution.statistics.accepted)
        for (_, value, bound) in budget_checks(case, solution, "window, compat = runes2p", DRIFT["quench"])
            @test value < bound
        end
        for (_, value, bound) in runaway_budget_checks(
            case, solution, "window, compat = runes2p",
            WINDOW_RUNAWAY_BUDGETS)
            @test value < bound
        end
    end
end
