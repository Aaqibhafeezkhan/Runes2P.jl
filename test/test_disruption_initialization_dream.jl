# The initialization of the disruption case against DREAM, at the size of
# test_disruption_dream.jl: runs 0 to 2 of DREAM's examples/Disruptions/generate.py --kinetic
# --cylindrical, scenario 1 (reference/dream/cases/disruption_kinetic.py --nr 2 --np1 10 --np2 20
# --nxi 4, dream_disruption_chain_small.h5). Runes2P builds the start of the ionization phase
# itself (`initialize` of example_disruption.jl): the conductivity of the initial plasma and the
# field of the prescribed current (run 0), 1 s in 3 steps from the Maxwellian with that field
# held (run 1), and again with the field rescaled to the plasma current (run 2). The start of run 1
# and each saved step after the start of runs 1 and 2 are compared with DREAM's, and the state
# handed to the ionization phase with DREAM's first state of that phase. The runaway density and current are compared over the radii:
# at the second radius n_re is 0.008 m⁻³, 1e-14 of the first radius and far below DREAM's
# absolute tolerance, and differs by up to 8e-4 there. Needs the ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

# the largest relative difference of each quantity over the steps of runs 1 and 2: about ten
# times the larger of the two runs, measured at the default optimisation level (in the comment).
# DREAM's Newton tolerance of these runs is 1e-7 (generate.py:276).
#! format: off
const TOLERANCES = Dict("E_field" => 1e-8,    # 7.7e-10, run 2 (the rescale by I_p of run 1)
                        "I_p" => 1e-8,        # 7.8e-10, run 1
                        "j_ohm" => 1e-8,      # 7.5e-10
                        "j_tot" => 1e-8,      # 7.5e-10
                        "j_hot" => 1e-8,      # 7.9e-10
                        "psi_p" => 5e-9,      # 4.9e-10
                        "n_re" => 1e-9,       # 8.4e-11
                        "j_re" => 1e-9,       # 8.4e-11
                        "f_hot" => 3e-10,     # 2.7e-11
                        "n_hot" => 2e-10,     # 1.7e-11
                        "n_i" => 1e-13,       # round-off of the D background
                        "n_cold" => 2e-14,    # 2.0e-15
                        "W_cold" => 1e-14,    # 2.7e-16
                        "T_cold" => 1e-14)    # round-off of the prescribed profile
#! format: on
# the state handed to the ionization phase against DREAM's first state of that phase, per unknown
# relative to its largest value: the differences of the end of run 2 (7.7e-10 measured)
const HANDOVER = 1e-8

const NAMES = ("E_field" => :electric_field, "j_ohm" => :ohmic_current_density,
    "j_hot" => :hot_electron_current_density, "j_re" => :runaway_current_density,
    "j_tot" => :total_current_density, "n_hot" => :hot_electron_density,
    "n_cold" => :cold_electron_density, "n_re" => :runaway_density,
    "psi_p" => :poloidal_flux, "W_cold" => :cold_thermal_energy,
    "T_cold" => :cold_electron_temperature)

# the largest difference of each quantity over the radii and the saved steps `steps`, relative
# to the largest value of DREAM over the radii at each step
function differences(run, dream, steps)
    layout, solution = run.case.layout, run.solution
    errors = Dict{String, Float64}()
    for (name, ours) in NAMES
        id = unknown_id(layout, ours)
        errors[name] = maximum(maximum(abs,
                                   unknown_values(solution.states[:, m], layout, id) .-
                                   dream[name][:, m]) / maximum(abs, dream[name][:, m])
        for m in steps)
    end
    current = only(unknown_range(layout, unknown_id(layout, :plasma_current)))
    errors["I_p"] = maximum(abs(solution.states[current, m] / dream["I_p"][m] - 1)
    for m in steps)
    f_id, ions_id = unknown_id(layout, :f), unknown_id(layout, :ion_densities)
    errors["f_hot"] = maximum(maximum(abs,
                                  kinetic_view(solution.states[:, m], layout, f_id, ir) .-
                                  dream["f_hot"][:, :, ir, m]) /
                              maximum(abs, dream["f_hot"][:, :, ir, m])
    for ir in 1:run.case.nr, m in steps)
    errors["n_i"] = maximum(maximum(abs,
                                radial_view(solution.states[:, m], layout, ions_id, ir) .-
                                dream["n_i"][ir, :, m]) /
                            maximum(abs, dream["n_i"][ir, :, m])
    for ir in 1:run.case.nr, m in steps)
    return errors
end

@testset "the initialization of the disruption case against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the initialization against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = reference_file("dream_disruption_chain_small.h5")
        dream = Dict(run => h5open(h5 -> read_dream_state(h5[run]), file, "r")
        for run in ("init", "init_rescaled", "ionization"))
        conductivity = h5open(h5 -> read(h5["conductivity/other/fluid/conductivity"]), file, "r")
        init = initialize(dream["init"])
        # run 0: the conductivity of the initial plasma and the field of the prescribed current
        @test maximum(abs, init.field.conductivity ./ conductivity[:, 1] .- 1) < 1e-13
        @test maximum(abs, init.field.electric_field ./ dream["init"]["E_field"][:, 1] .-
                           1) < 1e-13
        for (name, run) in (("init", init.first), ("init_rescaled", init.second))
            @test all(run.solution.statistics.accepted)
            @test run.solution.times == dream[name]["t"]
            # the state at the start of run 1, round-off: the Maxwellian, the moments and the flux
            # from the current. The hot current and the runaways start at zero in DREAM, and run 2
            # starts from the field of run 1, rescaled by its plasma current
            if name == "init"
                errors = differences(run, dream[name], 1:1)
                for quantity in setdiff(sort!(collect(keys(errors))), (
                    "j_hot", "n_re", "j_re"))
                    @test errors[quantity] < 1e-12
                end
            end
            errors = differences(run, dream[name], 2:length(run.solution.times))
            for quantity in sort!(collect(keys(errors)))
                @info "the initialization against DREAM, run $name: $quantity $(errors[quantity])"
                @test errors[quantity] < TOLERANCES[quantity]
            end
        end
        # the hand-over: the state at the start of the ionization phase from the end of run 2 and
        # the ions of the settings, against DREAM's first state of that phase
        start = dream["ionization"]
        case = build(start)
        ours = initial_state_after_initialization(case, init.second.case,
            init.second.solution.states[:, end])
        theirs = initial_state(case, start)
        for unknown in case.layout.unknowns
            entries = _entries(case.layout, unknown.name)
            reference = maximum(abs, theirs[entries])
            difference = maximum(abs, ours[entries] .- theirs[entries])
            @info "the hand-over to the ionization phase: $(unknown.name) $(difference / max(reference, floatmin()))"
            @test difference <= HANDOVER * reference
        end
    end
end
