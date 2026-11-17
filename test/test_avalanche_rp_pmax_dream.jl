# The avalanche of the kinetic Rosenbluth-Putvinski source against the upper momentum
# boundary p_max of the hot-tail grid, against DREAM's AVALANCHE_MODE_KINETIC
# (reference/dream/cases/avalanche_kinetic.py source=rp, DREAM v26.5). The case chooses
# fully ionized deuterium at 100 eV and 1e20 m⁻³, E = 20 E_c, complete screening, the
# constant Coulomb logarithm, centred interpolation, f = 0 beyond p_max, a hot-tail grid only
# with cells of 0.01 below p = 0.6 and of 0.1 above, 14 pitch cells, the cutoff of the
# source at p = 0.13, 384 steps to 160 ms and the start of n_re at 1e10 m⁻³. The start of f,
# a Maxwellian of 1 m⁻³ at 100 keV, is that of examples/avalanche/generateCylindrical.py of
# DREAM's branch boltzmann-eqterm.
#
# The primaries of the source are the electrons of n_re, beyond p_max. Its knock-ons on the
# grid are accelerated and leave through p_max into n_re, and those above p_max enter n_re
# through DREAM's external avalanche, the Rosenbluth-Putvinski rate above p_max matched to
# the fluid growth rate (`ExternalAvalancheTerm`, src/Settings/RunawaySourceTerms.cpp:44-55
# and src/Settings/Equations/n_re.cpp:104-146 of v26.5). The longer the way to p_max, the
# later the knock-ons become primaries, so the growth rate falls as p_max grows. Runes2P
# takes DREAM's first distribution, its time points and the same terms.
#
# DREAM's growth rate of the last window of 20 ms, against its own resolution (rate at the
# changed resolution divided by the rate of the set, minus 1):
#
#   p_max | rate of the set | cells of 0.05 | 28 pitch cells | 56 pitch cells | 768 steps
#   5     | 23.790 s⁻¹      | -4e-5         | -1.3e-2        | -1.5e-2        | -2.5e-3
#   10    | 21.552 s⁻¹      | +5e-5         | -1.4e-2        | -1.6e-2        | -2.3e-3
#   20    | 18.458 s⁻¹      | +1e-4         | -1.6e-2        | -1.8e-2        | -1.9e-3
#
# The ratio of the rates at p_max = 20 and 5 is 0.7759 in the set and 0.7735 with 56 pitch
# cells: with 56 pitch cells the fall of the rate is larger by about 0.2 percentage points.
#
# This is the same method in two codes. test_avalanche_rp.jl checks the rate of the source
# between the cutoff and p_max against the Møller rate, independently of DREAM.
using HDF5

const RP_PMAX_FILES = [
    5 => "dream_avalanche_rp_pmax5.h5", 10 => "dream_avalanche_rp_pmax10.h5",
    20 => "dream_avalanche_rp_pmax20.h5"]

# the run of Runes2P on the grid of DREAM's file, from DREAM's first state and on its time
# points: the runaway density over time, the final distribution and the fluid growth rate
function rp_pmax_run(h5)
    setting(key) = only(read(h5["settings/$key"]))
    temperature, density = read(h5["temperature"]), read(h5["density"])
    field, B = read(h5["E_field"]), read(h5["B"])
    mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
    rad = RadialGrid(read(h5["grid/r_f"]))
    g = cylindrical_grid(mom, rad)
    plasma = PlasmaState(
        cold_electron_temperature = [temperature], cold_electron_density = [density],
        ions = [IonSpecies("D", 1, 1, [density])], electric_field = [field],
        magnetic_field = [B])
    settings = CollisionSettings(mode = :full, screening = :complete,
        coulomb_logarithm = :relativistic, bremsstrahlung = :neglect)
    operator = fokker_planck_operator(
        g, plasma, settings; scheme = :central, synchrotron = false)  # DREAM's centred interpolation
    # the external avalanche reads the temperature and the density as unknowns, so the
    # Fokker-Planck term takes them from the same unknowns, which stay constant
    fp = FokkerPlanckTerm(:f, operator; plasma = PlasmaFromUnknowns(plasma), settings,
        synchrotron = false, lagged = :all)
    p_max = last(mom.p_faces)
    layout = UnknownLayout(rad,
        [KineticUnknown(:f, g; scale = 1e6),
            RadialUnknown(:runaway_density; scale = 1e10),
            RadialUnknown(:total_electron_density; scale = 1e20),
            RadialUnknown(:electric_field), RadialUnknown(:cold_electron_temperature; scale = 100.0),
            RadialUnknown(:cold_electron_density; scale = 1e20)])
    u0 = zeros(length(layout))
    kinetic_view(u0, layout, 1, 1) .= read(h5["f_hot_initial"])[:, :, 1]
    initial = (read(h5["n_re"])[1], density, field, temperature, density)
    for (id, value) in zip(2:6, initial)
        unknown_values(u0, layout, id) .= value
    end
    system = EquationSystem(layout,
        [
            Equation(:f, TimeDerivative(), fp,
                RosenbluthPutvinskiSourceTerm(
                    :f, g; p_cutoff = setting("eqsys/n_re/pCutAvalanche"))),
            # the flux through p_max and the external avalanche at the cutoff p_max
            Equation(:runaway_density, TimeDerivative(), MomentumBoundaryFluxTerm(:f, fp),
                ExternalAvalancheTerm(plasma; p_cutoff = p_max, settings)),
            Equation(:total_electron_density, TimeDerivative()),
            Equation(:electric_field, TimeDerivative()),
            Equation(:cold_electron_temperature, TimeDerivative()),
            Equation(:cold_electron_density, TimeDerivative())];
        initial_state = u0)
    times = read(h5["t"])
    solution = integrate(system, u0, times)
    n_re = [only(unknown_values(solution.states[:, k], layout, 2))
            for k in eachindex(times)]
    f_final = kinetic_view(solution.states[:, end], layout, 1, 1)
    fluid_rate, critical_field, _ = fluid_avalanche_growth_rate(plasma, 1, density; settings)
    return (; times, n_re, f_final, fluid_rate, critical_field)
end

# the growth rate of n_re over each of the windows of the file [s⁻¹]
function window_growth_rates(times, n_re, windows)
    w = (length(times) - 1) ÷ windows
    return [log(n_re[1 + k * w] / n_re[1 + (k - 1) * w]) /
            (times[1 + k * w] - times[1 + (k - 1) * w]) for k in 1:windows]
end

@testset "Rosenbluth-Putvinski avalanche against p_max, against DREAM" begin
    if !all(reference_available(last(entry)) for entry in RP_PMAX_FILES)
        # the set has no entry in Artifacts.toml, and RUNES2P_REFERENCE_DIR does not hold
        # its files
        @info "the set $(reference_set(last(first(RP_PMAX_FILES)))) is not available " *
              "(RUNES2P_REFERENCE_DIR): the comparison is skipped"
        @test_skip all(reference_available(last(entry)) for entry in RP_PMAX_FILES)
    else
        last_rates = Dict{Int, Float64}()
        for (p_max, name) in RP_PMAX_FILES
            @testset "p_max = $p_max" begin
                h5open(reference_file(name)) do h5
                    # the run that the file must hold, checked before the run of Runes2P, so
                    # that a stale or partial file of RUNES2P_REFERENCE_DIR fails here: the
                    # DREAM commit of the set, the arguments of the case, the settings that
                    # the Runes2P side assumes, p_max and the number of time points
                    dream = DREAM_REFERENCE_SETS[reference_set(name)]
                    setting(key) = only(read(h5["settings/$key"]))
                    #! format: off
                    run_checks = [
                        "dream_commit" => read(attributes(h5)["dream_commit"]) == dream.commit,
                        "dream_branch" => read(attributes(h5)["dream_branch"]) == dream.branch,
                        "arguments" => read(attributes(h5)["arguments"]) ==
                            "source=rp pmax=$p_max tmax=0.16 nt=384 windows=8 pcut=0.13 store=test",
                        "avalanche" => setting("eqsys/n_re/avalanche") == 4,             # KINETIC
                        "boundarycondition" => setting("eqsys/f_hot/boundarycondition") == 1, # BC_F_0
                        "collfreq_mode" => setting("collisions/collfreq_mode") == 2,     # FULL
                        "collfreq_type" => setting("collisions/collfreq_type") == 1,     # COMPLETELY_SCREENED
                        "lnlambda" => setting("collisions/lnlambda") == 1,               # CONSTANT
                        "bremsstrahlung_mode" =>
                            setting("collisions/bremsstrahlung_mode") == 1,              # NEGLECT
                        # COLLISIONLESS, the same p⋆ as COLLISIONAL in a cylinder
                        "pstar_mode" => setting("collisions/pstar_mode") == 2,
                        "adv_interp/p1" => setting("eqsys/f_hot/adv_interp/p1") == 1,    # CENTRED
                        "synchrotronmode" => setting("eqsys/f_hot/synchrotronmode") == 1, # neglected
                        "p_max" => isapprox(last(read(h5["grid/p1_f"])), p_max; rtol = 1e-14),
                        "time points" => length(read(h5["t"])) == 385]
                    #! format: on
                    for (label, matches) in run_checks
                        matches || @error "$name does not hold the run of the test" label
                        @test matches
                    end
                    all(last, run_checks) || return
                    run = rp_pmax_run(h5)

                    # the runaway density at every time point: largest difference 1.0e-8 of
                    # DREAM's value (p_max = 5), 8.1e-9 (10) and 5.9e-9 (20). The source of
                    # a cell is 1 + 3.0e-9 times DREAM's, the difference of the constants of
                    # its prefactor (test_avalanche_rp_dream.jl).
                    n_re = read(h5["n_re"])
                    @test maximum(abs.(run.n_re ./ n_re .- 1)) < 3e-8
                    # the growth rate per window of 20 ms: 2.2e-9 to 3.0e-9 of DREAM's
                    dream_rates = read(h5["window_rates"])
                    rates = window_growth_rates(run.times, run.n_re, length(dream_rates))
                    @test rates ≈ dream_rates rtol = 1e-8
                    last_rates[p_max] = last(rates)

                    # the distribution at the end, momentum along the first index
                    a = read(h5["f_hot_final"])[:, :, 1]
                    b = run.f_final
                    faces = read(h5["grid/p1_f"])
                    p = [(faces[i] + faces[i + 1]) / 2 for i in axes(a, 1)]
                    # the knock-ons below the critical momentum: each cell where f is above
                    # 1e-6 of its largest value, compared with DREAM's value of the cell.
                    # These cells lie at p ≤ 0.195 (p_max = 5), 0.185 (10) and 0.175 (20).
                    # Largest difference 1.4e-8 (p_max = 5), 1.2e-8 (10) and 9.5e-9 (20).
                    cut = 1e-6 * maximum(abs, a)
                    #! format: off
                    difference = maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
                    #! format: on
                    @test difference < 5e-8
                    # the runaway tail, the cells above the critical momentum p_c = 0.2294 of
                    # the case: the largest difference divided by the largest |f| of the tail,
                    # 1.3e-8 (p_max = 5), 1.1e-8 (10) and 8.9e-9 (20)
                    tail = p .> read(h5["p_critical"])
                    tail_difference = maximum(abs, a[tail, :] .- b[tail, :]) /
                                      maximum(abs, a[tail, :])
                    @test tail_difference < 5e-8

                    # the fluid growth rate of the match (2.0e-6 of DREAM's) and the effective
                    # critical field (2.1e-5 of DREAM's, which DREAM solves to 2e-3)
                    @test run.fluid_rate ≈ read(h5["fluid/GammaAva"]) rtol = 1e-5
                    @test run.critical_field ≈ read(h5["fluid/Eceff"]) rtol = 1e-4
                end
            end
        end

        @testset "the growth rate falls as p_max grows" begin
            # a file that fails its checks above leaves no rate
            @test issetequal(keys(last_rates), first.(RP_PMAX_FILES))
            if issetequal(keys(last_rates), first.(RP_PMAX_FILES))
                # the last window: 23.790 s⁻¹ at p_max = 5, 21.552 s⁻¹ at 10, 18.458 s⁻¹ at 20
                @test last_rates[5] > last_rates[10] > last_rates[20]
                @test last_rates[5] ≈ 23.790 rtol = 1e-4
                @test last_rates[10] ≈ 21.552 rtol = 1e-4
                @test last_rates[20] ≈ 18.458 rtol = 1e-4
                # the ratio of 0.7759, against 0.7735 with 56 pitch cells (the table above)
                @test 0.77 < last_rates[20] / last_rates[5] < 0.78
            end
        end
    end
end
