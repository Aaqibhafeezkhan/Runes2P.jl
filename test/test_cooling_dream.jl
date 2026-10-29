# The cooling-plasma scenario compared with a DREAM run
# (reference/dream/cases/cooling_positivity.py with
# np=200 nxi=5 nt=1000 t_final=10 t_end=1e-3 reltol=1e-10 save=10 variants=centred,tcdf):
# a prescribed temperature that falls from 15 keV to 10 eV in 1 ms, E = 20 E_c(T0), full
# collision mode, f = 0 beyond p_max = 2, 1000 backward-Euler steps of 1e-6 s. The
# coefficients change with time.
#
# Runes2P runs on DREAM's grid, from DREAM's first state and with DREAM's temperature, the
# table of 200 points that DREAM interpolates linearly in time.
using HDF5

const DREAM_COOLING_DATA = reference_file("dream_cooling_positivity.h5")

function cooling_table_value(t_table, values, t)
    t <= t_table[1] && return values[1]
    t >= t_table[end] && return values[end]
    k = searchsortedlast(t_table, t)
    weight = (t - t_table[k]) / (t_table[k + 1] - t_table[k])
    return (1 - weight) * values[k] + weight * values[k + 1]
end

@testset "cooling plasma vs DREAM" begin
    h5open(DREAM_COOLING_DATA) do h5
        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        @test size(g) == (200, 5, 1)
        t_table, T_table = read(h5["temperature/t"]), read(h5["temperature/T"])
        n, field, magnetic = read(h5["n"]), read(h5["E"]), read(h5["B"])
        # the case script evaluates the field with its own constants (1.2e-8 from Runes2P's)
        @test field ≈ 20 * critical_field(n, thermal_coulomb_logarithm(n, T_table[1])) rtol = 1e-7
        plasma_at(t) = PlasmaState(
            cold_electron_temperature = [cooling_table_value(t_table, T_table, t)],
            cold_electron_density = [n], ions = [IonSpecies("D", 1, 1, [n])],
            electric_field = [field], magnetic_field = [magnetic])
        settings = CollisionSettings(mode = :full, ion_size_table = :dream, bremsstrahlung = :neglect)
        nt = Int(read(attributes(h5)["nt"]))
        t_end = read(attributes(h5)["t_end"])
        times = [t_end * k / nt for k in 0:nt]

        # observed: TCDF 5.2e-13 at every saved time. Central 7.1e-11 up to 0.9 ms and 4.3e-9
        # at 1 ms: its values at p_max grow to 1e9 of either sign by then, which amplifies the
        # round-off of the two codes.
        variants = ("tcdf" => (:tcdf, 1e-11, 1e-11), "centred" => (:central, 1e-9, 1e-7))
        for (variant, (scheme, rtol, rtol_last)) in variants
            @testset "$variant" begin
                group = h5[variant]
                f_dream = read(group["f_hot"])          # (np, nξ, nr, saved states)
                saved = read(group["t"])
                @test length(saved) == 11
                save_times = [times[argmin(abs.(times .- t))] for t in saved]
                @test maximum(abs.(save_times .- saved)) <= 1e-12 * t_end

                f0 = f_dream[:, :, :, 1]
                layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = maximum(f0))])
                operator = AdvectionDiffusion(g; scheme, p_upper_boundary = :f_zero)
                system = EquationSystem(layout,
                    [Equation(:f, TimeDerivative(),
                        prescribed_fokker_planck_term(
                            :f, operator, plasma_at; settings, synchrotron = false))])
                solution = integrate(system, vec(f0), times; save_times)
                @test all(solution.statistics.accepted)
                @test size(solution.states, 2) == length(saved)

                errors = map(eachindex(saved)) do m
                    a, b = f_dream[:, :, :, m], reshape(solution.states[:, m], size(f0))
                    cut = 1e-12 * maximum(abs, a)
                    maximum(abs(a[k] - b[k]) / abs(a[k])
                    for k in eachindex(a) if abs(a[k]) > cut)
                end
                @test errors[1] == 0
                @test maximum(errors[1:(end - 1)]) < rtol
                @test errors[end] < rtol_last
                # the same cells are negative in the two codes
                negative_dream = [count(<(0), f_dream[:, :, :, m])
                                  for m in eachindex(saved)]
                negative_runes2p = [count(<(0), solution.states[:, m])
                                    for m in eachindex(saved)]
                @test negative_runes2p == negative_dream
                if variant == "tcdf"
                    @test all(==(0), negative_runes2p)
                else
                    @test negative_runes2p[end] > 100
                end
            end
        end
    end
end
