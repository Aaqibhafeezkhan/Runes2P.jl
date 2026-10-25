# Regression values for shared-grid results (collision frequencies, Fokker-Planck and
# knock-on applications) that a restructuring of the code must not change. Each case is
# reduced to (sum, sum of absolute values, maximum absolute value). Pure-Julia results are
# compared bit for bit at -O2, the level of the stored values; the `:blas` knock-on results
# use OpenBLAS, whose last bits depend on the CPU and the thread count, so they are compared
# to 1e-13. With `RUNES2P_REGRESSION_RTOL` set, as the Full suite of CI sets it, the
# pure-Julia results are compared to that relative tolerance instead: on a runner of CI the
# last bits can differ from those of the machine that stored the values. A pure-Julia entry
# that differs at -O2 is logged with its relative deviation, also when it passes. If a later
# change legitimately reorders a summation, relax that entry alone and state the reason in a
# comment at it.
using LinearAlgebra: BLAS

# the relative tolerance of the pure-Julia entries at -O2, or `nothing` for bit for bit
const REGRESSION_RTOL = let value = get(ENV, "RUNES2P_REGRESSION_RTOL", "")
    rtol = isempty(value) ? nothing : parse(Float64, value)
    rtol === nothing || (rtol > 0 && isfinite(rtol)) ||
        error("RUNES2P_REGRESSION_RTOL = $value: a relative tolerance is positive and finite")
    rtol
end

function regression_cases()
    ions = [IonSpecies("D", 1, 1, [1e20, 8e19]), IonSpecies("Ar", 18, 2, [1e18, 3e18])]
    plasma = PlasmaState(cold_electron_temperature = [50.0, 800.0], ions = ions,
        electric_field = [3.0, -2.0], magnetic_field = [5.0, 5.0])
    mom = MomentumGrid(biuniform_p_faces(40, 15, 0.0, 0.5, 5.0), uniform_xi_faces(12))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 0.5)))
    cases = Pair{String, Any}[]
    # partial screening at all momenta (`screening_window = :dream`)
    settings = (
        ("full", CollisionSettings(screening_window = :dream, bremsstrahlung = :neglect)),
        ("superthermal",
            CollisionSettings(mode = :superthermal, screening = :none,
                coulomb_logarithm = :energy_dependent, bremsstrahlung = :neglect)),
        ("ultra_relativistic",
            CollisionSettings(mode = :ultra_relativistic, screening = :complete,
                coulomb_logarithm = :relativistic, bremsstrahlung = :neglect)))
    for (name, s) in settings
        model = CollisionModel(plasma, s)
        p = mom.p_faces
        push!(cases, "slowing_down_$name" =>
            [slowing_down_frequency(model, ir, x) for x in p, ir in 1:2])
        push!(cases, "deflection_$name" =>
            [deflection_frequency(model, ir, x) for x in p, ir in 1:2])
        push!(cases,
            "parallel_diffusion_$name" =>
                [parallel_diffusion_frequency(model, ir, x) for x in p, ir in 1:2])
        push!(cases,
            "p3_slowing_down_at_zero_$name" =>
                [p3_slowing_down_frequency_at_zero(model, ir) for ir in 1:2])
        op = fokker_planck_operator(g, plasma, s; synchrotron = true, scheme = :central)  # values stored with it
        f = [1 + 0.5 * sin(1.3i + 0.7j + 2.1ir)
             for i in 1:momentum_cell_count(g), j in 1:pitch_cell_count(g), ir in 1:2]
        df = zeros(size(g))
        apply!(df, op, f)
        push!(cases, "fokker_planck_$name" => df)
    end
    f = [exp(-p / 0.9) * (1.3 + xi) for p in mom.p_centers, xi in mom.xi_centers, _ in 1:2]
    configurations = (("default", (;)),
        ("dream_compatible", (dream_compatible = true, primary_pitch_points = 1)),
        ("gauss", (primary_pitch = :gauss, table_size = 40)))
    for (name, options) in configurations, method in (:loop, :blas)

        op = KnockOnOperator(g, g, [0.1, 0.2]; method, options...)
        C = zeros(size(g))
        apply!(C, op, f, [1e20, 2e20]; scale = 0.5)
        push!(cases, "knockon_$(name)_$method" => C)
    end
    push!(cases, "xi_star" => [xi_star(p, 3.0) for p in 0.1:0.1:1.5])
    push!(cases, "choose_cutoff" => choose_cutoff(mom.p_faces, 0.05, 0.9).p_cutoff)
    return cases
end

const REGRESSION_VALUES = Dict(
    "slowing_down_full" => (1.5026395231942865e6, 1.5026395231942865e6, 977182.00713162),
    "deflection_full" => (5.257889668133548e6, 5.257889668133548e6, 2.0825939425774182e6),
    "parallel_diffusion_full" => (620.9277437401886, 620.9277437401886, 236.2260626948302),
    "p3_slowing_down_at_zero_full" => (0.0, 0.0, 0.0),
    "fokker_planck_full" => (5.672310702116309e7, 6.954715637362007e8, 5.142805130207202e7),
    "slowing_down_superthermal" =>
        (3.4197291605551415e6, 3.4197291605551415e6, 1.6000338883333444e6),
    "deflection_superthermal" =>
        (2.4494305349413004e7, 2.4494305349413004e7, 1.4349452334214503e7),
    "parallel_diffusion_superthermal" => (0.0, 0.0, 0.0),
    "p3_slowing_down_at_zero_superthermal" =>
        (101.12404269720635, 101.12404269720635, 59.13678888654365),
    "fokker_planck_superthermal" =>
        (8.882972198194519e6, 4.0408273305561304e9, 4.38987758035957e8),
    "slowing_down_ultra_relativistic" =>
        (10797.745895272034, 10797.745895272034, 1514.6472082387627),
    "deflection_ultra_relativistic" =>
        (283139.40410529525, 283139.40410529525, 91769.80144034857),
    "parallel_diffusion_ultra_relativistic" => (0.0, 0.0, 0.0),
    "p3_slowing_down_at_zero_ultra_relativistic" =>
        (96.84279033865613, 96.84279033865613, 50.488240274625426),
    "fokker_planck_ultra_relativistic" =>
        (-1.2197565925176533e8, 1.2875744167274424e8, 9.603149091879858e6),
    "knockon_default_loop" => (5.850711083655938e6, 5.850711083655938e6, 320172.895175692),
    "knockon_default_blas" => (5.850711083655938e6, 5.850711083655938e6, 320172.8951756923),
    "knockon_dream_compatible_loop" =>
        (5.850711083655939e6, 5.850711083655939e6, 347515.63968719315),
    "knockon_dream_compatible_blas" =>
        (5.8507110836559385e6, 5.8507110836559385e6, 347515.6396871932),
    "knockon_gauss_loop" => (5.850711083655938e6, 5.850711083655938e6, 320373.1543437839),
    "knockon_gauss_blas" => (5.850711083655938e6, 5.850711083655938e6, 320373.1543437841),
    "xi_star" => (6.823636828072939, 6.823636828072939, 0.7425278006798457),
    "choose_cutoff" => (0.2, 0.2, 0.2)
)

@testset "regression values" begin
    threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        cases = regression_cases()
        @test Set(first.(cases)) == Set(keys(REGRESSION_VALUES))
        optimized = Base.JLOptions().opt_level >= 2
        for (key, value) in cases
            x = collect(Float64, vec(value isa Number ? [value] : value))
            summary = (sum(x), sum(abs, x), maximum(abs, x))
            expected = REGRESSION_VALUES[key]
            @testset "$key" begin
                if endswith(key, "_blas")
                    @test all(isapprox.(summary, expected; rtol = 1e-13))
                elseif !optimized
                    # below -O2 the sums run in another order and agree to round-off
                    @test all(isapprox.(summary, expected; rtol = 1e-12))
                else
                    if !isequal(summary, expected)
                        deviation = abs.(summary .- expected) ./
                                    max.(abs.(expected), floatmin())
                        @info "regression value differs" key summary expected deviation
                    end
                    if REGRESSION_RTOL === nothing
                        @test isequal(summary, expected)
                    else
                        @test all(isapprox.(summary, expected; rtol = REGRESSION_RTOL))
                    end
                end
            end
        end
    finally
        BLAS.set_num_threads(threads)
    end
end
