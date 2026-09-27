# Records of a run's settings: the describe
# methods, the run record of integrate, its hashes, the code version and the settings
# group of the output file. The lists of names below are the documented ones, so a change
# of the file's content fails here until its documentation and the format version follow.
using HDF5
using InteractiveUtils: subtypes

# a term defined outside the package, without a describe method
struct UndescribedDecay <: EquationTerm
    unknown::Symbol
end
Runes2P.dependencies(term::UndescribedDecay) = [term.unknown]
Runes2P.is_linear(::UndescribedDecay) = true
function Runes2P.rhs!(out, term::UndescribedDecay, u, layout, row, t)
    for ir in 1:radial_cell_count(layout)
        range = unknown_range(layout, row, ir)
        @views out[range] .-= u[range]
    end
    return out
end
function Runes2P.sparsity!(entries, term::UndescribedDecay, layout, row)
    for ir in 1:radial_cell_count(layout), i in unknown_range(layout, row, ir)
        push!(entries, (i, i))
    end
    return entries
end

const DOCUMENTED_ROOT_ATTRIBUTES = ["runes2p_commit", "runes2p_tree_clean", "runes2p_version", "format_version"]
const DOCUMENTED_ROOT_GROUPS = ["layout", "settings", "solver_statistics", "time", "unknowns", "update_states"]
const DOCUMENTED_SETTINGS = ["checkpoint", "code", "complete", "environment", "grids", "hashes", "layout",
                             "log", "prescribed", "run", "segment", "setup_hash", "solver", "stop", "system",
                             "time", "updates"]
const DOCUMENTED_STATISTICS = ["accepted", "accepted_as_linear", "accepted_at_residual_floor",
                               "convergence_order", "factorization_time", "initial_residual_norm",
                               "iteration_attempt", "iteration_residual_norm", "iteration_step_length",
                               "iteration_update_norm", "jacobian_time", "krylov_iterations",
                               "linear_solve_time", "newton_iterations", "preconditioner_updates",
                               "residual_norm", "residual_time", "smallest_step_length", "status", "time",
                               "time_step", "update_norm", "wall_time"]
const DOCUMENTED_CODE = ["branch", "commit", "commit_known", "julia_version", "package", "package_version",
                         "tree_clean"]

_settings_plasma(temperature; density=1e20, field=1.0, nr=2) =
    PlasmaState(cold_electron_temperature=fill(temperature, nr), cold_electron_density=fill(density, nr),
                ions=[IonSpecies("D", 1, 1, fill(density, nr))], electric_field=fill(field, nr),
                magnetic_field=fill(5.0, nr))

# a cooling run with a prescribed plasma, radial diffusion set by hand and a knock-on term
function _settings_case(; scheme=:exponential_fitting, diffusion=1.0, final_temperature=100.0,
                        treatment=:explicit, cutoff=0.4)
    nr = 2
    mom = MomentumGrid(uniform_p_faces(12, 0.0, 1.2), uniform_xi_faces(4))
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 1.0)))
    plasma_at(t) = _settings_plasma(1000.0 * (final_temperature / 1000.0)^(t / 3e-4); nr)
    op = AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero)
    radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
    radial.diffusion_r_r .= diffusion
    knockon = KnockOnOperator(grid, grid, cutoff)
    terms = (PrescribedFokkerPlanckTerm(:f, op, plasma_at; settings=CollisionSettings(mode=:full)),
             FokkerPlanckTerm(:f, radial),
             KnockOnTerm(:f, knockon; target_density=fill(1e20, nr), treatment))
    layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=1e20)])
    system = EquationSystem(layout, [Equation(:f, TimeDerivative(), terms...)])
    theta = 1000.0 / Constants.ELECTRON_REST_ENERGY
    f0 = [1e20 * exp(-(sqrt(1 + p^2) - 1) / theta) for p in mom.p_centers, _ in 1:4, _ in 1:nr]
    times = [0.0, 1e-4, 2e-4, 3e-4]
    return (; grid, mom, system, layout, u0=vec(f0), times, plasma_at, radial, knockon)
end

_is_plain(x::Union{Bool,Int,Float64,String}) = true
_is_plain(x::Array{<:Union{Int,Float64}}) = !isempty(x)
_is_plain(x::Vector{String}) = !isempty(x)
_is_plain(x::Dict{String,Any}) = all(_is_plain, values(x))
_is_plain(x) = false

@testset "settings records" begin
    @testset "every field of a described type is classified" begin
        types = [MomentumGrid, RadialGrid, KineticGrid, UnknownLayout, AdvectionDiffusion, KnockOnOperator,
                 FokkerPlanckTerm, PrescribedFokkerPlanckTerm, KnockOnTerm, LinearTerm, DensityMomentTerm,
                 CurrentMomentTerm, MaxwellianSourceTerm, EquationSystem, Tolerance, CollisionSettings,
                 IonSpecies, PlasmaState, NewtonSolver, GMRES, SparseLU, RadiusBlockLU, RadiusBlockILU]
        for T in types
            fields = settings_fields(T)
            @test isempty(intersect(fields.described, fields.derived))
            @test Set(union(fields.described, fields.derived)) == Set(fieldnames(T))
        end
        # the terms and solvers of the package are all in the list above
        package_types(S) = [T for T in subtypes(S) if parentmodule(T) === Runes2P]
        @test issubset(package_types(EquationTerm), types)
        @test issubset(package_types(LinearSolver), types)
        @test issubset(package_types(Runes2P.BlockPreconditioner), types)
        # and have a record of their own, not the fallback
        fallback = which(describe, Tuple{EquationTerm,DescriptionContext})
        for T in package_types(EquationTerm)
            @test which(describe, Tuple{T,DescriptionContext}) != fallback
        end
    end

    @testset "grids and layout" begin
        case = _settings_case()
        record = describe(case.grid)
        @test record["type"] == "KineticGrid"
        @test record["geometry"] == "cylindrical"
        @test record["shared_momentum_grid"] === true
        @test record["momentum_grid"]["p_faces"] == case.mom.p_faces
        @test record["momentum_grid"]["xi_faces"] == case.mom.xi_faces
        @test record["r_faces"] == case.grid.radial_grid.r_faces
        @test _is_plain(record)

        other = MomentumGrid(uniform_p_faces(5, 0.0, 1.2), uniform_xi_faces(4))
        ragged = cylindrical_grid([case.mom, other], case.grid.radial_grid)
        record = describe(ragged)
        @test record["shared_momentum_grid"] === false
        @test record["momentum_grids"]["radius_2"]["p_faces"] == other.p_faces

        # grids with equal faces share a name, others get the next one
        context = DescriptionContext()
        same = cylindrical_grid(MomentumGrid(copy(case.mom.p_faces), copy(case.mom.xi_faces)),
                                case.grid.radial_grid)
        @test Runes2P.grid_name!(context, case.grid) == "grid_1"
        @test Runes2P.grid_name!(context, same) == "grid_1"
        @test Runes2P.grid_name!(context, ragged) == "grid_2"
        @test sort(collect(keys(describe(context)))) == ["grid_1", "grid_2"]

        layout = UnknownLayout(case.grid.radial_grid,
                               [ScalarUnknown(:plasma_current; scale=2.0),
                                RadialUnknown(:total_electron_density),
                                RadialUnknown(:ion_densities; components=3,
                                              nonnegative=LowerLimit(tolerance=1e-12,
                                                                     reference=:total_electron_density)),
                                KineticUnknown(:f, case.grid; scale=1e20)]; scalars=:first)
        record = describe(layout, DescriptionContext())
        @test record["order"] == ["plasma_current", "total_electron_density", "ion_densities", "f"]
        @test record["scalars"] == "first"
        @test record["unknown_count"] == length(layout)
        @test record["unknowns"]["f"]["grid"] == "grid_1"
        @test record["unknowns"]["f"]["kind"] == "kinetic"
        @test record["unknowns"]["f"]["scale"] == 1e20
        @test record["unknowns"]["ion_densities"]["components"] == 3
        @test record["unknowns"]["ion_densities"]["lower_limit_tolerance"] == 1e-12
        @test record["unknowns"]["ion_densities"]["lower_limit_reference"] == "total_electron_density"
        @test record["unknowns"]["plasma_current"]["sizes"] == [1]
        @test !haskey(record["unknowns"]["plasma_current"], "lower_limit_tolerance")
        @test _is_plain(record)
    end

    @testset "plasma, collisions and solvers" begin
        plasma = _settings_plasma(300.0)
        record = describe(plasma)
        @test record["cold_electron_temperature"] == [300.0, 300.0]
        @test record["ions"]["species_1"]["name"] == "D"
        @test record["ions"]["species_1"]["nuclear_charge"] == 1
        @test record["ions"]["species_1"]["charge_state_densities"] == [0.0 0.0; 1e20 1e20]
        settings = CollisionSettings(mode=:superthermal, coulomb_logarithm=:energy_dependent)
        record = describe(settings)
        @test record == Dict{String,Any}("type" => "CollisionSettings", "mode" => "superthermal",
                                         "screening" => "partial", "coulomb_logarithm" => "energy_dependent",
                                         "screened_diffusion" => "maxwellian", "ion_size_table" => "hesslow")

        solver = NewtonSolver(linear_solver=GMRES(preconditioner=RadiusBlockILU(drop_tolerance=1e-3),
                                                  restart_length=60, preconditioner_refresh=:every_step),
                              residual_tolerance=1e-7, line_search=true)
        record = describe(solver)
        @test record["type"] == "NewtonSolver"
        @test record["residual_tolerance"] == 1e-7
        @test record["line_search"] === true
        @test record["max_iterations"] === 30
        @test record["reuse_linear_matrix"] === true
        @test record["linear_solver"]["type"] == "GMRES"
        @test record["linear_solver"]["restart_length"] == 60
        @test record["linear_solver"]["preconditioner_refresh"] == "every_step"
        @test record["linear_solver"]["operator"] == "exact"
        @test record["linear_solver"]["preconditioner"] ==
              Dict{String,Any}("type" => "RadiusBlockILU", "drop_tolerance" => 1e-3)
        @test describe(NewtonSolver())["linear_solver"] == Dict{String,Any}("type" => "SparseLU")
        @test _is_plain(record)
    end

    @testset "operators" begin
        case = _settings_case()
        record = describe(case.radial)
        @test record["r_boundary"] == "absorbing"
        @test record["p_scheme"] == "central"
        @test record["p_upper_boundary"] == "outflow"
        @test collect(keys(record["coefficients"])) == ["diffusion_r_r"]
        @test record["coefficients"]["diffusion_r_r"] == case.radial.diffusion_r_r
        @test "advection_p" in record["zero_coefficients"]
        @test length(record["zero_coefficients"]) == 8
        @test !haskey(record, "coefficient_source")
        @test !haskey(record, "p_flux_at_lower_boundary")

        # coefficients built from a plasma name their source, and with_scheme shares it
        plasma = _settings_plasma(300.0)
        settings = CollisionSettings(mode=:full)
        op = fokker_planck_operator(case.grid, plasma, settings; synchrotron=true)
        record = describe(op)
        @test record["coefficient_source"]["plasma"] == describe(plasma)
        @test record["coefficient_source"]["collision_settings"] == describe(settings)
        @test record["coefficient_source"]["synchrotron"] === true
        @test record["coefficients"]["advection_p"] == op.advection_p
        @test describe(with_scheme(op, :upwind))["coefficient_source"] == record["coefficient_source"]
        @test describe(with_scheme(op, :upwind))["p_scheme"] == "upwind"
        zero_coefficients!(op)
        @test !haskey(describe(op), "coefficient_source")
        @test !haskey(describe(op; coefficients=false), "coefficients")

        lower = MomentumGrid(uniform_p_faces(6, 0.2, 1.2), uniform_xi_faces(4))
        op = AdvectionDiffusion(cylindrical_grid(lower, case.grid.radial_grid); p_lower_boundary=:prescribed)
        op.p_flux_at_lower_boundary .= 2.0
        @test describe(op; coefficients=false)["p_flux_at_lower_boundary"] == fill(2.0, 4, 2)

        record = describe(case.knockon)
        @test record["knockon"] == record["primary"] == "grid_1"
        @test record["p_cutoff"] == [0.4, 0.4]
        @test record["method"] == "blas"
        options = record["pitch_options"]
        @test options["dream_compatible"] === false
        @test options["xi_star_points"] == 4
        # the defaults filled in are the ones the kernels were built with
        kernel = pitch_kernel(case.knockon, 1)
        @test options["xi_star_mode"] == String(kernel.xi_star_mode)
        @test options["table_size"] == length(kernel.xi_star_values)
        dream = KnockOnOperator(case.grid, case.grid, 0.4; dream_compatible=true, table_size=40)
        options = describe(dream)["pitch_options"]
        @test options["xi_star_mode"] == String(pitch_kernel(dream, 1).xi_star_mode) == "centre"
        @test options["primary_pitch"] == "centre"
        @test options["table_size"] == length(pitch_kernel(dream, 1).xi_star_values) == 40
    end

    @testset "terms and the system" begin
        case = _settings_case()
        context = DescriptionContext(times=case.times)
        record = describe(case.system, context)
        equation = record["equations"]["f"]
        @test equation["kind"] == "time_derivative"
        @test equation["term_count"] == 3
        prescribed, radial, knockon = (equation["terms"]["term_$k"] for k in 1:3)
        @test prescribed["type"] == "PrescribedFokkerPlanckTerm"
        @test prescribed["unknown"] == "f"
        @test prescribed["jacobian"] == "exact"
        @test prescribed["treatment"] == "implicit"
        @test prescribed["coupling"] == "sparse"
        @test prescribed["linear"] === true
        @test prescribed["dependencies"] == ["f"]
        @test prescribed["operator"]["p_scheme"] == "exponential_fitting"
        @test !haskey(prescribed["operator"], "coefficients")
        @test prescribed["collision_settings"]["mode"] == "full"
        @test prescribed["synchrotron"] === false
        @test prescribed["plasma"] == "plasma_at"
        @test prescribed["plasma_table"] == "table_1"
        @test collect(keys(Runes2P.describe_tables(context))) == ["table_1"]
        plasma = Runes2P.describe_tables(context)["table_1"]
        @test plasma["source"] == "plasma_at"
        @test plasma["thinned"] === false
        @test plasma["times"] == case.times
        @test plasma["time_count"] == 4
        @test size(plasma["cold_electron_temperature"]) == (2, 4)
        for (k, t) in enumerate(case.times)
            state = case.plasma_at(t)
            @test plasma["cold_electron_temperature"][:, k] == state.cold_electron_temperature
            @test plasma["electric_field"][:, k] == state.electric_field
            @test plasma["ions"]["species_1"]["charge_state_densities"][:, :, k] ==
                  state.ions[1].charge_state_densities
        end
        @test !haskey(plasma, "described")
        @test radial["type"] == "FokkerPlanckTerm"
        @test radial["operator"]["coefficients"]["diffusion_r_r"] == case.radial.diffusion_r_r
        @test knockon["type"] == "KnockOnTerm"
        @test knockon["treatment"] == "explicit"
        @test knockon["coupling"] == "matrix_free"
        @test knockon["primary"] == "f"
        @test knockon["target_density"] == [1e20, 1e20]
        @test knockon["operator"]["p_cutoff"] == [0.4, 0.4]
        @test record["time_scale"] == 1.0
        @test record["newton_tolerances"]["f"] == Dict{String,Any}("absolute" => 1e-10 * 1e20, "relative" => 1e-10)
        @test _is_plain(record)

        # the record of the system does not depend on the time points, the table does
        @test describe(case.system, DescriptionContext(times=[case.times; 4e-4])) == record
        @test describe(case.system) == record
        # without time points the prescribed plasma is not described
        plasma = describe_prescribed_plasma(case.plasma_at, DescriptionContext())
        @test plasma["described"] === false
        @test !haskey(plasma, "times")
        # a table beyond the size limit is thinned: one state of this plasma is 12 numbers
        plasma = describe_prescribed_plasma(case.plasma_at, DescriptionContext(times=range(0, 3e-4; length=11),
                                                                               max_table_bytes=3 * 12 * 8))
        @test plasma["thinned"] === true
        @test !haskey(plasma, "described")
        @test plasma["times"] ≈ [0.0, 1.5e-4, 3e-4] rtol = 1e-14
        @test plasma["time_count"] == 11
        @test size(plasma["cold_electron_density"]) == (2, 3)
        failing(t) = t > 1e-4 ? error("outside the table") : case.plasma_at(t)
        plasma = describe_prescribed_plasma(failing, DescriptionContext(times=case.times))
        @test plasma["described"] === false
        @test occursin("outside the table", plasma["reason"])

        # the other terms
        context = DescriptionContext()
        @test describe(LinearTerm(:f; coefficient=-2.0), context) ==
              Dict{String,Any}("type" => "LinearTerm", "treatment" => "implicit", "coupling" => "sparse",
                               "linear" => true, "jacobian_kind" => "exact", "dependencies" => ["f"],
                               "unknown" => "f", "coefficient" => -2.0)
        @test describe(DensityMomentTerm(:f; coefficient=3.0), context)["kinetic"] == "f"
        record = describe(CurrentMomentTerm(:f, case.grid; pitch_rule=:cell_center), context)
        @test record["pitch_rule"] == "cell_center"
        @test record["grid"] == "grid_1"
        record = describe(MaxwellianSourceTerm(:f, case.grid, 250.0; target_density=[1e20, 2e20], rate=1e5,
                                               bulk_momentum=0.3), context)
        @test record["temperature"] == 250.0
        @test record["bulk_momentum"] == 0.3
        @test record["target_density"] == [1e20, 2e20]
        @test record["rate"] == 1e5
        @test record["treatment"] == "explicit"
        limiter = fokker_planck_operator(case.grid, _settings_plasma(300.0); scheme=:van_albada)
        record = describe(FokkerPlanckTerm(:f, limiter; jacobian=:upwind), context)
        @test record["jacobian"] == "upwind"
        @test record["jacobian_kind"] == "approximate"
        @test record["linear"] === false
    end

    @testset "hashes" begin
        record = Dict{String,Any}("a" => 1, "b" => [1.0, 2.0], "c" => Dict{String,Any}("d" => "text", "e" => true))
        reordered = Dict{String,Any}("c" => Dict{String,Any}("e" => true, "d" => "text"), "b" => [1.0, 2.0], "a" => 1)
        hash = settings_hash(record)
        @test length(hash) == 64 && all(c -> c in "0123456789abcdef", hash)
        @test settings_hash(reordered) == hash
        changed(key, value) = settings_hash(merge(record, Dict{String,Any}(key => value)))
        @test changed("a", 2) != hash
        @test changed("a", 1.0) != hash                  # the type counts
        @test changed("b", [1.0, 2.0 + 4eps()]) != hash
        @test changed("b", [1.0 2.0]) != hash            # the shape counts
        @test changed("b", [1, 2]) != hash
        @test changed("c", Dict{String,Any}("d" => "text", "e" => false)) != hash
        @test changed("c", Dict{String,Any}("d" => "text", "e" => 1)) != hash
        @test changed("f", ["x", "y"]) != changed("f", ["xy"])
        @test settings_hash(Dict{String,Any}("ab" => "c")) != settings_hash(Dict{String,Any}("a" => "bc"))
    end

    @testset "the record of a run" begin
        case = _settings_case()
        solution = integrate(case.system, case.u0, case.times; save_times=[0.0, 3e-4],
                             description=Dict("name" => "cooling test", "delta_b_over_b" => 1e-3,
                                              :initial_state => "Maxwellian at 1 keV"))
        settings = solution.settings
        @test sort(collect(keys(settings))) == DOCUMENTED_SETTINGS
        @test _is_plain(settings)
        @test settings["complete"] === true
        @test settings["time"] == Dict{String,Any}("time_count" => 4, "save_count" => 2, "max_step_halvings" => 10,
                                                   "times" => case.times, "save_times" => [0.0, 3e-4])
        @test settings["run"] == Dict{String,Any}("name" => "cooling test", "delta_b_over_b" => 1e-3,
                                                  "initial_state" => "Maxwellian at 1 keV")
        @test collect(keys(settings["grids"])) == ["grid_1"]
        @test settings["grids"]["grid_1"] == describe(case.grid)
        @test settings["solver"] == describe(NewtonSolver())
        @test sort(collect(keys(settings["hashes"]))) == sort(collect(Runes2P.SETUP_SECTIONS))
        for section in Runes2P.SETUP_SECTIONS
            @test settings["hashes"][section] == settings_hash(settings[section])
        end
        @test settings["setup_hash"] ==
              settings_hash(Dict{String,Any}(section => settings[section] for section in Runes2P.SETUP_SECTIONS))

        # the record is taken before the first step and equals run_settings of the setup
        again = run_settings(case.system; times=case.times, save_times=[0.0, 3e-4],
                             description=settings["run"])
        @test again["setup_hash"] == settings["setup_hash"]
        @test sort(collect(keys(again))) == setdiff(DOCUMENTED_SETTINGS, ["segment"])   # integrate adds it
        @test again["system"] == settings["system"]

        # the same setup built again has the same hash, and a change shows in its section
        base = run_settings(_settings_case().system; times=case.times)
        sections_changed(other) = sort([s for s in Runes2P.SETUP_SECTIONS if other["hashes"][s] != base["hashes"][s]])
        same = run_settings(_settings_case().system; times=case.times)
        @test same["setup_hash"] == base["setup_hash"]
        @test isempty(sections_changed(same))
        for changes in ((; scheme=:upwind), (; diffusion=1.0 + 1e-12), (; treatment=:implicit), (; cutoff=0.5))
            other = run_settings(_settings_case(; changes...).system; times=case.times)
            @test other["setup_hash"] != base["setup_hash"]
            @test sections_changed(other) == ["system"]
        end
        other = run_settings(_settings_case(; final_temperature=99.0).system; times=case.times)
        @test sections_changed(other) == ["prescribed"]
        other = run_settings(case.system; times=case.times, solver=NewtonSolver(max_iterations=31))
        @test sections_changed(other) == ["solver"]
        other = run_settings(case.system; times=[case.times; 4e-4])
        @test sections_changed(other) == ["prescribed", "time"]   # the plasma is sampled at the time points
        other = run_settings(case.system; times=case.times, max_step_halvings=3)
        @test sections_changed(other) == ["time"]
        other = run_settings(case.system; times=case.times, description=Dict("name" => "other"))
        @test sections_changed(other) == ["run"]

        # values a record cannot hold are refused before the run starts
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; description=Dict("f" => sin))
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; description=Dict("a/b" => 1))
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; description=Dict("a" => Float64[]))

        # a term without a record marks the run's record as incomplete
        layout = UnknownLayout(case.grid.radial_grid, [RadialUnknown(:n)])
        system = EquationSystem(layout, [Equation(:n, TimeDerivative(), UndescribedDecay(:n))])
        solution = integrate(system, [1.0, 2.0], [0.0, 0.1])
        @test solution.states[:, end] ≈ [1.0, 2.0] ./ 1.1
        @test solution.settings["complete"] === false
        term = solution.settings["system"]["equations"]["n"]["terms"]["term_1"]
        @test term["type"] == "UndescribedDecay"
        @test term["described"] === false
        @test term["linear"] === true
        @test isempty(solution.settings["grids"])

        # a solution built by hand has no record
        by_hand = TimeSolution(solution.layout, solution.times, solution.states, solution.statistics)
        @test isempty(by_hand.settings)
    end

    @testset "code version and environment" begin
        code = code_version()
        @test sort(collect(keys(code))) == DOCUMENTED_CODE
        @test code["package"] == "Runes2P"
        @test code["package_version"] == string(pkgversion(Runes2P))
        @test code["julia_version"] == string(VERSION)
        @test code["tree_clean"] isa Bool
        # the tests run in a checkout of the repository
        @test code["commit_known"] === true
        @test occursin(r"^[0-9a-f]{40}$", code["commit"])
        @test code == code_version()
        code["commit"] = "changed"           # a copy, the cached record is unchanged
        @test code_version()["commit"] != "changed"

        environment = run_environment()
        @test environment["hostname"] == gethostname()
        @test environment["cpu_model"] == String(first(Sys.cpu_info()).model)
        # the record of a run does not ask the machine again
        run_environment()
        @test (@elapsed run_environment()) < 0.05
        @test environment["julia_threads"] == Threads.nthreads()
        @test environment["cpu_threads"] == Sys.CPU_THREADS
        @test occursin(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}$", environment["start_time"])
        @test _is_plain(environment)
        @test OUTPUT_FORMAT_VERSION == v"0.6"
        @test Runes2P.output_format_version() == "0.6"
    end

    @testset "output file" begin
        case = _settings_case()
        solver = NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:every_step), line_search=true)
        solution = integrate(case.system, case.u0, case.times; solver,
                             description=Dict("name" => "cooling test", "levels" => [1, 2, 3]))
        file = tempname() * ".h5"
        write_output(file, solution)
        h5open(file) do h5
            @test sort(collect(keys(attributes(h5)))) == DOCUMENTED_ROOT_ATTRIBUTES
            @test read(attributes(h5)["format_version"]) == "0.6"
            @test VersionNumber(read(attributes(h5)["format_version"])) == OUTPUT_FORMAT_VERSION
            @test read(attributes(h5)["runes2p_commit"]) == code_version()["commit"]
            @test read(attributes(h5)["runes2p_version"]) == string(pkgversion(Runes2P))
            @test read(attributes(h5)["runes2p_tree_clean"]) == UInt8(code_version()["tree_clean"])
            @test sort(collect(keys(h5))) == DOCUMENTED_ROOT_GROUPS
            @test sort(collect(keys(h5["settings"]))) == DOCUMENTED_SETTINGS
            @test sort(collect(keys(h5["settings/code"]))) == DOCUMENTED_CODE
            @test sort(collect(keys(h5["solver_statistics"]))) == DOCUMENTED_STATISTICS
            @test DOCUMENTED_STATISTICS == sort([String(name) for name in fieldnames(SolverStatistics)])
            @test read(h5["settings/system/equations/f/terms/term_1/operator/p_scheme"]) == "exponential_fitting"
            @test read(h5["settings/system/equations/f/terms/term_1/plasma_table"]) == "table_1"
            @test read(h5["settings/prescribed/table_1/times"]) == case.times
            @test size(read(h5["settings/prescribed/table_1/cold_electron_temperature"])) == (2, 4)
            @test read(h5["settings/grids/grid_1/momentum_grid/p_faces"]) == case.mom.p_faces
            @test read(h5["settings/solver/line_search"]) === 0x01
            @test read(h5["settings/code/commit"]) == code_version()["commit"]
        end
        # read back, the record is the one written, type for type, so the hashes hold
        settings = read_settings(file)
        @test settings == solution.settings
        @test settings["solver"]["line_search"] === true
        @test settings["run"]["levels"] == [1, 2, 3]
        @test settings["time"]["time_count"] === 4
        @test settings["setup_hash"] ==
              settings_hash(Dict{String,Any}(section => settings[section] for section in Runes2P.SETUP_SECTIONS))
        for section in Runes2P.SETUP_SECTIONS
            @test settings_hash(settings[section]) == solution.settings["hashes"][section]
        end
        rm(file)

        # a solution without a record is written without the settings group
        by_hand = TimeSolution(solution.layout, solution.times, solution.states, solution.statistics)
        write_output(file, by_hand)
        h5open(file) do h5
            @test !haskey(h5, "settings")
            @test read(attributes(h5)["format_version"]) == "0.6"
            @test read(attributes(h5)["runes2p_commit"]) == code_version()["commit"]
        end
        @test isempty(read_settings(file))
        rm(file)
    end
end
