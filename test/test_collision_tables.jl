# The coefficient tables of the collision frequencies per momentum grid: the factors that
# depend on the momentum and the charge state alone, read by the index of a face or a cell
# centre, give ν_s, ν_∥ and ν_D bit for bit as the direct evaluation, in Float64 and with
# dual numbers, at every face and centre of two grids; the key of the tables holds every
# setting and every parameter of the charge states that the values read; and a model with
# different faces per radius builds one table per distinct faces vector.
using ForwardDiff

# three radii from a cold to a hot bulk, with deuterium and argon spread over its charge states
function _table_plasma()
    temperatures = [2.0, 100.0, 5000.0]
    nr = length(temperatures)
    deuterium = zeros(2, nr)
    deuterium[1, :] .= 1e19
    deuterium[2, :] .= 5e19
    argon = zeros(19, nr)
    for ir in 1:nr, z in 0:18

        argon[z + 1, ir] = 1e17 * exp(-abs(z - 3 - 2ir) / 2)
    end
    return PlasmaState(; cold_electron_temperature = temperatures,
        ions = [IonSpecies("D", 1, deuterium), IonSpecies("Ar", 18, argon)],
        electric_field = zeros(nr), magnetic_field = ones(nr), cold_electron_density = fill(4e19, nr))
end

# the same number, with the same partials for a dual number
function _same_number(a, b)
    return isequal(ForwardDiff.value(a), ForwardDiff.value(b)) &&
           isequal(ForwardDiff.partials(a), ForwardDiff.partials(b))
end
_same_number(a::Float64, b::Float64) = isequal(a, b)

# the count of the values of ν_s, ν_∥ (at the faces) and ν_D (at the centres) of the tabulated
# model that differ from those of the direct model, and the count of the values compared
function _table_differences(tabulated, direct, faces)
    centers = (faces[1:(end - 1)] .+ faces[2:end]) ./ 2
    differences = 0
    compared = 0
    for ir in 1:length(direct.free_electron_density)
        for (k, p) in enumerate(faces)
            entry = Runes2P._face_entry(tabulated, ir, k, p)
            entry === nothing && error("no table entry at face $k")
            nu_s, nu_parallel = Runes2P._slowing_down_and_parallel_diffusion(tabulated, ir, p, entry)
            differences += !_same_number(nu_s, slowing_down_frequency(direct, ir, p))
            differences += !_same_number(nu_parallel, parallel_diffusion_frequency(direct, ir, p))
            compared += 2
        end
        for (i, p) in enumerate(centers)
            entry = Runes2P._center_entry(tabulated, ir, i, p)
            entry === nothing && error("no table entry at centre $i")
            differences += !_same_number(
                Runes2P._deflection_frequency(tabulated, ir, p, entry),
                deflection_frequency(direct, ir, p))
            compared += 1
        end
    end
    return differences, compared
end

@testset "coefficient tables of the collision frequencies" begin
    plasma = _table_plasma()
    nr = 3
    dual = Runes2P.dual_plasma(plasma, true, true)
    # a grid from p = 0 through the screening windows and the quadrature band of the
    # bremsstrahlung integral (0.23 < p < 0.73), and a grid of runaways from p = 0.6
    grids = ("from p = 0" => biuniform_p_faces(40, 15, 0.0, 0.5, 5.0),
        "from p = 0.6" => biuniform_p_faces(30, 10, 0.6, 2.0, 40.0))
    comparison = CollisionSettings(; mode = :full, screening = :partial,
        coulomb_logarithm = :energy_dependent, screening_window = :dream, ion_size_table = :dream,
        bremsstrahlung = :stopping_power, bremsstrahlung_diffusion = :dream,
        bremsstrahlung_constant = :dream)
    variants = ["the comparisons with DREAM" => comparison,
        "the defaults" => CollisionSettings(),
        "superthermal" =>
            CollisionSettings(; mode = :superthermal, coulomb_logarithm = :energy_dependent),
        "ultra-relativistic" =>
            CollisionSettings(; mode = :ultra_relativistic, coulomb_logarithm = :relativistic),
        "a window from 0.5 to 6" =>
            CollisionSettings(; screening_window_start = 0.5, screening_window_end = 6.0),
        "the ionization split" => CollisionSettings(; bound_friction = :ionization_split,
            ionization_threshold = 5.0, ionization_cross_section_form = :dream),
        "free-electron diffusion" => CollisionSettings(; screened_diffusion = :zero),
        "no screening" => CollisionSettings(; screening = :none),
        "complete screening, no bremsstrahlung" =>
            CollisionSettings(; screening = :complete,
                bremsstrahlung = :neglect)]

    @testset "tabulated equals direct: $name, $grid_name" for (name, settings) in variants,
        (grid_name, faces) in grids

        face_vectors = [faces for _ in 1:nr]
        for (label, state) in (("Float64", plasma), ("dual", dual))
            direct = CollisionModel(state, settings)
            tabulated = CollisionModel(state, settings; faces = face_vectors)
            @test all(t === nothing for t in direct.coefficient_tables)
            @test all(t !== nothing for t in tabulated.coefficient_tables)
            differences, compared = _table_differences(tabulated, direct, faces)
            @test compared == nr * (3 * length(faces) - 1)
            @test differences == 0
        end
    end

    @testset "the coefficients of the operator: $name" for (name, settings) in variants[1:2]
        xi = uniform_xi_faces(4)
        radii = RadialGrid(uniform_r_faces(nr, 1.0))
        for (_, faces) in grids
            m = MomentumGrid(faces, xi)
            grid = cylindrical_grid(m, radii)
            for state in (plasma, dual)
                number = eltype(state.cold_electron_temperature)
                op = Runes2P.similar_coefficients(AdvectionDiffusion(grid), number)
                set_fokker_planck_coefficients!(op, state, settings;
                    synchrotron = false, field_in_coefficients = false)
                direct = CollisionModel(state, settings)
                # the operator adds each coefficient to a zeroed array, which turns a -0.0 at p = 0
                # and at ξ = ±1 into 0.0: the expected values take the same sum
                accumulated(x) = zero(x) + x
                differences = 0
                for ir in 1:nr
                    for (k, p) in enumerate(m.p_faces), j in 1:length(m.xi_centers)

                        friction = accumulated(-p * slowing_down_frequency(direct, ir, p))
                        diffusion = accumulated(parallel_diffusion_frequency(direct, ir, p))
                        differences += !_same_number(op.advection_p[k, j, ir], friction)
                        differences += !_same_number(op.diffusion_p_p[k, j, ir], diffusion)
                    end
                    for (i, p) in enumerate(m.p_centers), k in eachindex(m.xi_faces)

                        scattering = accumulated(deflection_frequency(direct, ir, p) *
                                                 (1 - m.xi_faces[k]^2) / 2)
                        differences += !_same_number(op.diffusion_xi_xi[i, k, ir], scattering)
                    end
                end
                @test differences == 0
            end
        end
    end

    @testset "the functions of a momentum evaluate directly" begin
        faces = last(grids[1])
        settings = CollisionSettings()
        tabulated = CollisionModel(plasma, settings; faces = [faces for _ in 1:nr])
        direct = CollisionModel(plasma, settings)
        # between the faces, and at a face
        for p in (0.3337, faces[20])
            @test isequal(slowing_down_frequency(tabulated, 2, p), slowing_down_frequency(direct, 2, p))
            @test isequal(deflection_frequency(tabulated, 2, p), deflection_frequency(direct, 2, p))
            @test isequal(parallel_diffusion_frequency(tabulated, 2, p),
                parallel_diffusion_frequency(direct, 2, p))
        end
        # an index whose face or centre is not the momentum has no entry
        @test Runes2P._face_entry(tabulated, 1, 2, faces[3]) === nothing
        @test Runes2P._face_entry(tabulated, 1, length(faces) + 1, 1.0) === nothing
        @test Runes2P._center_entry(tabulated, 1, 2, faces[2]) === nothing
        @test Runes2P._face_entry(direct, 1, 2, faces[2]) === nothing
        @test Runes2P._face_entry(tabulated, 1, 2, faces[2]) isa Runes2P.TableEntry
    end

    @testset "the key of the tables" begin
        faces = last(grids[1])
        face_vectors = [faces for _ in 1:nr]
        table(settings, state = plasma) = CollisionModel(state, settings; faces = face_vectors).coefficient_tables[1]
        base = CollisionSettings()
        reference = table(base)
        # built once per grid and key, also for a copy of the faces and a dual plasma
        @test table(base) === reference
        @test CollisionModel(plasma, base; faces = [copy(faces) for _ in 1:nr]).coefficient_tables[1] ===
              reference
        @test table(base, dual) === reference
        # each setting of the key alone gives a new table
        changes = (
            mode = :superthermal, screening_window = :dream, screening_window_start = 0.5,
            screening_window_end = 6.0, ion_size_table = :dream, bremsstrahlung_constant = :dream)
        for (setting, value) in pairs(changes)
            changed = table(CollisionSettings(; NamedTuple{(setting,)}((value,))...))
            @test changed !== reference
        end
        key = Runes2P._coefficient_table_key(base, CollisionModel(plasma, base).charge_states)
        @test keys(key) ==
              (:mode, :screening_window, :screening_window_start, :screening_window_end,
            :ion_size_table, :bremsstrahlung_constant, :charge_states)
        # a setting the tables do not read shares them
        @test table(CollisionSettings(; coulomb_logarithm = :energy_dependent)) ===
              reference
        @test table(CollisionSettings(; bremsstrahlung = :neglect)) === reference
        # other charge states give a new table: another element, and each parameter of a charge
        # state alone in the key
        neon = PlasmaState(; cold_electron_temperature = plasma.cold_electron_temperature,
            ions = [plasma.ions[1],
                IonSpecies("Ne", 10, plasma.ions[2].charge_state_densities[1:11, :])],
            electric_field = zeros(nr), magnetic_field = ones(nr), cold_electron_density = fill(4e19, nr))
        @test table(base, neon) !== reference
        states = CollisionModel(plasma, base).charge_states
        function changed_state(state, name, value)
            fields = [field === name ? value : getfield(state, field)
                      for field in fieldnames(Runes2P.ChargeState)]
            return Runes2P.ChargeState(fields...)
        end
        argon = findfirst(s -> s.nuclear_charge == 18 && s.charge == 2, states)
        for (name, value) in ((:element, "Ne"), (:nuclear_charge, 10), (:charge, 3),
            (:mean_excitation_energy, 2 * states[argon].mean_excitation_energy),
            (:effective_size, 2 * states[argon].effective_size))
            other = copy(states)
            other[argon] = changed_state(states[argon], name, value)
            @test !isequal(Runes2P._coefficient_table_key(base, other), key)
        end
    end

    @testset "one table per distinct faces vector of per-radius grids" begin
        lower = last(grids[1])
        upper = last(grids[2])
        settings = CollisionSettings()
        model = CollisionModel(plasma, settings; faces = [lower, upper, copy(lower)])
        tables = model.coefficient_tables
        @test tables[1] === tables[3]
        @test tables[1] !== tables[2]
        @test tables[1].faces == lower && tables[2].faces == upper
        direct = CollisionModel(plasma, settings)
        for (ir, faces) in enumerate((lower, upper, lower))
            single = CollisionModel(plasma, settings; faces = [faces for _ in 1:nr])
            @test single.coefficient_tables[ir] === tables[ir]
        end
        differences = 0
        for (ir, faces) in enumerate((lower, upper, lower)), (k, p) in enumerate(faces)

            entry = Runes2P._face_entry(model, ir, k, p)
            nu_s = first(Runes2P._slowing_down_and_parallel_diffusion(model, ir, p, entry))
            differences += !isequal(nu_s, slowing_down_frequency(direct, ir, p))
        end
        @test differences == 0
        @test_throws DimensionMismatch CollisionModel(plasma, settings; faces = [
            lower, upper])
    end
end
