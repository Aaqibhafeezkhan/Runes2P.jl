# The Rosenbluth-Putvinski avalanche source of a hot-tail grid: its rate against the Møller
# rate of an infinitely fast primary of the knock-on code, the number of electrons it
# creates on the grid, the pitch of the knock-ons and its Jacobian against central
# differences; the energy of the knock-ons against its sum over the cells and the closed form
# of the Møller rate.
using SparseArrays: findnz

@testset "Rosenbluth-Putvinski source" begin
    @testset "the rate is the Møller rate of a primary at the speed of light" begin
        # independent of DREAM's closed form: the knock-on code integrates the Møller cross
        # section, which for γ₁ → ∞ is 2π r₀² /(γ − 1)² per unit γ (`moller_total_rate`)
        for p in (0.01, 0.1, 0.8, 3.0)
            @test rosenbluth_putvinski_rate(p) ≈ moller_total_rate(Inf, p) rtol = 1e-10
        end
        @test rosenbluth_putvinski_rate(0.01, 0.8) ≈
              rosenbluth_putvinski_rate(0.01) - rosenbluth_putvinski_rate(0.8) rtol = 1e-14
        @test isinf(rosenbluth_putvinski_rate(0.0))
    end

    # the grid of DREAM's disruption example, small: biuniform to p = 0.8, two radii
    radial = RadialGrid(uniform_r_faces(2, 0.5))
    grid = cylindrical_grid(
        MomentumGrid(biuniform_p_faces(40, 15, 0.0, 0.05, 0.8), uniform_xi_faces(15)),
        radial)
    layout = UnknownLayout(radial,
        [RadialUnknown(:runaway_density; scale = 1e16),
            RadialUnknown(:total_electron_density; scale = 1e20),
            RadialUnknown(:electric_field), KineticUnknown(:f, grid; scale = 1e19)])
    term = RosenbluthPutvinskiSourceTerm(:f, grid)
    u = zeros(length(layout))
    unknown_values(u, layout, 1) .= [3e15, 1e16]
    unknown_values(u, layout, 2) .= [1.5e20, 1.2e20]
    unknown_values(u, layout, 3) .= [2.0, -1.5]
    f_id = unknown_id(layout, :f)
    source = Runes2P.rhs!(zeros(length(u)), term, u, layout, f_id, 0.0)

    @testset "the electrons it creates" begin
        # the density moment of the source is n_re n_tot times the rate between the cutoff
        # and p_max, exactly: the cell parts telescope (DREAM's internal avalanche)
        p_max = last(momentum_grid(grid, 1).p_faces)
        S = kinetic_array(source, layout, f_id)
        created = density(grid, kinetic_array(source, layout, f_id))
        expected = [3e15 * 1.5e20, 1e16 * 1.2e20] .* rosenbluth_putvinski_rate(0.01, p_max)
        @test created ≈ expected rtol = 1e-12
        @test all(>=(0), S)
        # nothing below the cutoff
        m = momentum_grid(grid, 1)
        below = findall(<=(0.01), m.p_faces[2:end])
        @test all(iszero, kinetic_view(source, layout, f_id, 1)[below, :])
    end

    @testset "the pitch of the knock-ons follows the field" begin
        m = momentum_grid(grid, 1)
        for (ir, sign) in ((1, 1), (2, -1))
            shape = kinetic_view(source, layout, f_id, ir)
            for i in axes(shape, 1)
                any(!iszero, shape[i, :]) || continue
                gamma = sqrt(1 + m.p_centers[i]^2)
                star = sign * sqrt((gamma - 1) / (gamma + 1))
                j = only(findall(!iszero, shape[i, :]))
                @test m.xi_faces[j] < star < m.xi_faces[j + 1]
            end
        end
        without_field = RosenbluthPutvinskiSourceTerm(:f, grid; electric_field = nothing)
        @test without_field.positive == term.positive
        @test :electric_field ∉ dependencies(without_field)
    end

    @testset "Jacobian against central differences" begin
        system = EquationSystem(layout,
            [Equation(:runaway_density, TimeDerivative()),
                Equation(:total_electron_density, TimeDerivative()),
                Equation(:electric_field, TimeDerivative()),
                Equation(:f, TimeDerivative(), term)];
            initial_state = u)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        rows = collect(Iterators.flatten(unknown_range(layout, f_id, ir) for ir in 1:2))
        rhs_f(x) = (out = zeros(length(x)); rhs!(out, x, system, 0.0); out[rows])
        for id in (1, 2), ir in 1:2

            column = first(unknown_range(layout, id, ir))
            h = 1e-6 * u[column]
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (rhs_f(up) - rhs_f(down)) / 2h
            @test maximum(abs, difference) > 0
            @test maximum(abs, Vector(J[rows, column]) - difference) <=
                  1e-8 * maximum(abs, difference)
        end
        @test_throws ArgumentError RosenbluthPutvinskiSourceTerm(:f, grid; p_cutoff = 0.0)
        @test_throws ArgumentError Runes2P.validate(term, layout, 1)
    end

    @testset "record" begin
        record = describe(term, Runes2P.DescriptionContext())
        @test record["p_cutoff"] == 0.01 && record["runaway_density"] == "runaway_density"
        @test record["electric_field"] == "electric_field"
    end

    @testset "the energy of the knock-ons" begin
        mec2 = Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE
        # A pitch grid with a face at +ξ* of one momentum cell above the cutoff and none at
        # −ξ*: that cell receives knock-ons with ξ* < 0 alone, so the two signs differ.
        p_faces = biuniform_p_faces(40, 15, 0.0, 0.05, 0.8)
        centres = (p_faces[1:(end - 1)] .+ p_faces[2:end]) ./ 2
        i0 = findfirst(>(0.3), centres)
        star0 = sqrt((sqrt(1 + centres[i0]^2) - 1) / (sqrt(1 + centres[i0]^2) + 1))
        xi_faces = collect(uniform_xi_faces(15))
        xi_faces[argmin(abs.(xi_faces .- star0))] = star0
        @test issorted(xi_faces) && !any(≈(-star0), xi_faces)
        energy_grid = cylindrical_grid(MomentumGrid(p_faces, xi_faces), radial)
        energy_layout = UnknownLayout(radial,
            [RadialUnknown(:runaway_density; scale = 1e16),
                RadialUnknown(:total_electron_density; scale = 1e20),
                RadialUnknown(:electric_field), KineticUnknown(:f, energy_grid; scale = 1e19),
                RadialUnknown(:heating; scale = 1e3)])
        rp_source = RosenbluthPutvinskiSourceTerm(:f, energy_grid)
        heating = KineticEnergyChangeTerm(rp_source)
        row = unknown_id(energy_layout, :heating)
        v = zeros(length(energy_layout))
        unknown_values(v, energy_layout, 1) .= [3e15, 1e16]
        unknown_values(v, energy_layout, 2) .= [1.5e20, 1.2e20]
        Runes2P.validate(heating, energy_layout, row)

        # Σ_i ε_i R_i over the momentum cells above the cutoff whose pitch cells hold s ξ*
        # strictly, with the rate of each cell and its kinetic energy ε_i = mₑc²(γ_i − 1)
        function expected_energy(s)
            total = 0.0
            for i in eachindex(centres)
                p_faces[i + 1] <= 0.01 && continue
                gamma = sqrt(1 + centres[i]^2)
                star = s * sqrt((gamma - 1) / (gamma + 1))
                any(j -> xi_faces[j] < star < xi_faces[j + 1], 1:15) || continue
                rate = rosenbluth_putvinski_rate(max(p_faces[i], 0.01), p_faces[i + 1])
                total += mec2 * (gamma - 1) * rate
            end
            return total
        end
        @test expected_energy(1) < (1 - 1e-3) * expected_energy(-1)
        n_re_n_tot = [3e15 * 1.5e20, 1e16 * 1.2e20]
        for fields in ([2.0, -1.5], [-2.0, 1.5])
            unknown_values(v, energy_layout, 3) .= fields
            out = Runes2P.rhs!(zeros(length(v)), heating, v, energy_layout, row, 0.0)
            power = unknown_values(out, energy_layout, row)
            for ir in 1:2
                @test power[ir] ≈ n_re_n_tot[ir] * expected_energy(sign(fields[ir])) rtol = 1e-13
            end
        end
        # without a field, the pitch of E ≥ 0
        unsigned = KineticEnergyChangeTerm(
            RosenbluthPutvinskiSourceTerm(:f, energy_grid; electric_field = nothing))
        out = Runes2P.rhs!(zeros(length(v)), unsigned, v, energy_layout, row, 0.0)
        @test unknown_values(out, energy_layout, row) ≈ n_re_n_tot .* expected_energy(1) rtol = 1e-13

        # The closed form: the rate per unit γ is 2π r₀² c/(γ − 1)², so the energy per runaway
        # and per electron of the total density is 2π r₀² c mₑc² ln[(γ_max − 1)/(γ_cut − 1)]
        # for a cutoff on a face. The sum over the cells converges to it at second order
        # (measured: relative errors 1.75e-2, 4.41e-3, 1.10e-3 and 2.76e-4 for 40 to 320 cells).
        gamma_minus_one(p) = p^2 / (1 + sqrt(1 + p^2))
        closed = 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT *
                 mec2 *
                 log(gamma_minus_one(0.8) / gamma_minus_one(0.04))
        errors = map((40, 80, 160, 320)) do cells
            refined = cylindrical_grid(
                MomentumGrid(uniform_p_faces(cells, 0.0, 0.8), uniform_xi_faces(7)), radial)
            refined_term = KineticEnergyChangeTerm(RosenbluthPutvinskiSourceTerm(
                :f, refined; p_cutoff = 0.04, electric_field = nothing))
            return abs(refined_term.energies.positive[1] - closed) / closed
        end
        @test errors[1] < 3e-2
        orders = log2.(errors[1:(end - 1)] ./ errors[2:end])
        @test all(order -> 1.9 < order < 2.1, orders)

        # the Jacobian: the columns of n_re and n_tot against central differences, none of
        # the field, whose sign enters without a derivative
        unknown_values(v, energy_layout, 3) .= [2.0, -1.5]
        system = EquationSystem(energy_layout,
            [Equation(:runaway_density, TimeDerivative()),
                Equation(:total_electron_density, TimeDerivative()),
                Equation(:electric_field, TimeDerivative()),
                Equation(:f, TimeDerivative(), rp_source),
                Equation(:heating, TimeDerivative(), heating)];
            initial_state = v)
        J = jacobian_prototype(system)
        jacobian!(J, system, v, 0.0)
        rows = [first(unknown_range(energy_layout, row, ir)) for ir in 1:2]
        rhs_rows(x) = (o = zeros(length(x)); rhs!(o, x, system, 0.0); o[rows])
        for id in (1, 2), ir in 1:2

            column = first(unknown_range(energy_layout, id, ir))
            h = 1e-6 * v[column]
            up, down = copy(v), copy(v)
            up[column] += h
            down[column] -= h
            difference = (rhs_rows(up) - rhs_rows(down)) / 2h
            @test maximum(abs, difference) > 0
            @test maximum(abs, Vector(J[rows, column]) - difference) <=
                  1e-8 * maximum(abs, difference)
        end
        field_columns = [first(unknown_range(energy_layout, 3, ir)) for ir in 1:2]
        @test iszero(J[rows, field_columns])
        @test all(c -> c ∉ field_columns, findnz(J[rows, :])[2])

        # the record holds the source and the region; no allocation after the first call
        record = describe(heating, Runes2P.DescriptionContext())
        @test record["coefficient"] == 1.0 && record["kinetic"] == "f"
        @test record["source"]["p_cutoff"] == 0.01
        @test !haskey(record, "region")
        fixed = KineticEnergyChangeTerm(rp_source; region = MomentumRegion(0.2))
        @test describe(fixed, Runes2P.DescriptionContext())["region"]["limit"] == 0.2
        out = zeros(length(v))
        @test call_allocation(Runes2P.rhs!, out, heating, v, energy_layout, row, 0.0) == 0 skip = !OPTIMIZED

        # a fixed region weights the cells by its envelope; a moving limit and another source
        # are refused
        above = KineticEnergyChangeTerm(rp_source; region = MomentumRegion(0.0))
        @test above.energies.positive ≈ heating.energies.positive rtol = 1e-14
        @test all(iszero, KineticEnergyChangeTerm(
            rp_source; region = MomentumRegion(0.9)).energies.positive)
        @test all(fixed.energies.positive .< heating.energies.positive)
        @test_throws "fixed limit" KineticEnergyChangeTerm(
            rp_source; region = MomentumRegion(ThermalMomentum(:cold_electron_temperature)))
        @test_throws "fixed limit" KineticEnergyChangeTerm(
            rp_source; region = MomentumRegion(t -> 0.2))
        @test_throws DimensionMismatch KineticEnergyChangeTerm(
            rp_source; region = MomentumRegion([0.1, 0.2, 0.3]))
        @test_throws "takes a RosenbluthPutvinskiSourceTerm" KineticEnergyChangeTerm(
            LinearTerm(:f))
        @test_throws ArgumentError Runes2P.validate(heating, energy_layout, 4)
        @test_throws DimensionMismatch Runes2P.validate(
            KineticEnergyChangeTerm(term), energy_layout, row)
    end
end

# DREAM's fluid avalanche growth rate with the effective critical field
@testset "effective critical field and the fluid avalanche growth rate" begin
    dream_settings = CollisionSettings(;
        screening = :partial, coulomb_logarithm = :energy_dependent,
        screening_window = :dream,
        ion_size_table = :dream, bremsstrahlung_constant = :dream)

    @testset "the limit of Connor and Hastie" begin
        # A fully ionized plasma with a constant Coulomb logarithm, no screening, no
        # synchrotron loss (B = 0) and no bremsstrahlung: U(p) rises to Ê − Ê_c as p → ∞, so
        # the effective critical field is the critical field of J. W. Connor and R. J.
        # Hastie, Nucl. Fusion 15, 415 (1975), doi:10.1088/0029-5515/15/3/007, up to the cap of
        # the momentum at 1000; with ν̂_s = C n lnΛ and ν̂_D = (1 + Z) C n lnΛ, p⋆²Ê = √(5 + Z) C n lnΛ
        # and Γ = n_tot C (E/E_c − 1)/√(5 + Z).
        n, T, Z = 1e20, 1000.0, 1
        ions = [IonSpecies("D", Z, Z, [n / Z])]
        plain = CollisionSettings(;
            screening = :complete, coulomb_logarithm = :thermal, bremsstrahlung = :neglect)
        plasma(E) = PlasmaState(; cold_electron_temperature = [T], ions,
            electric_field = [E], magnetic_field = [0.0],
            cold_electron_density = [n])
        lnL = thermal_coulomb_logarithm(n, T)
        Ec = critical_field(n, lnL)
        # the method of a collision model keeps the constant logarithm, which DREAM's settings
        # of the effective critical field replace by the energy-dependent one
        model = CollisionModel(plasma(1.0),
            CollisionSettings(; mode = :superthermal, screening = :complete,
                coulomb_logarithm = :thermal, bremsstrahlung = :neglect))
        field, p_opt = effective_critical_field(model, 1)
        @test field ≈ Ec rtol = 2e-3
        @test field >= Ec
        @test p_opt ≈ 1000
        C = 4π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT
        # The whole growth rate through its method on collision models (the same constant
        # logarithm for p⋆): p⋆²Ê_eff = √(5 + Z) C n lnΛ at every field, so
        # Γ = n_tot C (E − E_ceff)/(√(5 + Z) E_c) above and below the effective critical field;
        # below it the rate is negative and linear in |E|, with the sign of the field taken away.
        rate(E) = first(fluid_avalanche_growth_rate(
            CollisionModel(plasma(E), model.settings),
            CollisionModel(plasma(E), model.settings), 1, n))
        for E in (0.3Ec, 0.6Ec, 3Ec, 10Ec)
            @test rate(E) ≈ n * C * (E - field) / (sqrt(5 + Z) * Ec) rtol = 1e-8
        end
        @test rate(0.3Ec) < rate(0.6Ec) < 0 && rate(-0.6Ec) == rate(0.6Ec)
        # the method of a plasma applies DREAM's settings: the energy-dependent logarithm raises
        # the effective critical field above the limit
        dream_rate, dream_field, _ = fluid_avalanche_growth_rate(plasma(3Ec), 1, n; settings = plain)
        @test dream_field > 1.1 * field && 0 < dream_rate < rate(3Ec)
    end

    @testset "the maximum and the root by quadrature, with the synchrotron loss" begin
        # An independent reference of the effective critical field with what the limit of Connor
        # and Hastie switches off: the synchrotron loss of a magnetic field, which holds the largest
        # advection at a finite momentum, and the pitch averages away from ⟨ξ₀⟩ = 1 (A = 6.3 at the
        # optimum, ⟨ξ₀⟩ = 0.84). The pitch averages of exp(−A(1 − ξ₀)) are integrated numerically
        # over ξ₀, the maximum over p is found by a scan in log p and a ternary search, and the
        # root in the field by bisection: no closed form and no search of the code is used.
        # The frequencies are those of the code; test_physics.jl checks their ratio
        # ν_D/ν_s = (1 + Z)/γ for complete screening, and test_radiation_losses.jl the
        # synchrotron loss against the Larmor power.
        n, T, Z, B = 1e20, 1000.0, 1, 5.0
        ions = [IonSpecies("D", Z, Z, [n / Z])]
        settings = CollisionSettings(;
            mode = :superthermal, screening = :complete, coulomb_logarithm = :thermal,
            bremsstrahlung = :neglect)
        plasma = PlasmaState(; cold_electron_temperature = [T], ions,
            electric_field = [1.0], magnetic_field = [B],
            cold_electron_density = [n])
        model = CollisionModel(plasma, settings)
        c, e, me = Constants.SPEED_OF_LIGHT, Constants.ELEMENTARY_CHARGE,
        Constants.ELECTRON_MASS
        tau_inverse = e^4 * B^2 / (6π * Constants.VACUUM_PERMITTIVITY * me^3 * c^3)
        # ⟨ξ₀⟩ and ⟨1 − ξ₀²⟩ of exp(−A(1 − ξ₀)) by Simpson's rule on 2000 intervals of [−1, 1]
        function pitch_averages(A)
            N = 2000
            h = 2 / N
            z, zx, zxx = 0.0, 0.0, 0.0
            for k in 0:N
                xi = -1 + k * h
                g = (k == 0 || k == N ? 1 : isodd(k) ? 4 : 2) * exp(-A * (1 - xi))
                z += g
                zx += g * xi
                zxx += g * xi^2
            end
            return zx / z, 1 - zxx / z
        end
        function advection(p, E)
            w = e * E / (me * c)
            A = 2w / (p * deflection_frequency(model, 1, p))
            mean_xi, mean_perp = pitch_averages(A)
            return w * mean_xi - p * slowing_down_frequency(model, 1, p) -
                   p * sqrt(1 + p^2) * tau_inverse * mean_perp
        end
        # the largest advection over p: the best of 601 points in log p, then a ternary search
        # between its neighbours
        function largest(E)
            xs = range(log(1e-2), log(1e3); length = 601)
            k = argmax([advection(exp(x), E) for x in xs])
            a, b = xs[max(k - 1, 1)], xs[min(k + 1, end)]
            for _ in 1:80
                m1, m2 = a + (b - a) / 3, b - (b - a) / 3
                advection(exp(m1), E) < advection(exp(m2), E) ? (a = m1) : (b = m2)
            end
            return advection(exp((a + b) / 2), E), exp((a + b) / 2)
        end
        Ec = critical_field(n, thermal_coulomb_logarithm(n, T))
        lower, upper = Ec, 20Ec
        @test first(largest(lower)) < 0 < first(largest(upper))
        for _ in 1:60
            mid = (lower + upper) / 2
            first(largest(mid)) < 0 ? (lower = mid) : (upper = mid)
        end
        E_ref = (lower + upper) / 2
        p_ref = last(largest(E_ref))
        field, p_opt = effective_critical_field(model, 1)
        @test field ≈ E_ref rtol = 1e-6
        @test p_opt ≈ p_ref rtol = 1e-5
        # the synchrotron loss raises the field above the limit of Connor and Hastie and holds
        # the maximum at a finite momentum
        @test field > 1.5Ec && 1 < p_opt < 10
        # the closed forms of the pitch averages against the quadrature at the optimum
        A = 2 * e * field / (me * c) / (p_opt * deflection_frequency(model, 1, p_opt))
        mean_xi, mean_perp = pitch_averages(A)
        @test 3 < A < 10 && mean_xi < 0.9
        @test Runes2P._langevin(A) ≈ mean_xi rtol = 1e-8
        @test 2 * Runes2P._langevin_over(A) ≈ mean_perp rtol = 1e-8
    end

    @testset "DREAM's regression values" begin
        # DREAM's unit tests of its RunawayFluid at 93d27e1 (tests/cxx/tests/DREAM/RunawayFluid.cpp,
        # CompareEceffWithTabulated L313-357 and CompareGammaAvaWithTabulated L394-436, the plasma
        # of GetUnknownHandler L78-136): every charge state of Ne and Ar in order gets (k + 1)
        # 1e18 m⁻³, T = 1 eV, in a cylinder. DREAM's tolerances are 1e-3 and 2e-2; DREAM solves to
        # 2e-3 in the field (Runes2P to 1e-13), and interpolates the pitch averages (Runes2P: closed form).
        neon = [(k + 1) * 1e18 for k in 0:10]
        argon = [(k + 1) * 1e18 for k in 11:29]
        free = sum((0:10) .* neon) + sum((0:18) .* argon)
        total = 10sum(neon) + 18sum(argon)
        plasma(E, B) = PlasmaState(;
            cold_electron_temperature = [1.0], electric_field = [E], magnetic_field = [B],
            cold_electron_density = [free],
            ions = [IonSpecies("Ne", 10, reshape(neon, :, 1)),
                IonSpecies("Ar", 18, reshape(argon, :, 1))])
        @test first(effective_critical_field(plasma(20.0, 5.0), 1; settings = dream_settings)) ≈
              8.928710 rtol = 1e-3
        @test first(effective_critical_field(plasma(20.0, 0.1), 1; settings = dream_settings)) ≈
              8.062882 rtol = 1e-3
        for (E, rate) in ((20.0, 159.791), (620.0, 11533.2), (1220.0, 24326.7))
            @test first(fluid_avalanche_growth_rate(plasma(E, 5.0), 1, total; settings = dream_settings)) ≈
                  rate rtol = 2e-2
        end
    end

    @testset "the external avalanche term" begin
        radial = RadialGrid(uniform_r_faces(2, 0.5))
        composition = IonComposition(["D" => 1, "Ar" => 18])
        argon = zeros(19, 2)
        argon[2, :] .= [3e18, 1e18]
        argon[5, :] .= [1e18, 2e18]
        prescribed = PlasmaState(; cold_electron_temperature = [5.0, 3.0],
            electric_field = zeros(2), magnetic_field = fill(2.5, 2),
            ions = [
                IonSpecies("D", 1, [1e18 2e18; 2.5e19 2e19]), IonSpecies("Ar", 18, argon)])
        layout = UnknownLayout(radial,
            [RadialUnknown(:runaway_density; scale = 1e16),
                RadialUnknown(:total_electron_density; scale = 1e20),
                RadialUnknown(:electric_field), RadialUnknown(:cold_electron_temperature; scale = 10.0),
                RadialUnknown(:cold_electron_density; scale = 1e19), ion_densities_unknown(composition)])
        term = ExternalAvalancheTerm(
            prescribed; p_cutoff = 0.8, ions = :ion_densities, settings = dream_settings)
        u = zeros(length(layout))
        set_ion_densities!(u, layout, prescribed)
        unknown_values(u, layout, 1) .= [1e16, 3e15]
        unknown_values(u, layout, 2) .= [sum(1 .*
                                             prescribed.ions[1].charge_state_densities[:, ir]) +
                                         18 * sum(argon[:, ir]) for ir in 1:2]
        unknown_values(u, layout, 3) .= [60.0, 0.05]      # above and below the effective critical field
        unknown_values(u, layout, 4) .= [5.0, 3.0]
        unknown_values(u, layout, 5) .= [free_electron_density(prescribed, ir)
                                         for ir in 1:2]
        out = Runes2P.rhs!(zeros(length(u)), term, u, layout, 1, 0.0)
        rates = unknown_values(out, layout, 1) ./ unknown_values(u, layout, 1)
        # the match: below the fluid rate and the external rate where both are positive, the
        # fluid rate where it is negative
        local_plasma(ir) = PlasmaState(;
            cold_electron_temperature = [unknown_values(u, layout, 4)[ir]],
            cold_electron_density = [unknown_values(u, layout, 5)[ir]],
            electric_field = [unknown_values(u, layout, 3)[ir]], magnetic_field = [2.5],
            ions = [IonSpecies(s.name, s.nuclear_charge, s.charge_state_densities[:, ir:ir])
                    for s in prescribed.ions])
        fluid = [first(fluid_avalanche_growth_rate(
                     local_plasma(ir), 1, unknown_values(u, layout, 2)[ir];
                     settings = dream_settings)) for ir in 1:2]
        external = unknown_values(u, layout, 2) .* rosenbluth_putvinski_rate(0.8)
        @test fluid[1] > 0 && fluid[2] < 0
        @test rates[1] ≈ (external[1]^-2 + fluid[1]^-2)^-0.5 rtol = 1e-12
        @test rates[1] < min(external[1], fluid[1])
        @test rates[2] ≈ fluid[2] rtol = 1e-12
        # the exact Jacobian, every column against central differences (default optimisation)
        system = EquationSystem(layout,
            [Equation(:runaway_density, TimeDerivative(), term),
                Equation(:total_electron_density, TimeDerivative()),
                Equation(:electric_field, TimeDerivative()),
                Equation(:cold_electron_temperature, TimeDerivative()),
                Equation(:cold_electron_density, TimeDerivative()),
                Equation(:ion_densities, TimeDerivative())];
            initial_state = u)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        rows = [first(unknown_range(layout, 1, ir)) for ir in 1:2]
        rhs_rows(x) = (o = zeros(length(x)); rhs!(o, x, system, 0.0); o[rows])
        for id in 1:6, ir in 1:2, column in unknown_range(layout, id, ir)
            u[column] == 0 && continue
            h = 1e-5 * abs(u[column])
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (rhs_rows(up) - rhs_rows(down)) / 2h
            @test Vector(J[rows, column])≈difference rtol=1e-4 atol=1e-9*maximum(abs, out[rows])/abs(u[column])
        end
        # the rows that hold the term, and the residual and the Jacobian at one state, share one
        # evaluation per radius; the Jacobian takes every partial in one pass
        Runes2P.rhs!(zeros(length(u)), term, u, layout, 1, 0.0)     # the cache at the state u, after the differences
        Runes2P.add_jacobian!(J, term, u, layout, 1, 0.0)
        evaluations = [entry.evaluations for entry in term.cache]
        Runes2P.rhs!(zeros(length(u)), term, u, layout, 1, 0.0)
        Runes2P.add_jacobian!(J, term, u, layout, 1, 0.0)
        @test [entry.evaluations for entry in term.cache] == evaluations
        u[first(unknown_range(layout, 1, 1))] *= 1.1
        Runes2P.rhs!(zeros(length(u)), term, u, layout, 1, 0.0)
        @test [entry.evaluations for entry in term.cache] == evaluations .+ [1, 0]
        Runes2P.add_jacobian!(J, term, u, layout, 1, 0.0)
        @test [entry.evaluations for entry in term.cache] == evaluations .+ [2, 0]
        Runes2P.rhs!(zeros(length(u)), term, u, layout, 1, 1.0)    # another time: the prescribed plasma may change
        @test [entry.evaluations for entry in term.cache] == evaluations .+ [3, 1]
        @test describe(term, Runes2P.DescriptionContext())["p_cutoff"] == 0.8
        @test_throws ArgumentError ExternalAvalancheTerm(PlasmaFromUnknowns(prescribed); p_cutoff = 0.8)
    end
end
