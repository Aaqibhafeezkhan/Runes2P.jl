# Momentum-conserving field-particle term of the electron-electron collisions:
# conservation, the matrix elements of the test-particle operator against
# Braginskii's, the Lorentz limit and the Spitzer-Härm conductivity, the steady state of
# a run with a prescribed field, and the Woodbury update of the solvers.
import LinearAlgebra
using Random: Xoshiro

const FIELD_PARTICLE_SETTINGS = CollisionSettings(mode=:full, screening=:complete, coulomb_logarithm=:thermal)

function _field_particle_plasma(; T=10.0, ne=1e20, Z=1, E=0.0, nr_=1, cold_fraction=1.0)
    ions = [IonSpecies("X", Z, Z, fill(ne / Z, nr_))]
    return PlasmaState(; cold_electron_temperature=fill(T, nr_), ions, electric_field=fill(E, nr_),
                       magnetic_field=zeros(nr_), cold_electron_density=fill(cold_fraction * ne, nr_))
end

function _field_particle_grid(; np_=80, nxi=4, T=10.0, nr_=1, xi_faces=uniform_xi_faces(nxi))
    theta = T / Constants.ELECTRON_REST_ENERGY
    mom = MomentumGrid(uniform_p_faces(np_, 0.0, 7 * sqrt(2theta)), xi_faces)
    return cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr_, 1.0)))
end

_maxwellian_on(g, T) = [exp(-(sqrt(1 + p^2) - 1) / (T / Constants.ELECTRON_REST_ENERGY))
                        for p in momentum_grid(g).p_centers, _ in 1:pitch_cell_count(g), _ in 1:radial_cell_count(g)]

# Lorentz conductivity σ_L = 64√(2π) ε₀² T^{3/2} / (√m e² Z lnΛ) (Braginskii, Z → ∞)
_lorentz_conductivity(T, ne, Z) =
    64 * sqrt(2π) * Constants.VACUUM_PERMITTIVITY^2 * (T * Constants.ELEMENTARY_CHARGE)^1.5 /
    (sqrt(Constants.ELECTRON_MASS) * Constants.ELEMENTARY_CHARGE^2 * Z * thermal_coulomb_logarithm(T, ne))

# σ/σ_L from the linear response C f₁ = -E f_M, with C the test-particle operator plus the
# field-particle term of rank `modes` (none for modes = 0)
function _field_particle_conductivity(; Z, np_=160, nxi=4, T=10.0, ne=1e20, modes=3, scheme=:central,
                                      cold_fraction=1.0)
    g = _field_particle_grid(; np_, nxi, T)
    plasma = _field_particle_plasma(; T, ne, Z, cold_fraction)
    C = Matrix(operator_matrix(fokker_planck_operator(g, plasma, FIELD_PARTICLE_SETTINGS; scheme)))
    if modes > 0
        op = FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS; modes, scheme)
        C .+= op.restoring_shapes[1] * transpose(op.loss_functionals[1])
    end
    field = AdvectionDiffusion(g; scheme=:central, p_upper_boundary=:f_zero)
    add_electric_field!(field, _field_particle_plasma(; T, ne, Z, E=1.0))
    fM = _maxwellian_on(g, T)
    fM .*= ne / density(g, fM)[1]
    # C is singular on isotropic functions, f₁ is odd in ξ: a tiny shift selects f₁
    shift = 1e-10 * minimum(abs, filter(!iszero, LinearAlgebra.diag(C)))
    f1 = reshape((shift * LinearAlgebra.I - C) \ (operator_matrix(field) * vec(fM)), size(g))
    return abs(current_density(g, f1)[1]) / _lorentz_conductivity(T, ne, Z)
end

# Spitzer and Härm (1953), Table III: σ/σ_L = γ_E(Z)
const SPITZER_HARM = Dict(1 => 0.5816, 2 => 0.6833, 4 => 0.7849, 16 => 0.9225)

@testset "field-particle term" begin
    @testset "conservation to round-off" begin
        for (faces, symmetric) in ((uniform_xi_faces(6), true), (biuniform_theta_xi_faces(4, 3, 0.9), false))
            g = _field_particle_grid(; np_=40, nr_=2, xi_faces=faces)
            plasma = _field_particle_plasma(; nr_=2, Z=2)
            electron_electron = electron_electron_operator(g, plasma, FIELD_PARTICLE_SETTINGS)
            f = rand(Xoshiro(54), size(g)...)
            for modes in 1:3
                op = FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS; modes)
                dee = zeros(size(g))
                Runes2P.apply!(dee, electron_electron, f)
                dfp = zeros(size(g))
                Runes2P.apply!(dfp, op, f)
                loss = parallel_momentum(op, dee)
                # a loss rate far from zero (the collision frequencies are about 1e7 s⁻¹), so
                # the relative check below is not trivially met
                @test all(abs.(loss) .> 1e-3 * maximum(abs, parallel_momentum(op, f)))
                # the sum cancels to round-off times the ratio of the terms to their sum: with an
                # unseeded state the ratio read 1.3e-12 in one suite and below 1e-12 in others
                @test maximum(abs, parallel_momentum(op, dee .+ dfp) ./ loss) < 1e-11
                # the term moves no particles, and the e-e operator conserves them
                @test maximum(abs, density(g, dfp)) < 1e-12 * maximum(abs, density(g, abs.(dfp)))
                @test maximum(abs, density(g, dee)) < 1e-12 * maximum(abs, density(g, abs.(dee)))
                if symmetric
                    # nothing for an f even in ξ, so no energy exchange with the bulk either
                    even = f .+ f[:, end:-1:1, :]
                    deven = zeros(size(g))
                    Runes2P.apply!(deven, op, even)
                    @test maximum(abs, deven) < 1e-12 * maximum(abs, dfp)
                end
            end
        end
    end

    @testset "test-particle matrix elements against Braginskii" begin
        # T_jk = ℓ_j · φ_k are the matrix elements of the discrete e-e test-particle operator
        # between the Laguerre modes p ξ L_k f_M. Their ratios are those of the Landau
        # operator for like species: M⁰¹/M⁰⁰ = 3/4, M¹¹/M⁰⁰ = 59/16, M⁰²/M⁰⁰ = 15/32,
        # M¹²/M⁰⁰ = 417/128, M²²/M⁰⁰ = 8385/1024 (Sugama et al. 2019, appendix C).
        T = 10.0
        theta = T / Constants.ELECTRON_REST_ENERGY
        g = _field_particle_grid(; np_=320, T)
        op = FieldParticleOperator(g, _field_particle_plasma(; T), FIELD_PARTICLE_SETTINGS; scheme=:central)
        m = momentum_grid(g)
        x2 = [p^2 / (1 + sqrt(1 + p^2)) / theta for p in m.p_centers, _ in m.xi_centers]
        pxi = [p * xi for p in m.p_centers, xi in m.xi_centers]
        phi = hcat((vec(pxi .* Runes2P._laguerre.(k, x2) .* exp.(-x2)) for k in 0:2)...)
        elements = transpose(op.loss_functionals[1]) * phi
        expected = [1 3/4 15/32; 3/4 59/16 417/128; 15/32 417/128 8385/1024]
        @test elements ./ elements[1, 1] ≈ expected rtol = 1e-3
    end

    @testset "Lorentz limit" begin
        # without cold electrons the term vanishes with the e-e collisions
        @test _field_particle_conductivity(; Z=1, cold_fraction=1e-10) ≈ 1 atol = 3e-3
    end

    @testset "conductivity against Spitzer-Härm" begin
        # three modes: within 0.3% of γ_E at every Z (320 p cells: -0.22%, +0.30%, +0.07%,
        # +0.10%); the rank-one momentum-restoring model is 11% below at Z = 1
        for Z in (1, 2, 4, 16)
            @test _field_particle_conductivity(; Z) ≈ SPITZER_HARM[Z] rtol = 5e-3
        end
        @test _field_particle_conductivity(; Z=1, modes=2) ≈ SPITZER_HARM[1] rtol = 1e-2
        @test 0.51 < _field_particle_conductivity(; Z=1, modes=1) < 0.52
        @test 0.29 < _field_particle_conductivity(; Z=1, modes=0) < 0.30
        # the cell-centre momentum weights make the result independent of the pitch grid
        @test _field_particle_conductivity(; Z=1, nxi=8) ≈ _field_particle_conductivity(; Z=1) rtol = 1e-6
        # the scheme of the e-e operator changes it at the discretization error only
        @test _field_particle_conductivity(; Z=1, scheme=:exponential_fitting) ≈
              _field_particle_conductivity(; Z=1) rtol = 1e-3
    end

    @testset "steady state with a prescribed field" begin
        # A run from the Maxwellian with a small prescribed E reaches the ohmic steady state:
        # j/E at two radii with different densities against Spitzer-Härm, the same with
        # every solver, and linear in E.
        T, Z, np_, nr_ = 10.0, 1, 80, 2
        densities = [1e20, 5e19]
        g = _field_particle_grid(; np_, T, nr_)
        ions = [IonSpecies("D", Z, Z, densities)]
        plasma(E) = PlasmaState(; cold_electron_temperature=fill(T, nr_), ions, electric_field=fill(E, nr_),
                                magnetic_field=zeros(nr_))
        fM = _maxwellian_on(g, T)
        fM .*= reshape(densities ./ density(g, fM), 1, 1, :)
        op = FieldParticleOperator(g, plasma(0.0), FIELD_PARTICLE_SETTINGS)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e20)])
        times = range(0.0, 2e-5; length=21)   # steps of 1e-6 s, about 30 collision times
        function run_to_steady_state(E, linear_solver)
            fp = fokker_planck_operator(g, plasma(E), FIELD_PARTICLE_SETTINGS)
            system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, fp),
                                                      FieldParticleTerm(:f, op))]; time_scale=1e-6)
            solution = integrate(system, vec(fM), times; solver=NewtonSolver(; linear_solver))
            f = reshape(solution.states[:, end], size(g))
            return current_density(g, f) ./ E, solution.statistics
        end
        E = 0.05   # V/m, E/E_D ≈ 1e-5
        sigma, stats = run_to_steady_state(E, GMRES())
        for ir in 1:nr_
            @test sigma[ir] / _lorentz_conductivity(T, densities[ir], Z) ≈ SPITZER_HARM[Z] rtol = 1e-2
        end
        sigma_lu, _ = run_to_steady_state(E, SparseLU())
        @test sigma_lu ≈ sigma rtol = 1e-8
        sigma_plain, stats_plain = run_to_steady_state(E, GMRES(; low_rank_update=false))
        @test sigma_plain ≈ sigma rtol = 1e-8
        # the Woodbury update takes the term into the preconditioner
        @test sum(stats.krylov_iterations) < sum(stats_plain.krylov_iterations)
        # the same inside the kinetic blocks of FluidSchurLU
        sigma_schur, stats_schur = run_to_steady_state(E, GMRES(; preconditioner=FluidSchurLU()))
        @test sigma_schur ≈ sigma rtol = 1e-8
        @test sum(stats_schur.krylov_iterations) <= sum(stats.krylov_iterations)
        sigma_double, _ = run_to_steady_state(2E, GMRES())
        @test sigma_double ≈ sigma rtol = 1e-4
    end

    @testset "Jacobian and interface" begin
        g = _field_particle_grid(; np_=20, nr_=2)
        plasma = _field_particle_plasma(; nr_=2)
        term = FieldParticleTerm(:f, FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS; modes=2))
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g)])
        u, v = rand(Xoshiro(55), length(layout)), rand(Xoshiro(56), length(layout))
        product = zeros(length(layout))
        Runes2P.jacobian_vector_product!(product, term, u, v, layout, 1, 0.0)
        blocks = low_rank_jacobian(term, u, layout, 1, 0.0)
        @test length(blocks) == 2 && all(b -> size(b.left, 2) == 2, blocks)
        dense = zeros(length(layout))
        for b in blocks
            dense[b.rows] .+= b.left * (transpose(b.right) * v[b.columns])
        end
        @test dense ≈ product rtol = 1e-12
        @test has_low_rank_jacobian(term)
        record = describe(term, Runes2P.DescriptionContext())
        @test record["operator"]["modes"] == 2
        @test record["operator"]["coefficient_source"]["collision_settings"]["mode"] == "full"
        @test !has_low_rank_jacobian(LinearTerm(:f))
        @test_throws ArgumentError FieldParticleOperator(g, plasma, CollisionSettings(mode=:superthermal))
        @test_throws ArgumentError FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS; modes=4)
    end

    @testset "pitch grids per radius (#86)" begin
        # The term is local to each radius, so on a grid whose pitch faces differ between
        # radii each radius must give what it gives on a grid with its pitch faces at every
        # radius. The p faces are shared, so the temperature is the same at every radius.
        T = 10.0
        theta = T / Constants.ELECTRON_REST_ENERGY
        # pitch faces of a uniform grid with the interior faces moved by `shift` cell widths
        shifted(nxi, shift) = vcat(-1.0, uniform_xi_faces(nxi)[2:end-1] .+ shift * 2 / nxi, 1.0)
        grid_of(np_, faces) = cylindrical_grid([MomentumGrid(uniform_p_faces(np_, 0.0, 7 * sqrt(2theta)), x) for x in faces],
                                               RadialGrid(uniform_r_faces(length(faces), 1.0)))
        maxwellian(g, densities) = (f = [exp(-(sqrt(1 + p^2) - 1) / theta) for p in momentum_grid(g, 1).p_centers,
                                         _ in 1:pitch_cell_count(g), _ in 1:radial_cell_count(g)];
                                    f .* reshape(densities ./ density(g, f), 1, 1, :))
        plasma_of(densities, E=0.0) = PlasmaState(; cold_electron_temperature=fill(T, length(densities)),
                                                  ions=[IonSpecies("D", 1, 1, collect(densities))],
                                                  electric_field=fill(E, length(densities)),
                                                  magnetic_field=zeros(length(densities)))
        A, B = uniform_xi_faces(6), shifted(6, 0.3)

        @testset "conservation per radius" begin
            g = grid_of(40, [A, B])
            @test !has_shared_momentum_grid(g)
            plasma = _field_particle_plasma(; nr_=2, Z=2)
            electron_electron = electron_electron_operator(g, plasma, FIELD_PARTICLE_SETTINGS)
            f = rand(Xoshiro(57), size(g)...)
            dee = zeros(size(g))
            Runes2P.apply!(dee, electron_electron, f)
            # the e-e operator of each radius conserves the particles on the faces of that radius
            @test maximum(abs, density(g, dee)) < 1e-12 * maximum(abs, density(g, abs.(dee)))
            for modes in 1:3
                op = FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS; modes)
                dfp = zeros(size(g))
                Runes2P.apply!(dfp, op, f)
                loss = parallel_momentum(op, dee)
                @test all(abs.(loss) .> 1e-3 * maximum(abs, parallel_momentum(op, f)))
                @test maximum(abs, parallel_momentum(op, dee .+ dfp) ./ loss) < 1e-11
                @test maximum(abs, density(g, dfp)) < 1e-12 * maximum(abs, density(g, abs.(dfp)))
            end
        end

        @testset "conductivity per radius" begin
            densities = [1e20, 5e19]
            function conductivity(faces)
                g = grid_of(80, faces)
                plasma = plasma_of(densities)
                return kinetic_conductivity(fokker_planck_operator(g, plasma, FIELD_PARTICLE_SETTINGS);
                                            field_particle=FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS))
            end
            # identical to round-off (0 in the first run). The tolerance is a loose one: the
            # conductivity does not depend on the pitch grid (A against B: 8e-15), so the short
            # run below is the check that each radius uses its own pitch faces
            mixed, on_A, on_B = conductivity([A, B]), conductivity([A, A]), conductivity([B, B])
            @test mixed[1] ≈ on_A[1] rtol = 1e-5
            @test mixed[2] ≈ on_B[2] rtol = 1e-5
        end

        @testset "errors and description" begin
            g = grid_of(20, [A, B])
            plasma = plasma_of([1e20, 5e19])
            unequal = grid_of(20, [A, uniform_xi_faces(4)])
            @test !has_same_cell_counts(unequal)
            @test_throws ArgumentError FieldParticleOperator(unequal, plasma, FIELD_PARTICLE_SETTINGS)
            # no cold electrons at a radius: wrong input data or settings (decision of the author, 2026-09-29)
            without = PlasmaState(; cold_electron_temperature=fill(T, 2), ions=[IonSpecies("D", 1, 1, [1e20, 5e19])],
                                  electric_field=zeros(2), magnetic_field=zeros(2), cold_electron_density=[1e20, 0.0])
            thrown = try
                FieldParticleOperator(g, without, FIELD_PARTICLE_SETTINGS)
                nothing
            catch e
                e
            end
            @test thrown isa DomainError
            @test occursin("no cold electrons at radius 2", sprint(showerror, thrown))
            term = FieldParticleTerm(:f, FieldParticleOperator(g, plasma, FIELD_PARTICLE_SETTINGS))
            record = describe(term, Runes2P.DescriptionContext())
            @test record["operator"]["modes"] == 3
            @test haskey(describe(g)["momentum_grids"], "radius_2")
        end

        @testset "short run on 11 radii" begin
            # pitch grids alternating between radii, densities falling outwards, a field of
            # 0.05 V/m for three steps of 1 μs: each radius against the run on the grid with
            # its pitch faces at every radius
            nr_ = 11
            A4, B4 = uniform_xi_faces(4), shifted(4, 0.3)
            densities = collect(range(1e20, 2e19; length=nr_))
            function short_run(faces, linear_solver)
                g = grid_of(20, faces)
                op = FieldParticleOperator(g, plasma_of(densities), FIELD_PARTICLE_SETTINGS)
                fp = fokker_planck_operator(g, plasma_of(densities, 0.05), FIELD_PARTICLE_SETTINGS)
                layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e20)])
                system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, fp),
                                                          FieldParticleTerm(:f, op))]; time_scale=1e-6)
                solution = integrate(system, vec(maxwellian(g, densities)), range(0.0, 3e-6; length=4);
                                     solver=NewtonSolver(; linear_solver))
                return current_density(g, reshape(solution.states[:, end], size(g)))
            end
            mixed_faces = [isodd(ir) ? A4 : B4 for ir in 1:nr_]
            mixed = short_run(mixed_faces, SparseLU())
            on_A, on_B = short_run(fill(A4, nr_), SparseLU()), short_run(fill(B4, nr_), SparseLU())
            reference = [isodd(ir) ? on_A[ir] : on_B[ir] for ir in 1:nr_]
            # radial lines off: they need one pitch grid
            schur = short_run(mixed_faces, GMRES(; preconditioner=FluidSchurLU(; radial_lines=false)))
            # first run: mixed against the reference 0, the Schur run 3e-13 from the direct one,
            # while the currents of the two pitch grids differ by 3e-5
            @test all(!iszero, mixed)
            @test mixed ≈ reference rtol = 1e-8
            @test schur ≈ mixed rtol = 1e-8
        end
    end
end
