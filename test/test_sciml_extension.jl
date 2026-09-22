# SciML form of an equation system (package extension Runes2PSciMLBaseExt).
using SciMLBase: SciMLBase

@testset "SciML extension" begin
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    mom = MomentumGrid(uniform_p_faces(8, 0.0, 2.0), uniform_xi_faces(4))
    g = cylindrical_grid(mom, rad)
    plasma = PlasmaState(cold_electron_temperature=[100.0, 200.0], ions=[IonSpecies("D", 1, 1, [1e20, 1e20])],
                         electric_field=[2.0, 1.0], magnetic_field=[5.0, 5.0])
    fp = fokker_planck_operator(g, plasma)
    layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale=1e16),
                                 RadialUnknown(:hot_electron_density; scale=1e16)])
    density_equation = Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:hot_tail_distribution),
                                LinearTerm(:hot_electron_density; coefficient=-1))
    sparse_only = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                   FokkerPlanckTerm(:hot_tail_distribution, fp)),
                                          density_equation])
    knockon = KnockOnOperator(g, g, 0.3)
    with_knockon = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                    FokkerPlanckTerm(:hot_tail_distribution, fp),
                                                    KnockOnTerm(:hot_tail_distribution, knockon;
                                                                target_density=[1e20, 1e20])),
                                           density_equation])
    u = [1e16 * (1 + 0.1 * sin(k)) for k in 1:length(layout)]
    v = [cos(0.3k) for k in 1:length(layout)] .* unknown_scales(layout)
    t = 0.0

    F = ode_function(sparse_only)
    @test F isa SciMLBase.ODEFunction
    du, expected = zeros(length(u)), zeros(length(u))
    F(du, u, nothing, t)
    @test du == rhs!(expected, u, sparse_only, t)
    @test F.mass_matrix == mass_matrix(sparse_only)
    # the stored Jacobian is complete here, so it is offered as jac
    @test SciMLBase.has_jac(F)
    J = copy(F.jac_prototype)
    F.jac(J, u, nothing, t)
    J_ref = jacobian_prototype(sparse_only)
    jacobian!(J_ref, sparse_only, u, t)
    @test J == J_ref
    Jv = zeros(length(u))
    F.jvp(Jv, v, u, nothing, t)
    @test Jv ≈ J_ref * v rtol = 1e-14

    # with a matrix-free term only the Jacobian-vector product is complete
    G = ode_function(with_knockon)
    @test !SciMLBase.has_jac(G) && SciMLBase.has_jvp(G)
    G(du, u, nothing, t)
    @test du == rhs!(expected, u, with_knockon, t)
    J2 = jacobian_prototype(with_knockon)
    jacobian!(J2, with_knockon, u, t)
    G.jvp(Jv, v, u, nothing, t)
    @test Jv ≈ jacobian_operator(J2, with_knockon, u, t) * v rtol = 1e-14
end
