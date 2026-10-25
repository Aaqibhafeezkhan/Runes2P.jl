# Advective interpolation schemes and flux limiters against DREAM, on
# the settings of DREAM's examples/fluxlimiter/generate_data.py (Ola Embréus, 2020-11-23)
# run with its linearly implicit solver by reference/dream/cases/flux_limiters.py:
# 500 × 40 cells to p = 0.3, T = 100 eV, n = 5e19 m⁻³ D, E = 6.745 V/m, full collisions,
# 10 steps to 1 ms. For the schemes linear in f the linearly implicit solver is backward
# Euler, compared over the whole run. For the flux limiters DREAM takes the limiter weights
# from the previous time level, and upwind weights in its first step; each such step is
# reproduced with the limiter frozen at DREAM's own state (`apply!(...; limiter_state)`),
# which checks the slope ratio, the limiters, the κ-regions and the ghost cells in p and ξ.
using HDF5
using LinearAlgebra: I
using SparseArrays: sparse

const DREAM_FLUX_LIMITER_DATA = reference_file("dream_flux_limiters.h5")

@testset "flux limiters vs DREAM" begin
    h5open(DREAM_FLUX_LIMITER_DATA) do h5
        attr = attributes(h5)
        @test read(attr["example"]) == "examples/fluxlimiter/generate_data.py"
        setting(key) = only(read(h5["settings/$key"]))
        # the example's values
        @test (setting("hottailgrid/np"), setting("hottailgrid/nxi"),
            setting("hottailgrid/pmax")) == (500, 40, 0.3)
        @test setting("collisions/collfreq_mode") == 2         # COLLFREQ_MODE_FULL
        @test setting("collisions/collfreq_type") == 3         # PARTIALLY_SCREENED
        @test setting("collisions/lnlambda") == 3              # LNLAMBDA_THERMAL (default)
        @test setting("eqsys/f_hot/boundarycondition") == 1    # BC_F_0
        @test setting("eqsys/f_hot/synchrotronmode") == 1      # neglected
        @test setting("eqsys/f_hot/adv_interp/fluxlimiterdamping") == 1.0
        @test setting("solver/type") == 1                      # LINEAR_IMPLICIT, as the example
        @test (setting("timestep/nt"), setting("timestep/tmax")) == (10, 1e-3)

        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        g = cylindrical_grid(mom, RadialGrid(read(h5["grid/r_f"])))
        n, T, E = read(h5["n"]), read(h5["T"]), read(h5["E"])
        plasma = PlasmaState(cold_electron_temperature = [T], cold_electron_density = [n],
            ions = [IonSpecies("D", 1, 1, [n])], electric_field = [E], magnetic_field = [5.0])
        settings = CollisionSettings(
            mode = :full, screening = :partial, coulomb_logarithm = :thermal,
            ion_size_table = :dream, bremsstrahlung = :neglect)
        times = read(h5["t"])
        dt = times[2] - times[1]
        f_initial = read(h5["f_initial"])   # (p, ξ)
        relative(a, b) = maximum(abs, a .- b) / maximum(abs, b)

        @testset "schemes linear in f: backward Euler over the run" begin
            # observed: centred 1.0e-10, QUICK 2.2e-10, second-order upwind 4.8e-11
            for (name, scheme, dream_code) in (("centred", :central, 1), (
                "quick", :quick, 5),
                ("upwind_second_order", :upwind_second_order, 3))
                @test only(read(h5["$name/adv_interp/p1"])) ==
                      only(read(h5["$name/adv_interp/p2"])) == dream_code
                op = fokker_planck_operator(
                    g, plasma, settings; scheme, synchrotron = false)
                layout = UnknownLayout(g.radial_grid, [KineticUnknown(:hot_tail_distribution, g; scale = n)])
                system = EquationSystem(layout,
                    [Equation(:hot_tail_distribution, TimeDerivative(),
                        FokkerPlanckTerm(:hot_tail_distribution, op))])
                solution = integrate(system, vec(f_initial), times)
                f_dream = read(h5["$name/f_10"])
                @test relative(reshape(solution.states[:, end], size(f_dream)), f_dream) <
                      1e-9
            end
        end

        @testset "QUICK in superthermal mode: the p = 0 face" begin
            # p²Aᵖ does not vanish at p = 0 in superthermal mode, so the flux through p = 0
            # carries the scheme's face value there, with the stencil direction from the
            # E-field advection on that face (observed 1.2e-14)
            @test only(read(h5["quick_superthermal/collfreq_mode"])) == 1   # SUPERTHERMAL
            superthermal = CollisionSettings(
                mode = :superthermal, screening = :partial, coulomb_logarithm = :thermal,
                ion_size_table = :dream, bremsstrahlung = :neglect)
            op = fokker_planck_operator(
                g, plasma, superthermal; scheme = :quick, synchrotron = false)
            @test all(!=(0), op.advection_p_times_p2_at_zero)
            layout = UnknownLayout(g.radial_grid, [KineticUnknown(:hot_tail_distribution, g; scale = n)])
            system = EquationSystem(layout,
                [Equation(:hot_tail_distribution, TimeDerivative(),
                    FokkerPlanckTerm(:hot_tail_distribution, op))])
            solution = integrate(system, vec(f_initial), times)
            f_dream = read(h5["quick_superthermal/f_10"])
            @test relative(reshape(solution.states[:, end], size(f_dream)), f_dream) < 1e-12
        end

        # a step of the frozen-limiter operator: (I/dt - L[weights of state]) f_new = f_old/dt
        function frozen_step(op, f_old, state)
            residual!(y, x) = (
                fill!(y, 0); apply!(y, op, x; limiter_state = state); nothing)
            sj = SparseJacobian(residual!, g; backend = sparse_ad_backend(pattern = stencil_pattern(op)))
            L = jacobian(sj, zeros(size(g)))
            return reshape((sparse(1.0I, size(L)...) ./ dt .- L) \ vec(f_old ./ dt), size(f_old))
        end

        @testset "flux limiters: DREAM's lagged steps" begin
            # observed: step 1 1.2e-11 (upwind), step 2 3e-12 to 3.3e-11
            upwind = fokker_planck_operator(
                g, plasma, settings; scheme = :upwind, synchrotron = false)
            first_step_upwind = frozen_step(upwind, reshape(f_initial, size(g)), nothing)
            for (name, dream_code) in (("smart", 6), ("muscl", 7), ("ospre", 8), (
                "tcdf", 9))
                @test only(read(h5["$name/adv_interp/p1"])) ==
                      only(read(h5["$name/adv_interp/p2"])) == dream_code
                f1, f2 = read(h5["$name/f_1"]), read(h5["$name/f_2"])
                # DREAM's first rebuild of a flux limiter uses upwind weights
                @test relative(first_step_upwind[:, :, 1], f1) < 1e-10
                op = fokker_planck_operator(
                    g, plasma, settings; scheme = Symbol(name), synchrotron = false)
                step = frozen_step(op, reshape(f1, size(g)), reshape(f1, size(g)))
                @test relative(step[:, :, 1], f2) < 1e-10
                # the limiter weights matter: frozen at the initial state instead, the step
                # differs at a level far above round-off
                @test relative(
                    frozen_step(op, reshape(f1, size(g)), reshape(f_initial, size(g)))[:, :, 1],
                    f2) > 1e-6
            end
        end
    end
end
