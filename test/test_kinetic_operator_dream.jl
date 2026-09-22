# One backward-Euler step of the small-angle kinetic operator compared with a
# linearly implicit DREAM step (reference/dream/cases/kinetic_operator.py).
using HDF5
using LinearAlgebra: I

const DREAM_KINETIC_DATA = joinpath(@__DIR__, "data", "dream_kinetic_operator.h5")

@testset "kinetic operator step vs DREAM" begin
    h5open(DREAM_KINETIC_DATA) do h5
        mom = MomentumGrid(read(h5["grid/p1_f"]), read(h5["grid/p2_f"]))
        g = cylindrical_grid(mom, RadialGrid(read(h5["grid/r_f"])))
        # DREAM's cylindrical Jacobian differs from Runes2P's by the constant 4π²
        ratio = read(h5["grid/Vprime"]) ./ g.phase_space_jacobian
        @test all(r -> isapprox(r, 4π^2; rtol=1e-14), ratio)

        names = read(h5["ions/name"])
        Zs = read(h5["ions/Z"])
        Z0s = read(h5["ions/Z0"])
        n = read(h5["ions/n"])
        ions = [IonSpecies(names[k], Zs[k], Z0s[k], n[:, k]) for k in eachindex(Zs)]
        dt = read(h5["dt"])
        variants = (
            "full_partial_thermal" => (CollisionSettings(ion_size_table=:dream), 1e-9),
            "superthermal_complete_energy" => (CollisionSettings(mode=:superthermal,
                screening=:complete, coulomb_logarithm=:energy_dependent), 1e-12),
        )
        for (variant, (settings, rtol)) in variants
            @testset "$variant" begin
                grp = h5[variant]
                f = read(grp["f_hot"])
                f0 = f[:, :, :, 1]
                f1 = f[:, :, :, end]
                # linearly implicit: coefficients from the previous time level
                plasma = PlasmaState(cold_electron_temperature=read(grp["T_cold"])[:, 1],
                                     cold_electron_density=read(grp["n_cold"])[:, 1], ions=ions,
                                     electric_field=read(h5["E"]),
                                     magnetic_field=fill(read(h5["B"]), radial_cell_count(g)))
                op = fokker_planck_operator(g, plasma, settings; synchrotron=true)
                M = operator_matrix(op)
                f1_runes2p = reshape((I - dt * M) \ vec(f0), size(g))
                for ir in 1:radial_cell_count(g)
                    a = f1[:, :, ir]
                    b = f1_runes2p[:, :, ir]
                    cut = 1e-12 * maximum(abs, a)
                    rel = maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
                    @test rel < rtol
                end
                # particles leave only through p = 0, equally in both codes
                @test density(g, f1_runes2p) ≈ density(g, f1) rtol = 1e-12
            end
        end
    end
end

@testset "DREAM reference data provenance" begin
    checkout = joinpath(@__DIR__, "..", "reference", "dream", "DREAM")
    for file in (DREAM_COLLISION_DATA, DREAM_KINETIC_DATA)
        h5open(file) do h5
            attr = attributes(h5)
            @test haskey(attr, "dream_commit")
            @test read(attr["dream_branch"]) == "boltzmann-eqterm"
            if isdir(joinpath(checkout, ".git"))
                head = readchomp(`git -C $checkout rev-parse HEAD`)
                @test read(attr["dream_commit"]) == head
            end
        end
    end
end
