# The kinetic ionization against DREAM in its fully kinetic mode
# (reference/dream/cases/kinetic_ionization.py): deuterium and deposited argon at 20 and
# 10 eV, an f_hot with a tail of 1e-3 at 300 eV, DREAM's kinetic ionization above 3 thermal
# momenta next to its fluid ADAS rate (`ions.cpp:280`, Runes2P's IonRateTerm without
# `kinetic`). One backward-Euler step of 1e-9 s: the change of the charge states over the
# step is dt times the rates at the state of the end of the step, which Runes2P evaluates on
# DREAM's state. The same cross sections, moment and threshold in both codes, so the rates
# agree to round-off and DREAM's Newton tolerance. Needs the ADF11 cache.
using HDF5

@testset "kinetic ionization against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the comparison with DREAM is skipped"
    else
        data = h5open(joinpath(@__DIR__, "data", "dream_kinetic_ionization.h5"), "r") do h5
            Dict(name => read(h5[name]) for name in keys(h5))
        end
        # DREAM stores its arrays transposed in HDF5: n_i is (radius, state, time) and
        # f_hot (p, ξ, radius, time) here
        n_i, f_hot = data["n_i"], data["f_hot"]
        nr = size(n_i, 1)
        radial = RadialGrid(uniform_r_faces(nr, data["minor_radius"]))
        @test radial.r_centers ≈ data["r_grid"] rtol = 1e-12
        grid = cylindrical_grid(MomentumGrid(data["p1_f"], data["p2_f"]), radial)
        composition = IonComposition(["D" => 1, "Ar" => 18])
        layout = UnknownLayout(radial, [ion_densities_unknown(composition),
                                        RadialUnknown(:cold_electron_density; scale=1e19),
                                        RadialUnknown(:cold_electron_temperature; scale=10.0),
                                        KineticUnknown(:f, grid; scale=1e19)])
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=data["threshold"]))
        kinetic = KineticIonizationTerm(composition, :f, grid; region)
        fluid = IonRateTerm(composition)
        # the state of the end of the step
        u = zeros(length(layout))
        for ir in 1:nr
            radial_view(u, layout, 1, ir) .= n_i[ir, :, 2]
            kinetic_view(u, layout, 4, ir) .= f_hot[:, :, ir, 2]
        end
        radial_values(u, layout, 2) .= data["n_cold"][:, 2]
        radial_values(u, layout, 3) .= data["T_cold"][:, 2]
        # the hot density of DREAM is the moment of f above the threshold
        hot = DensityMomentTerm(:f, grid; region)
        n_hot = [Runes2P.rhs!(zeros(length(u)), hot, u, layout, 2, 0.0)[first(unknown_range(layout, 2, ir))] for ir in 1:nr]
        @test n_hot ≈ data["n_hot"][:, 2] rtol = 1e-10
        rates = Runes2P.rhs!(zeros(length(u)), fluid, u, layout, 1, 0.0) .+
                Runes2P.rhs!(zeros(length(u)), kinetic, u, layout, 1, 0.0)
        dt = data["dt"]
        for ir in 1:nr
            change = (n_i[ir, :, 2] .- n_i[ir, :, 1]) ./ dt
            computed = rates[unknown_range(layout, 1, ir)]
            for s in 1:2
                states = species_components(composition, s)
                @test maximum(abs, computed[states] .- change[states]) <= 1e-7 * maximum(abs, change[states])
            end
        end
        # the kinetic part is not negligible: without it the argon rates miss by more than 1e-4
        fluid_only = Runes2P.rhs!(zeros(length(u)), fluid, u, layout, 1, 0.0)
        argon = unknown_range(layout, 1, 1)[species_components(composition, 2)]
        change = (n_i[1, 3:21, 2] .- n_i[1, 3:21, 1]) ./ dt
        @test maximum(abs, fluid_only[argon] .- change) > 1e-4 * maximum(abs, change)
    end
end
