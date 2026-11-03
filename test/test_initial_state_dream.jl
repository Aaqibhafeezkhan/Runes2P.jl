# The starts of a run against DREAM's (reference/dream/cases/initial_state.py, set
# dream-v26.5-initial-state): the coronal start of DREAM's examples/ionequil, neon in equilibrium
# next to singly ionized deuterium at 800 eV and at a temperature falling from 800 to 2 eV
# (dream_coronal_start.h5), against fill_coronal_equilibrium!; and the four shapes of f_re of
# DREAM's examples/init_f_re on its runaway grid, at fields of both signs
# (dream_runaway_shapes.h5), against fill_runaway_distribution!. Both at t = 0. The coronal start
# needs the ADF11 cache.
using HDF5

# the largest difference relative to the largest value of DREAM's quantity, about ten times the
# value measured (in the comment)
const CORONAL = 1e-13   # 1.2e-14 of the neon density, over the profile
const SHAPES = 1e-14    # 3.1e-16 of the largest f_re, round-off

@testset "the starts of a run against DREAM" begin
    @testset "runaway shapes" begin
        file = reference_file("dream_runaway_shapes.h5")
        for shape in RUNAWAY_SHAPES
            h5open(file, "r") do h5
                group = h5[String(shape)]
                radial = RadialGrid(read(group["grid/r_f"]))
                grid = cylindrical_grid(
                    MomentumGrid(read(group["grid/runaway/p1_f"]),
                        read(group["grid/runaway/p2_f"])),
                    radial)
                layout = UnknownLayout(radial, [KineticUnknown(:f_re, grid; scale = 1e16)])
                dream = read(group["eqsys/f_re"])   # (p, ξ, radius, time)
                u = zeros(length(layout))
                fill_runaway_distribution!(
                    u, layout, :f_re, read(group["eqsys/n_re"])[:, 1];
                    shape, electric_field = read(group["eqsys/E_field"])[:, 1])
                for ir in 1:radial_cell_count(radial)
                    reference = dream[:, :, ir, 1]
                    difference = maximum(abs, kinetic_view(u, layout, 1, ir) .- reference) /
                                 maximum(abs, reference)
                    @info "the runaway shape $shape against DREAM, radius $ir: $difference"
                    @test difference < SHAPES
                end
            end
        end
    end

    @testset "coronal start" begin
        if !isdir(adas_cache_directory())
            @info "no ADF11 cache at $(adas_cache_directory()): the coronal start against DREAM is skipped"
            @test_skip isdir(adas_cache_directory())
        else
            composition = IonComposition(["D" => 1, "Ne" => 10])
            term = IonRateTerm(composition)
            neon = species_components(composition, "Ne")
            for profile in ("uniform", "profile")
                h5open(reference_file("dream_coronal_start.h5"), "r") do h5
                    group = h5[profile]
                    radial = RadialGrid(read(group["grid/r_f"]))
                    layout = UnknownLayout(radial, [ion_densities_unknown(composition)])
                    dream = read(group["eqsys/n_i"])   # (radius, charge state of D then Ne, time)
                    temperature = read(group["eqsys/T_cold"])[:, 1]
                    # DREAM's input: D singly ionized as given, the total of neon in equilibrium
                    u = zeros(length(layout))
                    for ir in 1:radial_cell_count(radial)
                        states = radial_view(u, layout, 1, ir)
                        states[1:2] .= dream[ir, 1:2, 1]
                        states[first(neon)] = sum(dream[ir, neon, 1])
                    end
                    fill_coronal_equilibrium!(u, layout, term, temperature; species = "Ne")
                    difference = maximum(maximum(abs,
                                             radial_view(u, layout, 1, ir)[neon] .-
                                             dream[ir, neon, 1]) /
                                         sum(dream[ir, neon, 1])
                    for ir in 1:radial_cell_count(radial))
                    @info "the coronal start against DREAM, $profile: $difference"
                    @test difference < CORONAL
                end
            end
        end
    end
end
