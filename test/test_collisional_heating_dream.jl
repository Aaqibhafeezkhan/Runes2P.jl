# The heating of the cold electrons by the friction of a distribution, FrictionEnergyTransferTerm,
# against DREAM's CollisionalEnergyTransferKineticTerm as DREAM saves it (set
# dream-v26.5-initial-state), term by term on DREAM's own states at every saved step:
# - by the runaway grid of DREAM's examples/init_f_re with T_cold self-consistent
#   (dream_runaway_heating.h5, Tcold_fre_coll). The example takes semi-implicit steps
#   (LINEAR_IMPLICIT), in which DREAM evaluates the collision frequency at the plasma of the start
#   of the step, so the term takes T_cold, n_cold and the ions of the step before;
# - by the hot tail of the superthermal disruption phases of test_disruption_superthermal_dream.jl
#   (dream_disruption_superthermal_channels_small.h5, Tcold_fhot_coll), whose Newton steps take
#   the plasma of the same step: DREAM's saved value applies the frequency of the last Newton
#   iterate to the converged f, so the two agree to the Newton tolerance, not to round-off.
# DREAM saves the channels of its row 0 = dW/dt + Σ terms, so the heating is the negative of the
# saved value; the value saved after step m belongs to the state m (OtherQuantityHandler.cpp).
using HDF5

# the largest difference relative to the largest value of DREAM's heating over the radii at each
# step, about ten times the value measured (in the comment)
const HEATING_RUNAWAY = 2e-14   # 1.7e-15, round-off
const HEATING_HOT_TAIL = 3e-7   # 2.3e-8 in the ionization phase, 7.1e-9 in the quench

# the heating of FrictionEnergyTransferTerm per radius from the distribution `f` on `grid`
# (DREAM's (p, ξ, radius)) and the plasma `plasma`
function friction_heating(grid, f, plasma)
    radial = grid.radial_grid
    layout = UnknownLayout(radial,
        [RadialUnknown(:heating; scale = 1e4), KineticUnknown(:f, grid; scale = 1e16)])
    u = zeros(length(layout))
    for ir in 1:radial_cell_count(radial)
        kinetic_view(u, layout, 2, ir) .= f[:, :, ir]
    end
    term = FrictionEnergyTransferTerm(:f, grid, plasma)
    out = rhs!(zeros(length(u)), term, u, layout, 1, 0.0)
    return collect(unknown_values(out, layout, 1))
end

# the plasma of DREAM's state m: the cold temperature and density and the ions of `composition`
function dream_plasma(group, composition, m, magnetic_field)
    states = read(group["eqsys/n_i"])   # (radius, charge state, time)
    ions = [IonSpecies(name, Z, permutedims(states[:, species_components(composition, s), m]))
            for (s, (name, Z)) in enumerate(zip(composition.names, composition.nuclear_charges))]
    nr = size(states, 1)
    return PlasmaState(; cold_electron_temperature = read(group["eqsys/T_cold"])[:, m],
        cold_electron_density = read(group["eqsys/n_cold"])[:, m], ions,
        electric_field = zeros(nr), magnetic_field = fill(magnetic_field, nr))
end

@testset "the collisional heating against DREAM" begin
    @testset "by a runaway grid" begin
        h5open(reference_file("dream_runaway_heating.h5"), "r") do h5
            group = h5["heating"]
            grid = cylindrical_grid(
                MomentumGrid(read(group["grid/runaway/p1_f"]),
                    read(group["grid/runaway/p2_f"])),
                RadialGrid(read(group["grid/r_f"])))
            f = read(group["eqsys/f_re"])
            saved = read(group["other/fluid/Tcold_fre_coll"])   # (radius, step)
            composition = IonComposition(["D" => 1])
            worst = 0.0
            for m in 2:size(f, 4)
                # the plasma of the start of the step, as DREAM's semi-implicit step takes it
                plasma = dream_plasma(group, composition, m - 1, 5.0)
                ours = friction_heating(grid, f[:, :, :, m], plasma)
                theirs = -saved[:, m - 1]
                worst = max(worst, maximum(abs, ours .- theirs) / maximum(abs, theirs))
            end
            @info "the heating by the runaway grid against DREAM: $worst"
            @test worst < HEATING_RUNAWAY
        end
    end

    @testset "by the superthermal hot tail" begin
        # the ions of the disruption case (COMPOSITION of example_disruption.jl); the magnetic field
        # is not read by the term
        composition = IonComposition(["D" => 1, "D_inj" => 1, "Ar" => 18])
        for phase in ("ionization", "quench")
            h5open(reference_file("dream_disruption_superthermal_channels_small.h5"), "r") do h5
                group = h5[phase]
                grid = cylindrical_grid(
                    MomentumGrid(read(group["grid/hottail/p1_f"]),
                        read(group["grid/hottail/p2_f"])),
                    RadialGrid(read(group["grid/r_f"])))
                f = read(group["eqsys/f_hot"])
                saved = read(group["other/fluid/Tcold_fhot_coll"])
                worst = 0.0
                for m in 2:size(f, 4)
                    plasma = dream_plasma(group, composition, m, 2.5)
                    ours = friction_heating(grid, f[:, :, :, m], plasma)
                    theirs = -saved[:, m - 1]
                    worst = max(worst, maximum(abs, ours .- theirs) / maximum(abs, theirs))
                end
                @info "the heating by the hot tail against DREAM, $phase: $worst"
                @test worst < HEATING_HOT_TAIL
            end
        end
    end
end
