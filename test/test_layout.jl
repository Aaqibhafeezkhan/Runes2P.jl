# Unknown layout: radius-by-radius order, views, ranges, the description written to files,
# and independence of the code from the order.
using ForwardDiff: Dual

@testset "unknown layout" begin
    rad = RadialGrid(uniform_r_faces(3, 0.6))
    m1 = MomentumGrid(uniform_p_faces(4, 0.0, 2.0), uniform_xi_faces(3))
    m2 = MomentumGrid(uniform_p_faces(5, 0.0, 2.0), uniform_xi_faces(2))
    hot = cylindrical_grid([m1, m2, m1], rad)
    runaway = cylindrical_grid(MomentumGrid(uniform_p_faces(2, 2.0, 10.0), uniform_xi_faces(2)), rad)
    unknowns = [ScalarUnknown(:plasma_current; scale = 1e6),
        RadialUnknown(:electric_field; scale = 10.0),
        KineticUnknown(:hot_tail_distribution, hot; scale = 1e18),
        RadialUnknown(:ion_densities; components = 3, scale = 1e20),
        KineticUnknown(:runaway_distribution, runaway),
        ScalarUnknown(:wall_current)]
    layout = UnknownLayout(rad, unknowns)
    E, f_hot, ions, f_re = (unknown_id(layout, n)
    for n in (:electric_field, :hot_tail_distribution, :ion_densities, :runaway_distribution))
    Ip, Iw = unknown_id(layout, :plasma_current), unknown_id(layout, :wall_current)

    @testset "order: radial, then kinetic unknowns per radius, scalars last" begin
        # radius 1: E (1), ions (3), f_hot (4 × 3), f_re (2 × 2) = 20 entries
        @test unknown_range(layout, E, 1) == 1:1
        @test unknown_range(layout, ions, 1) == 2:4
        @test unknown_range(layout, f_hot, 1) == 5:16
        @test unknown_range(layout, f_re, 1) == 17:20
        @test radius_block_range(layout, 1) == 1:20
        # radius 2 has a hot-tail grid of 5 × 2 cells
        @test radius_block_range(layout, 2) == 21:38
        @test unknown_range(layout, f_hot, 2) == 25:34
        @test radius_block_range(layout, 3) == 39:58
        @test scalar_block_range(layout) == 59:60
        @test unknown_range(layout, Ip) == 59:59 && unknown_range(layout, Iw) == 60:60
        @test length(layout) == 60
        @test unknown_names(layout) ==
              [:plasma_current, :electric_field, :hot_tail_distribution,
            :ion_densities, :runaway_distribution, :wall_current]
        @test unknown(layout, :ion_densities).components == 3
        @test_throws ArgumentError unknown_id(layout, :temperature)
        @test_throws ArgumentError unknown_range(layout, Ip, 1)
        @test_throws ArgumentError unknown_range(layout, E)
        @test_throws ArgumentError UnknownLayout(rad, [
            ScalarUnknown(:a), ScalarUnknown(:a)])
        other_radii = cylindrical_grid(m1, RadialGrid(uniform_r_faces(3, 0.7)))
        @test_throws ArgumentError UnknownLayout(rad, [KineticUnknown(:f, other_radii)])
    end

    # a state with a distinct value per entry, written through the views
    function fill_state(layout)
        u = zeros(length(layout))
        scalar_view(u, layout, unknown_id(layout, :plasma_current))[] = 1.5e6
        scalar_view(u, layout, unknown_id(layout, :wall_current))[] = -2.0
        for ir in 1:3
            radial_view(u, layout, unknown_id(layout, :electric_field), ir) .= 10ir
            radial_view(u, layout, unknown_id(layout, :ion_densities), ir) .= [1, 2, 3] .*
                                                                              1e19 .+ ir
            f = kinetic_view(u, layout, unknown_id(layout, :hot_tail_distribution), ir)
            f .= [100ir + 10i + j for i in axes(f, 1), j in axes(f, 2)]
            kinetic_view(u, layout, unknown_id(layout, :runaway_distribution), ir) .= -ir
        end
        return u
    end

    @testset "views read and write the state" begin
        u = fill_state(layout)
        @test u[59] == 1.5e6 && u[60] == -2.0
        f2 = kinetic_view(u, layout, f_hot, 2)
        @test size(f2) == (5, 2) && f2[3, 2] == 232
        # p fastest within a kinetic block
        @test u[unknown_range(layout, f_hot, 2)] == vec(f2)
        @test u[unknown_range(layout, ions, 3)] == [1e19, 2e19, 3e19] .+ 3
        @test unknown_view(u, layout, :hot_tail_distribution, 1) ==
              kinetic_view(u, layout, f_hot, 1)
        @test unknown_view(u, layout, :electric_field, 3) == [30.0]
        @test unknown_view(u, layout, :plasma_current)[] == 1.5e6
        @test_throws ArgumentError kinetic_view(u, layout, E, 1)
        @test_throws ArgumentError radial_view(u, layout, f_hot, 1)
        # the grid of a kinetic unknown, as terms read it
        @test kinetic_grid(layout, f_hot) === hot && kinetic_grid(layout, f_re) === runaway
        @test_throws ArgumentError kinetic_grid(layout, E)
        @test_throws ArgumentError scalar_view(u, layout, E)
    end

    @testset "type stability, dual numbers, no allocation" begin
        u = fill_state(layout)
        @inferred kinetic_view(u, layout, f_hot, 2)
        @inferred radial_view(u, layout, ions, 2)
        @inferred scalar_view(u, layout, Ip)
        d = Dual.(u, 1.0)
        @test kinetic_view(d, layout, f_hot, 2)[3, 2] == Dual(232.0, 1.0)
        @inferred kinetic_view(d, layout, f_hot, 2)
        total(u) = sum(kinetic_view(u, layout, f_hot, 2)) +
                   radial_view(u, layout, ions, 1)[2] +
                   scalar_view(u, layout, Ip)[]
        total(u)
        @test (@allocated total(u)) == 0
    end

    @testset "scales and description" begin
        scales = unknown_scales(layout)
        @test scales[unknown_range(layout, f_hot, 2)] == fill(1e18, 10)
        @test scales[unknown_range(layout, Ip)] == [1e6]
        @test scales[unknown_range(layout, f_re, 1)] == fill(1.0, 4)
        # one record for the root group of the files and the settings: the settings form
        # with the grids inline
        d = describe(layout)
        @test d["layout_version"] == 2
        entries = d["unknowns"]
        @test entries["hot_tail_distribution"]["kind"] == "kinetic"
        @test entries["hot_tail_distribution"]["sizes"] == [12, 10, 12]
        grid = d["grids"][entries["hot_tail_distribution"]["grid"]]
        @test grid["momentum_grids"]["radius_2"]["p_faces"] == m2.p_faces
        @test entries["ion_densities"]["components"] == 3
        @test entries["wall_current"]["kind"] == "scalar"
        @test d["r_faces"] == rad.r_faces
        @test entries["plasma_current"]["unit"] == "A"
        @test entries["wall_current"]["unit"] == ""
        # the form of the settings differs only by the grids and the units, so the setup
        # hash does not depend on the units
        context = DescriptionContext()
        without_grids = describe(layout, context)
        expected = delete!(copy(d), "grids")
        expected["unknowns"] = Dict(name => delete!(copy(entry), "unit")
        for (name, entry) in entries)
        @test without_grids == expected
        @test all(!haskey(entry, "unit") for entry in values(without_grids["unknowns"]))
        @test describe(context) == d["grids"]
    end

    @testset "units" begin
        # the default units of the package, pinned against the table of the docs page, whose
        # column "In the file" holds the strings written to the output file
        page = read(joinpath(@__DIR__, "..", "docs", "src", "model", "unknowns.md"), String)
        section = split(split(page, "### Units of the unknowns")[2], "\n## ")[1]
        documented = Dict{Symbol, String}()
        for line in split(section, '\n')
            startswith(line, "| `:") || continue
            cells = strip.(split(strip(line, '|'), '|'))
            for m in eachmatch(r"`:(\w+)`", cells[1])
                documented[Symbol(m[1])] = strip(cells[end], '`')
            end
        end
        @test length(documented) == 15
        @test documented == Runes2P.DEFAULT_UNKNOWN_UNITS
        @test all(isascii, values(Runes2P.DEFAULT_UNKNOWN_UNITS))
        radial = RadialGrid(uniform_r_faces(2, 0.5))
        grid = cylindrical_grid(MomentumGrid(uniform_p_faces(3, 0.0, 1.0), uniform_xi_faces(2)), radial)
        for (name, unit) in documented
            @test ScalarUnknown(name).unit == unit
            @test RadialUnknown(name).unit == unit
            # a distribution is in m^-3 whatever its name
            @test KineticUnknown(name, grid).unit == "m^-3"
        end
        @test KineticUnknown(:f, grid).unit == "m^-3"
        @test KineticUnknown(:f, grid; unit = "").unit == ""
        # another name has no unit unless it is declared
        @test RadialUnknown(:temperature).unit == ""
        @test RadialUnknown(:temperature; unit = "eV").unit == "eV"
        @test ScalarUnknown(:plasma_current; unit = "MA").unit == "MA"
        @test unknown_unit(layout, :electric_field) == "V/m"
        @test unknown_unit(layout, f_re) == "m^-3"
        @test unknown_unit(layout, Iw) == ""
        # the helpers keep the unit of the default name under another name
        names = InductionNames(electric_field = :E, plasma_current = :Ip)
        induction = UnknownLayout(radial, induction_unknowns(radial; names))
        @test unknown_unit(induction, :E) == "V/m"
        @test unknown_unit(induction, :Ip) == "A"
        @test unknown_unit(induction, :poloidal_flux) == "V s/m"
        @test unknown_unit(induction, :wall_poloidal_flux) == "V s/m"
        @test unknown_unit(induction, :total_current_density) == "A/m^2"
        ion_unknown = ion_densities_unknown(IonComposition(["D" => 1]); ions = :n_i)
        @test ion_unknown.unit == "m^-3"
        # a local layout keeps the unit
        local_ions = UnknownLayout(radial, [ion_unknown, RadialUnknown(:T; unit = "keV")])
        reduced = Runes2P.local_layout(local_ions, [2, 1])
        @test unknown_unit(reduced, :n_i) == "m^-3"
        @test unknown_unit(reduced, :T) == "keV"
    end

    @testset "code does not depend on the order" begin
        # the same checks through the interface with the scalars placed first, and a state
        # carried between the two orders by name
        first = UnknownLayout(rad, unknowns; scalars = :first)
        @test scalar_block_range(first) == 1:2
        @test radius_block_range(first, 1) == 3:22
        u_first = fill_state(first)
        u_last = fill_state(layout)
        @test u_first != u_last
        radius_unknowns = (
            :electric_field, :ion_densities, :hot_tail_distribution, :runaway_distribution)
        for ir in 1:3, name in radius_unknowns

            @test unknown_view(u_first, first, name, ir) ==
                  unknown_view(u_last, layout, name, ir)
        end
        carried = zeros(length(first))
        @test copy_unknowns!(carried, first, u_last, layout) == sort(unknown_names(layout))
        @test carried == u_first
        # a layout with an unknown removed and one added takes the common ones
        smaller = UnknownLayout(rad,
            [RadialUnknown(:electric_field),
                KineticUnknown(:hot_tail_distribution, hot),
                RadialUnknown(:cold_electron_temperature)])
        v = fill(NaN, length(smaller))
        @test copy_unknowns!(v, smaller, u_last, layout) ==
              [:electric_field, :hot_tail_distribution]
        @test unknown_view(v, smaller, :hot_tail_distribution, 2) ==
              kinetic_view(u_last, layout, f_hot, 2)
        @test all(isnan, unknown_view(v, smaller, :cold_electron_temperature, 1))
        # a kinetic unknown on other momentum grids with the same sizes is refused
        other_grid = cylindrical_grid(MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(2)), rad)
        swapped = UnknownLayout(rad, [KineticUnknown(:runaway_distribution, other_grid)])
        @test_throws ArgumentError copy_unknowns!(zeros(length(swapped)), swapped, u_last, layout)
        # the same counts of radii and components on another radial grid, or with other
        # component groups (ion species in another order), are refused
        wider = RadialGrid(uniform_r_faces(3, 0.7))
        from = UnknownLayout(rad, [RadialUnknown(:electric_field)])
        @test_throws ArgumentError copy_unknowns!(
            zeros(3), UnknownLayout(wider, [RadialUnknown(:electric_field)]),
            zeros(3), from)
        ions_from = UnknownLayout(rad, [RadialUnknown(:ion_densities; components = 13, groups = [
            1:2, 3:13])])
        ions_to = UnknownLayout(rad, [RadialUnknown(:ion_densities; components = 13, groups = [
            1:11, 12:13])])
        @test_throws ArgumentError copy_unknowns!(
            zeros(length(ions_to)), ions_to, zeros(length(ions_from)), ions_from)
        @test copy_unknowns!(zeros(length(ions_from)), ions_from, ones(length(ions_from)), ions_from) ==
              [:ion_densities]
    end
end
