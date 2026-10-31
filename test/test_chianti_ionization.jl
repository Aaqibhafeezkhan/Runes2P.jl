# The cross sections of the kinetic ionization from CHIANTI's parameters (K. P. Dere, Astron.
# Astrophys. 466, 771 (2007)), continued relativistically by the Bethe theory: the
# table against the files, the threshold, the continuity of the cross section and its slope
# where the continuation joins the Bethe form, the derivative in the number type of ForwardDiff,
# the fallback of the elements without data, and the argon ions against the relativistic
# distorted-wave calculations of Bao and Li (2024). The measurements of the neutral atoms are
# in test_independent_references.jl, the Maxwellian averages against ADAS in
# test_kinetic_ionization.jl.
using ForwardDiff

@testset "CHIANTI cross sections of the kinetic ionization" begin
    mc2 = Constants.ELECTRON_REST_ENERGY
    momentum(E) = sqrt((1 + E / mc2)^2 - 1)

    @testset "the table reads back the files" begin
        # values copied from the files of CHIANTI 11.0.2 (SolarSoft mirror, read 2026-10-13)
        @test [length(CHIANTI_IONIZATION_FITS[e]) for e in ("H", "Ne", "Ar")] == [1, 10, 18]
        ar9 = only(CHIANTI_IONIZATION_FITS["Ar"][9].direct)                    # ar_9.diparams
        @test CHIANTI_IONIZATION_FITS["Ar"][9].ion == "ar_9"
        @test (ar9.scale, ar9.threshold) == (2.4, 422.451)
        @test ar9.knots == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        @test ar9.values == [0.0, 11.0626, 19.989, 25.11, 22.746, 17.0842]
        ar17 = only(CHIANTI_IONIZATION_FITS["Ar"][17].direct)                  # ar_17.diparams
        @test (ar17.scale, ar17.threshold) == (2.0, 4120.841)
        @test ar17.knots == collect(0:0.125:1)
        @test ar17.values == [0.0, 1.195, 2.335, 3.734, 5.365, 6.847, 7.577, 6.816, 3.824]
        h = only(CHIANTI_IONIZATION_FITS["H"][1].direct)                       # h_1.diparams
        @test (h.scale, h.threshold, h.values[end]) == (3.8, 13.598, 1.85)
        ar3 = CHIANTI_IONIZATION_FITS["Ar"][3]                                 # ar_3: three shells
        @test [s.threshold for s in ar3.direct] == [40.741, 59.64, 280.1]
        ar4 = CHIANTI_IONIZATION_FITS["Ar"][4]                                 # ar_4.easplom
        @test length(ar4.autoionization) == 12 && ar4.autoionization_scale == 1.0
        t = first(ar4.autoionization)
        @test (t.kind, t.oscillator_strength, t.energy, t.scale) == (1, 3.89, 18.72, 1.3)
        @test t.values == [0.1953, 0.1696, 0.1681, 0.1664, 0.207]
        @test [length(f.autoionization) > 0 for f in CHIANTI_IONIZATION_FITS["Ar"]] ==
              [false, false, true, true, true, true, true, true, false,
            false, false, false, false, false, true, true, false, false]
        # the generator writes the same data file from the files where they are at hand
        directory = get(ENV, "RUNES2P_CHIANTI_DIR", joinpath(homedir(), "chianti"))
        if isdir(directory)
            mktempdir() do tmp
                out = joinpath(tmp, "chianti_ionization.toml")
                script = joinpath(@__DIR__, "..", "scripts", "generate_chianti_ionization.jl")
                run(pipeline(`$(Base.julia_cmd()) --startup-file=no $script $directory $out`; stdout = devnull))
                @test read(out, String) ==
                      read(joinpath(@__DIR__, "..", "data", "chianti_ionization.toml"), String)
            end
        else
            @info "no CHIANTI files at $directory (RUNES2P_CHIANTI_DIR): the regeneration of the table is skipped"
            @test_skip isdir(directory)
        end
        # the docstring cites the version and the read date of the header of the data file;
        # scripts/code_changes.jl lists its file, so that a change of the docstring runs the tests
        metadata = Runes2P.parse_data_file(joinpath(Runes2P.DATA_DIRECTORY,
            "chianti_ionization.toml"))["metadata"]
        source = match(r"^CHIANTI (\S+) .*, read on (\S+)$", metadata["version_or_date"])
        docstring = join(split(string(@doc CHIANTI_IONIZATION_FITS)), " ")
        @test source !== nothing
        @test occursin("version $(source[1]) ", docstring)
        @test occursin("read on $(source[2])", docstring)
    end

    @testset "the threshold and the bare nucleus" begin
        for (element, Z) in (("H", 1), ("Ne", 10), ("Ar", 18)), Z0 in 0:(Z - 1)

            I = minimum(s.threshold
            for s in CHIANTI_IONIZATION_FITS[element][Z0 + 1].direct)
            @test ionization_cross_section(element, Z0, momentum(0.999I)) == 0
            @test ionization_cross_section(element, Z0, momentum(1.01I)) > 0
            @test ionization_cross_section(element, Z0, 0.0) == 0
            # positive from the threshold to 10⁶ times it
            @test all(E -> ionization_cross_section(element, Z0, momentum(E)) > 0,
                I .* 10 .^ range(0.01, 6; length = 200) .+ 0.0)
        end
        @test ionization_cross_section("Ar", 18, 1.0) == 0
        @test_throws ArgumentError ionization_cross_section("Ar", 19, 1.0)
    end

    @testset "the cross section and its slope are continuous where the Bethe form takes over" begin
        # the weight passes from the continued fit to the Bethe form between E_m/2 and 2E_m
        # (CHIANTI_MATCHING_ENERGY, CHIANTI_MATCHING_WIDTH); on each side of both ends, and at
        # E_m, where the Bethe form equals the continued fit
        dsigma(element, Z0, p) = ForwardDiff.derivative(x -> ionization_cross_section(element, Z0, x), p)
        Em = Runes2P.CHIANTI_MATCHING_ENERGY
        for (element, Z) in (("H", 1), ("Ne", 10), ("Ar", 18)), Z0 in 0:(Z - 1)

            for E in (Em / Runes2P.CHIANTI_MATCHING_WIDTH, Em *
                                                           Runes2P.CHIANTI_MATCHING_WIDTH)
                below, above = momentum(E * (1 - 1e-9)), momentum(E * (1 + 1e-9))
                @test ionization_cross_section(element, Z0, above) ≈
                      ionization_cross_section(element, Z0, below) rtol = 1e-7
                @test dsigma(element, Z0, above) ≈ dsigma(element, Z0, below) rtol = 1e-6
            end
            fit = CHIANTI_IONIZATION_FITS[element][Z0 + 1]
            p = momentum(Em)
            beta2 = p^2 / (1 + p^2)
            @test Runes2P._fano_ordinate(fit, p) ≈
                  fit.bethe_m2 * (log(p^2) - beta2) + fit.bethe_c rtol = 1e-12
        end
    end

    @testset "the derivative in the number type of ForwardDiff" begin
        for (element, Z0) in (("H", 0), ("Ar", 0), ("Ar", 4), ("Ar", 16), ("Ne", 7)),
            E in (1.5e2, 3e3, 7e4, 1.4e5, 1e6)

            I = minimum(s.threshold
            for s in CHIANTI_IONIZATION_FITS[element][Z0 + 1].direct)
            E > 1.2I || continue
            p = momentum(E)
            h = 1e-6 * p
            central = (ionization_cross_section(element, Z0, p + h) -
                       ionization_cross_section(element, Z0, p - h)) / 2h
            @test ForwardDiff.derivative(x -> ionization_cross_section(element, Z0, x), p) ≈
                  central rtol = 1e-5
        end
    end

    @testset "elements without CHIANTI data keep DREAM's form" begin
        for (element, Z0) in (("W", 3), ("Fe", 10), ("He", 0), ("C", 2)),
            p in (0.05, 0.5, 5.0)

            @test ionization_cross_section(element, Z0, p) ==
                  ionization_cross_section(element, Z0, p; form = :dream)
        end
        @test_throws ArgumentError ionization_cross_section("Xe", 0, 1.0)
        @test Runes2P.chianti_element("D") == Runes2P.chianti_element("T") == "H"
    end

    @testset "argon ions against Bao and Li (2024)" begin
        # R. Bao and B. Li, At. Data Nucl. Data Tables 159, 101663 (2024),
        # doi:10.1016/j.adt.2024.101663: level-to-level distorted wave with the relativistic
        # Flexible Atomic Code, direct ionization and excitation-autoionization, Ar⁺ to Ar¹⁷⁺,
        # given by the fit of their eq. (5), σ = 10⁻¹³/(εI₀) [A₀ ln(ε/I₀) + Σᵢ Aᵢ (1 − I₀/ε)ⁱ] cm²,
        # with I₀ and A₀..A₈ of their Table A (ε and I₀ in eV). Their figures run to about 16
        # times the threshold (Figs. 19 and 20); the crossed-beam measurements they cite are in the figures
        # only, not in the paper's tables, so the test takes their calculation.
        #! format: off
        bao = Dict(1 => (26.36, (2.0082, 7.9127, -82.7503, 423.7535, -1084.3120, 1474.9080, -1055.0340, 359.2106, -43.4720)),
                   2 => (39.39, (3.8239, 2.1709, -80.9699, 636.3123, -2548.8810, 5635.6100, -7026.3510, 4635.2580, -1262.9480)),
                   3 => (58.4, (0.8373, 1.4281, -2.9451, 22.0089, -97.4737, 225.3819, -259.3369, 131.0904, -17.0746)),
                   4 => (73.84, (0.9751, 1.8570, -10.9787, 61.1485, -253.0837, 653.4733, -952.0333, 706.2564, -204.9391)),
                   5 => (90.29, (1.6525, -0.4395, 19.9742, -279.0198, 1490.5440, -4041.1520, 5830.4780, -4250.2500, 1230.6750)),
                   6 => (122.44, (0.5236, 2.6883, -68.4342, 612.7398, -2588.4750, 5696.9570, -6648.5050, 3875.9340, -876.5442)),
                   7 => (142.88, (3.6384, -2.5656, -32.8905, 400.5364, -2306.2180, 6642.0540, -9969.7250, 7476.9940, -2217.3990)),
                   8 => (420.84, (4.6531, 0.5533, -15.4060, 105.0697, -496.7266, 1397.2090, -2207.6330, 1800.3500, -590.8566)),
                   9 => (478.13, (-0.9035, 5.5855, -2.1442, -38.0146, 295.1279, -802.1748, 1095.4690, -751.6319, 210.4895)),
                   10 => (538.61, (-6.0375, 9.4955, 13.6749, -157.2531, 861.0354, -2271.8360, 3246.0140, -2405.2730, 734.9028)),
                   11 => (618.84, (-6.2058, 8.8411, 9.6914, -100.9669, 596.1669, -1652.6500, 2466.7310, -1902.8690, 605.0461)),
                   12 => (686.21, (3.6268, -0.8101, -15.2386, 84.2517, -285.0698, 594.4060, -776.7238, 572.5060, -181.8620)),
                   13 => (755.99, (2.0925, -0.8046, 0.4966, -25.0211, 149.3286, -405.6372, 558.7868, -384.0218, 103.2065)),
                   14 => (849.92, (2.6518, -1.7710, -5.0019, 39.9007, -217.9642, 654.8455, -1092.7170, 932.3148, -317.7987)),
                   15 => (918.1, (1.5115, -1.0740, -2.5212, 18.9358, -105.2461, 322.5170, -549.0846, 476.0584, -164.1591)),
                   16 => (4119.27, (-0.5900, 1.5208, -2.5486, 26.2808, -110.6533, 285.7379, -418.3374, 318.6666, -96.1045)),
                   17 => (4426.22, (-0.1223, 0.5942, -1.4626, 14.5349, -64.8382, 174.0111, -263.2082, 206.3366, -64.3983)))
        #! format: on
        function bao_li(Z0, E)
            I0, A = bao[Z0]
            return 1e-17 / (E * I0) *
                   (A[1] * log(E / I0) + sum(A[i + 1] * (1 - I0 / E)^i for i in 1:8))   # m²
        end
        # Bao and Li are independent of Dere's fits (their own distorted-wave calculations). The
        # bands below are set to contain the ratios found, and the reasons are given with them:
        # they are not uncertainty bands.
        # Bands: Bao and Li find their results within 20 % of the literature; Dere's final errors
        # have the median 10 % and reach 25 % for the ions of this sequence. Where direct
        # ionization dominates (Ar⁺, Ar⁸⁺ to Ar¹⁷⁺), from 1.5 to 10 times the threshold: ±30 %;
        # CHIANTI lies at 0.71 to 1.25 of Bao and Li (Ar⁺ 0.71 at 1.5 I: CHIANTI's fit to the
        # measurements, which Bao and Li exceed near the threshold, their Sect. 3.1). Where the
        # excitation-autoionization is large (Ar²⁺ to Ar⁷⁺), CHIANTI takes the excitations to
        # n = 4, 5, Bao and Li to n ≤ 15 (their Fig. 1), and the thresholds of the channels
        # differ: from 3 to 10 times the threshold ±40 %, where CHIANTI lies at 0.68 to 1.35.
        # Closer to the threshold the onsets differ: Ar⁷⁺ at 1.5 and 2 I is 1.79 and 0.42.
        # The two- and one-electron ions Ar¹⁶⁺ and Ar¹⁷⁺, for which Dere's text takes the
        # relativistic fits of Fontes et al. (1999) instead of the files of the table, rise
        # from 0.97 at 1.5 I to 1.24 at 5 I and 1.35 and 1.38 at 10 I (beyond their figures the
        # fit has a negative Bethe coefficient A₀ and is not usable): tested to 5 I.
        for Z0 in 1:17
            fit = CHIANTI_IONIZATION_FITS["Ar"][Z0 + 1]
            I = minimum(s.threshold for s in fit.direct)
            autoionizing = 2 <= Z0 <= 7
            for k in (autoionizing ? (3, 5, 10) :
                      Z0 >= 16 ? (1.5, 2, 3, 5) : (1.5, 2, 3, 5, 10))
                ratio = ionization_cross_section("Ar", Z0, momentum(k * I)) /
                        bao_li(Z0, k * I)
                @test abs(ratio - 1) < (autoionizing ? 0.4 : 0.3)
            end
        end
        # the excitation-autoionization is needed: Ar⁷⁺ at 3 I, the direct ionization alone is
        # 0.33 of Bao and Li, with the autoionization 1.13
        fit = CHIANTI_IONIZATION_FITS["Ar"][8]
        direct_only = ChiantiIonization(
            fit.ion, fit.direct, fit.autoionization_scale, Runes2P.ChiantiAutoionization[])
        E = 3 * fit.direct[1].threshold
        @test chianti_cross_section(direct_only, momentum(E)) / bao_li(7, E) < 0.4
        @test 0.9 < chianti_cross_section(fit, momentum(E)) / bao_li(7, E) < 1.3
    end
end
