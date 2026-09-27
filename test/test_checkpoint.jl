# Knock-on cutoff that follows the plasma during a run, checkpoints, segments and resume.
# The reference of every resume test is the uninterrupted run of the same setup: a resumed
# run must repeat it, bit for bit where the solver carries nothing from step to step.
using HDF5

# a plasma that cools while the field rises, so that the window of the knock-on cutoff
# moves down: hotter and with a weaker field at the second radius
function _checkpoint_plasma(t; nr=2, t_end=8e-4)
    temperature = [1000.0, 2000.0][1:nr] .* (0.05)^(t / t_end)
    field = [1.0, 0.8][1:nr] .* (1 + 5 * t / t_end)
    return PlasmaState(cold_electron_temperature=temperature, cold_electron_density=fill(1e20, nr),
                       ions=[IonSpecies("D", 1, 1, fill(1e20, nr))], electric_field=field,
                       magnetic_field=fill(5.0, nr))
end

# Fokker-Planck term with the prescribed plasma, radial diffusion, a knock-on term whose
# cutoff follows the plasma, and a density moment as a second unknown
function _checkpoint_case(; treatment=:explicit, with_density=true)
    nr = 2
    mom = MomentumGrid(uniform_p_faces(12, 0.0, 1.2), uniform_xi_faces(4))
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 1.0)))
    plasma_at(t) = _checkpoint_plasma(t; nr)
    op = AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero)
    radial = AdvectionDiffusion(grid; scheme=:central, r_boundary=:absorbing)
    radial.diffusion_r_r .= 50.0
    knockon = KnockOnOperator(grid, grid, 0.4)
    terms = (PrescribedFokkerPlanckTerm(:f, op, plasma_at; settings=CollisionSettings(mode=:full)),
             FokkerPlanckTerm(:f, radial),
             KnockOnTerm(:f, knockon; target_density=fill(1e20, nr), treatment))
    unknowns = Any[KineticUnknown(:f, grid; scale=1e20)]
    equations = Any[Equation(:f, TimeDerivative(), terms...)]
    if with_density
        push!(unknowns, RadialUnknown(:hot_density; scale=1e20))
        push!(equations, Equation(:hot_density, Algebraic(), DensityMomentTerm(:f),
                                  LinearTerm(:hot_density; coefficient=-1.0)))
    end
    layout = UnknownLayout(grid.radial_grid, unknowns)
    system = EquationSystem(layout, equations)
    theta = 1000.0 / Constants.ELECTRON_REST_ENERGY
    f0 = [1e20 * exp(-(sqrt(1 + p^2) - 1) / theta) for p in mom.p_centers, _ in 1:4, _ in 1:nr]
    u0 = zeros(length(layout))
    for ir in 1:nr
        unknown_view(u0, layout, :f, ir) .= f0[:, :, ir]
        with_density && (unknown_view(u0, layout, :hot_density, ir) .= density(grid, f0)[ir])
    end
    times = [0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4]
    save_times = [0.0, 1.5e-4, 2e-4, 4e-4, 4.5e-4, 7e-4, 8e-4]   # at time points and between them
    update() = KnockOnCutoffUpdate(knockon, plasma_at)
    return (; grid, mom, system, layout, u0, times, save_times, plasma_at, knockon, update)
end

# the fields of the statistics that do not hold wall times
const _UNTIMED_STATISTICS = [field for field in fieldnames(SolverStatistics) if !endswith(String(field), "time") ||
                                                                               field === :time]

function _same_statistics(a::SolverStatistics, b::SolverStatistics)
    return all(field -> isequal(getfield(a, field), getfield(b, field)), _UNTIMED_STATISTICS)
end

_checkpoint_directory() = mktempdir()

@testset "checkpoint and resume" begin
    @testset "knock-on cutoff update" begin
        case = _checkpoint_case()
        update = case.update()
        @test Runes2P.update_name(update) == "knockon_cutoff"
        @test Runes2P.update_state(update) == [0.4, 0.4]
        faces = case.mom.p_faces
        seen = Set{Float64}()
        for t in case.times
            before = copy(case.knockon.p_cutoff)
            changed = Runes2P.apply_update!(update, case.u0, case.layout, t)
            plasma = case.plasma_at(t)
            for ir in 1:2
                window = cutoff_window(thermal_momentum(plasma.cold_electron_temperature[ir]),
                                       Runes2P.free_critical_momentum(plasma, ir))
                expected = choose_cutoff(faces, window.lower, window.upper; current_cutoff=before[ir])
                @test case.knockon.p_cutoff[ir] == expected.p_cutoff
                @test update.inside_window[ir] == expected.inside_window
                @test case.knockon.p_cutoff[ir] in faces
                # a cutoff inside its window is kept
                window.lower <= before[ir] < window.upper && @test case.knockon.p_cutoff[ir] == before[ir]
                push!(seen, case.knockon.p_cutoff[ir])
            end
            @test changed == (case.knockon.p_cutoff != before)
            # applying it again at the same time changes nothing
            @test Runes2P.apply_update!(update, case.u0, case.layout, t) === false
        end
        @test length(seen) >= 3          # the cutoff moved during the sequence
        @test update.rebuilds[] >= 2
        @test Runes2P.update_state(update) == case.knockon.p_cutoff

        # DREAM's free critical field: the cold electrons with the relativistic Coulomb logarithm
        plasma = case.plasma_at(0.0)
        critical = critical_field(1e20, relativistic_coulomb_logarithm(1000.0, 1e20))
        @test Runes2P.free_critical_momentum(plasma, 1) == critical_momentum(1.0, critical)
        @test Runes2P.free_critical_momentum(plasma, 1) ≈ 1 / sqrt(1.0 / critical - 1) rtol = 1e-14

        # restoring a state sets the operator, and the kernels are those of a fresh operator
        state = Runes2P.update_state(update)
        other = _checkpoint_case()
        restored = other.update()
        Runes2P.restore_update!(restored, state)
        @test other.knockon.p_cutoff == state
        fresh = KnockOnOperator(other.grid, other.grid, state)
        @test energy_kernel(other.knockon, 1).rates == energy_kernel(fresh, 1).rates
        @test_throws DimensionMismatch Runes2P.restore_update!(restored, [0.1])

        record = describe(update, DescriptionContext(times=case.times))
        @test record["type"] == "KnockOnCutoffUpdate"
        @test record["name"] == "knockon_cutoff"
        @test record["thermal_factor"] == 4.0
        @test record["critical_momentum"] == "free_critical_momentum"
        @test record["plasma"] == "plasma_at"
        @test record["plasma_table"] == "table_1"
        fields = settings_fields(KnockOnCutoffUpdate)
        @test Set(union(fields.described, fields.derived)) == Set(fieldnames(KnockOnCutoffUpdate))

        @test_throws ArgumentError KnockOnCutoffUpdate(case.knockon, case.plasma_at; thermal_factor=0)
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; updates=(update, case.update()))
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; updates=(1,))
    end

    @testset "a run with the cutoff update" begin
        case = _checkpoint_case()
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),))
        cutoffs = solution.update_states["knockon_cutoff"]
        @test size(cutoffs) == (2, length(case.save_times))
        @test length(unique(cutoffs)) >= 3
        # the state of a saved time is the cutoff of the interval that computed it
        reference = _checkpoint_case()
        update = reference.update()
        in_force = Dict{Float64,Vector{Float64}}()
        for (k, t) in enumerate(reference.times[1:end-1])
            Runes2P.apply_update!(update, reference.u0, reference.layout, t)
            k == 1 && (in_force[t] = copy(reference.knockon.p_cutoff))
            for s in reference.save_times
                t < s <= reference.times[k+1] && (in_force[s] = copy(reference.knockon.p_cutoff))
            end
        end
        for (k, s) in enumerate(case.save_times)
            @test cutoffs[:, k] == in_force[s]
        end
        # the record describes the setup as built: the cutoff the operator was given
        operator = solution.settings["system"]["equations"]["f"]["terms"]["term_3"]["operator"]
        @test operator["p_cutoff"] == [0.4, 0.4]
        @test solution.settings["updates"]["knockon_cutoff"]["type"] == "KnockOnCutoffUpdate"
        @test collect(keys(solution.settings["prescribed"])) == ["table_1"]   # one table for the one function
        @test solution.settings["segment"] ==
              Dict{String,Any}("index" => 1, "start_time" => 0.0, "resumed" => false, "saved_before" => 0,
                               "status" => "finished", "end_time" => 8e-4, "intervals" => 8,
                               "checkpoint_written" => false)
        @test solution.settings["stop"] == Dict{String,Any}("type" => "none")
        @test run_status(solution) === :finished
        @test exit_status(solution) == 0
        @test solution.settings["checkpoint"] == Dict{String,Any}("type" => "none")
        # with a fixed cutoff the run differs: the update changes the result
        fixed = _checkpoint_case()
        other = integrate(fixed.system, fixed.u0, fixed.times; save_times=fixed.save_times)
        @test other.states[:, end] != solution.states[:, end]
        @test isempty(other.update_states)
    end

    @testset "atomic write" begin
        directory = _checkpoint_directory()
        path, previous = joinpath(directory, "file.txt"), joinpath(directory, "file.prev.txt")
        @test Runes2P.write_atomically(temporary -> write(temporary, "first"), path; previous) == path
        @test read(path, String) == "first"
        @test !isfile(previous)
        Runes2P.write_atomically(temporary -> write(temporary, "second"), path; previous)
        @test read(path, String) == "second"
        @test read(previous, String) == "first"
        # a write that dies leaves both files as they were and no temporary file
        @test_throws ErrorException Runes2P.write_atomically(path; previous) do temporary
            write(temporary, "third, incomp")
            error("killed")
        end
        @test read(path, String) == "second"
        @test read(previous, String) == "first"
        @test sort(readdir(directory)) == ["file.prev.txt", "file.txt"]
        Runes2P.write_atomically(temporary -> write(temporary, "alone"), joinpath(directory, "other.txt"))
        @test read(joinpath(directory, "other.txt"), String) == "alone"
    end

    @testset "checkpoint settings and files" begin
        settings = CheckpointSettings("run"; directory="/scratch/case", interval=600)
        @test Runes2P.checkpoint_file(settings) == "/scratch/case/run.checkpoint.h5"
        @test Runes2P.previous_checkpoint_file(settings) == "/scratch/case/run.checkpoint.prev.h5"
        @test Runes2P.segment_file(settings, 3) == "/scratch/case/run.segment0003.h5"
        @test settings.interval == 600.0
        @test CheckpointSettings("run").interval == 1800.0
        @test_throws ArgumentError CheckpointSettings("")
        @test_throws ArgumentError CheckpointSettings("a/b")
        @test_throws ArgumentError CheckpointSettings("run"; interval=-1)
        @test isempty(Runes2P.segment_files(settings))
        @test describe(settings) == Dict{String,Any}("type" => "CheckpointSettings", "name" => "run",
                                                     "directory" => "/scratch/case", "interval" => 600.0,
                                                     "overwrite" => false)
        case = _checkpoint_case()
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; resume=true)
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; resume=1)
    end

    @testset "a resumed run repeats the uninterrupted one" begin
        solvers = [("direct", NewtonSolver(), :explicit, true),
                   ("GMRES, every iteration",
                    NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:every_iteration)), :implicit, true),
                   ("GMRES, every step",
                    NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:every_step)), :implicit, true),
                   ("GMRES, on demand",
                    NewtonSolver(linear_solver=GMRES(preconditioner_refresh=:on_demand)), :implicit, false)]
        @testset "$name" for (name, solver, treatment, exact) in solvers
            case = _checkpoint_case(; treatment)
            whole = integrate(case.system, case.u0, case.times; solver, save_times=case.save_times,
                              updates=(case.update(),))

            # first segment: to the fifth time point, then the run from its checkpoint
            case = _checkpoint_case(; treatment)
            checkpoint = CheckpointSettings("run"; directory=_checkpoint_directory(), interval=0)
            stop = 5
            first_saves = filter(<=(case.times[stop]), case.save_times)
            first_segment = integrate(case.system, case.u0, case.times[1:stop]; solver, save_times=first_saves,
                                      updates=(case.update(),), checkpoint)
            @test first_segment.states == whole.states[:, 1:length(first_saves)]
            @test isfile(Runes2P.checkpoint_file(checkpoint))
            @test isfile(Runes2P.previous_checkpoint_file(checkpoint))
            stored = read_checkpoint(checkpoint)
            @test stored.time == case.times[stop]
            @test stored.segment == 1
            @test stored.saved_count == length(first_saves)
            @test stored.state == first_segment.states[:, end] ||
                  case.times[stop] != last(first_saves)       # the last state is saved when it is a save time
            @test stored.update_states["knockon_cutoff"] == first_segment.update_states["knockon_cutoff"][:, end]
            @test stored.summary["accepted_steps"] == stop - 1
            @test stored.summary["newton_iterations"] == sum(first_segment.statistics.newton_iterations)
            @test unknown_names(stored.layout) == unknown_names(case.layout)
            @test read_checkpoint(Runes2P.previous_checkpoint_file(checkpoint)).time == case.times[stop-1]

            # the script of the second segment builds the setup again
            case = _checkpoint_case(; treatment)
            second_segment = @test_logs (:warn, r"changed settings") integrate(
                case.system, case.u0, case.times; solver, save_times=case.save_times, updates=(case.update(),),
                checkpoint, resume=true)
            later = length(first_saves)+1:length(case.save_times)
            @test second_segment.times == case.save_times[later]
            if exact
                @test second_segment.states == whole.states[:, later]
            else
                @test second_segment.states ≈ whole.states[:, later] rtol = 1e-8
            end
            @test second_segment.update_states["knockon_cutoff"] == whole.update_states["knockon_cutoff"][:, later]
            record = second_segment.settings["segment"]
            @test record["index"] == 2
            @test record["resumed"] === true
            @test record["start_time"] == case.times[stop]
            @test record["saved_before"] == length(first_saves)
            @test record["resumed_from"] == Runes2P.checkpoint_file(checkpoint)
            @test record["commit_changed"] === false
            @test record["previous_setup_hash"] == first_segment.settings["setup_hash"]
            @test record["changed_sections"] == ["prescribed", "time"]   # the first segment had fewer time points

            # the segments joined are the output of the uninterrupted run
            @test Runes2P.segment_files(checkpoint) == [Runes2P.segment_file(checkpoint, 1), Runes2P.segment_file(checkpoint, 2)]
            joined = read_output(checkpoint)
            @test joined.times == whole.times
            @test unknown_names(joined.layout) == unknown_names(whole.layout)
            if exact
                @test joined.states == whole.states
                @test _same_statistics(joined.statistics, whole.statistics)
            else
                @test joined.states ≈ whole.states rtol = 1e-8
                @test joined.statistics.time == whole.statistics.time
            end
            @test joined.update_states == whole.update_states
            @test joined.settings["segment"]["index"] == 2
            # the checkpoint at the end of the run
            final = read_checkpoint(checkpoint)
            @test final.time == last(case.times)
            @test final.segment == 2
            @test final.saved_count == length(case.save_times)
            @test final.summary["accepted_steps"] == length(case.times) - 1
        end
    end

    @testset "a segment killed after its last checkpoint" begin
        # The output is written before the checkpoint. A segment killed between the two
        # leaves an output that reaches beyond the last checkpoint: here the checkpoint at
        # the sixth time point is removed, so the one before it is the last.
        case = _checkpoint_case()
        whole = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),))
        case = _checkpoint_case()
        checkpoint = CheckpointSettings("killed"; directory=_checkpoint_directory(), interval=0)
        integrate(case.system, case.u0, case.times[1:6]; save_times=filter(<=(case.times[6]), case.save_times),
                  updates=(case.update(),), checkpoint)
        rm(Runes2P.checkpoint_file(checkpoint))
        stored = read_checkpoint(checkpoint)
        @test stored.time == case.times[5]
        @test stored.source == Runes2P.previous_checkpoint_file(checkpoint)

        # a checkpoint that cannot be read is passed over as well
        write(Runes2P.checkpoint_file(checkpoint), "not an HDF5 file")
        stored = @test_logs (:warn, r"cannot be read") read_checkpoint(checkpoint)
        @test stored.time == case.times[5]

        case = _checkpoint_case()
        second = @test_logs (:warn, r"cannot be read") (:warn, r"changed settings") integrate(
            case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),), checkpoint,
            resume=true)
        @test second.settings["segment"]["start_time"] == case.times[5]
        joined = read_output(checkpoint)
        @test joined.times == whole.times          # no saved time twice, none missing
        @test joined.states == whole.states
        @test joined.update_states == whole.update_states
        @test _same_statistics(joined.statistics, whole.statistics)
        # and with neither checkpoint the run starts from its initial state
        empty_run = CheckpointSettings("empty"; directory=_checkpoint_directory(), interval=0)
        @test read_checkpoint(empty_run) === nothing
        case = _checkpoint_case()
        fresh = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),),
                          checkpoint=empty_run, resume=true)
        @test fresh.states == whole.states
        @test fresh.settings["segment"]["index"] == 1
        @test fresh.settings["segment"]["resumed"] === false
    end

    @testset "a segment killed between its output and its checkpoint" begin
        # The attempt leaves the file of its segment, and the checkpoint names the segment
        # before. The run submitted again replaces the file.
        case = _checkpoint_case()
        whole = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),))
        case = _checkpoint_case()
        checkpoint = CheckpointSettings("attempt"; directory=_checkpoint_directory(), interval=1e6)
        integrate(case.system, case.u0, case.times[1:5]; save_times=filter(<=(case.times[5]), case.save_times),
                  updates=(case.update(),), checkpoint)
        cp(Runes2P.segment_file(checkpoint, 1), Runes2P.segment_file(checkpoint, 2))
        case = _checkpoint_case()
        second = @test_logs (:warn, r"changed settings") integrate(
            case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),), checkpoint,
            resume=true)
        @test second.settings["segment"]["index"] == 2
        joined = read_output(checkpoint)
        @test joined.times == whole.times
        @test joined.states == whole.states
        # the same in the first segment of a run, which has no checkpoint yet
        first_run = CheckpointSettings("first"; directory=_checkpoint_directory(), interval=1e6)
        mkpath(first_run.directory)
        cp(Runes2P.segment_file(checkpoint, 1), Runes2P.segment_file(first_run, 1))
        case = _checkpoint_case()
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),),
                             checkpoint=first_run, resume=true)
        @test solution.settings["segment"]["index"] == 1
        @test read_output(first_run).states == whole.states
        # without resume = true the file is another run's: a run from its start, and a run
        # from a stored state given by hand, which would write the second segment
        case = _checkpoint_case()
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; checkpoint=first_run)
        stored = read_checkpoint(Runes2P.previous_checkpoint_file(checkpoint))
        @test stored.segment == 1
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                             updates=(case.update(),), checkpoint, resume=stored)
    end

    @testset "checkpoint interval and segment files" begin
        case = _checkpoint_case()
        directory = _checkpoint_directory()
        # an interval no run of the test reaches: one checkpoint, at the end
        checkpoint = CheckpointSettings("long"; directory, interval=1e6)
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times, checkpoint)
        @test isfile(Runes2P.checkpoint_file(checkpoint))
        @test !isfile(Runes2P.previous_checkpoint_file(checkpoint))
        @test read_checkpoint(checkpoint).time == last(case.times)
        @test solution.settings["checkpoint"]["name"] == "long"
        joined = read_output(checkpoint)
        @test joined.states == solution.states
        @test joined.settings == read_settings(Runes2P.segment_file(checkpoint, 1))
        @test isempty(filter(name -> endswith(name, ".tmp"), readdir(directory)))

        # a run does not replace the segment file of another
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; checkpoint)
        again = integrate(case.system, case.u0, case.times; save_times=case.save_times,
                          checkpoint=CheckpointSettings("long"; directory, interval=1e6, overwrite=true))
        @test again.states == solution.states

        # the settings of a resumed run that are the same are not reported: the state before
        # the last of a run with the full time points
        case = _checkpoint_case()
        checkpoint = CheckpointSettings("every"; directory, interval=0)
        whole = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),),
                          checkpoint)
        stored = read_checkpoint(Runes2P.previous_checkpoint_file(checkpoint))
        @test stored.time == case.times[end-1]
        case = _checkpoint_case()
        last_step = @test_logs integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                         updates=(case.update(),), resume=stored)
        @test last_step.settings["segment"]["settings_changed"] === false
        @test !haskey(last_step.settings["segment"], "changed_sections")
        @test last_step.settings["setup_hash"] == whole.settings["setup_hash"]
        @test last_step.states == whole.states[:, end:end]
        @test last_step.times == [last(case.times)]

        # a setup used by a run before is another setup: its operator has the cutoff that run
        # left, and the record says so
        used = @test_logs (:warn, r"changed settings") integrate(
            case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),), resume=stored)
        @test used.settings["segment"]["changed_sections"] == ["system", "updates"]
        @test used.states == last_step.states       # the stored cutoff is restored all the same

        # a state at the end of the run leaves nothing to do
        case = _checkpoint_case()
        done = @test_logs integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                    updates=(case.update(),), resume=read_checkpoint(checkpoint))
        @test isempty(done.times)
        @test size(done.states) == (length(case.layout), 0)
        # a solution without a saved state is written and read back as such
        file = tempname() * ".h5"
        write_output(file, done)
        back = read_output(file)
        @test isempty(back.times)
        @test size(back.states) == (length(case.layout), 0)
        @test isempty(back.update_states)
        @test back.settings == done.settings
        @test_throws ArgumentError read_state(file)
        rm(file)
    end

    @testset "checkpoints before the first saved state" begin
        # Many time points and few saved states, the pattern of a production run: a
        # checkpoint comes before the segment has saved anything, in a run from its start
        # and in a resumed one.
        save_times = [3.5e-4, 8e-4]
        case = _checkpoint_case()
        whole = integrate(case.system, case.u0, case.times; save_times, updates=(case.update(),))
        case = _checkpoint_case()
        checkpoint = CheckpointSettings("sparse"; directory=_checkpoint_directory(), interval=0)
        first_segment = integrate(case.system, case.u0, case.times[1:3]; save_times=Float64[],
                                  updates=(case.update(),), checkpoint)
        @test isempty(first_segment.times)
        @test read_checkpoint(checkpoint).saved_count == 0
        only_statistics = read_output(Runes2P.segment_file(checkpoint, 1))
        @test isempty(only_statistics.times)
        @test only_statistics.statistics.time == case.times[2:3]
        h5open(Runes2P.segment_file(checkpoint, 1)) do h5
            @test sort(collect(keys(h5))) == ["layout", "settings", "solver_statistics", "update_states"]
        end
        case = _checkpoint_case()
        second_segment = @test_logs (:warn, r"changed settings") integrate(
            case.system, case.u0, case.times; save_times, updates=(case.update(),), checkpoint, resume=true)
        @test second_segment.times == save_times
        @test second_segment.states == whole.states
        @test second_segment.settings["segment"]["saved_before"] == 0
        joined = read_output(checkpoint)
        @test joined.times == whole.times
        @test joined.states == whole.states
        @test joined.update_states == whole.update_states
        @test _same_statistics(joined.statistics, whole.statistics)
        # the run in one segment, with a checkpoint at every time point
        case = _checkpoint_case()
        single = CheckpointSettings("sparse-single"; directory=_checkpoint_directory(), interval=0)
        solution = integrate(case.system, case.u0, case.times; save_times, updates=(case.update(),),
                             checkpoint=single)
        @test solution.states == whole.states
        @test read_output(single).states == whole.states
        # only segments without a saved state
        @test isempty(read_output(Runes2P.segment_file(checkpoint, 1), Runes2P.segment_file(checkpoint, 1)).times)
    end

    @testset "resume from an output file" begin
        case = _checkpoint_case()
        whole = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),))
        file = tempname() * ".h5"
        write_output(file, whole)
        h5open(file) do h5
            @test read(h5["update_states/knockon_cutoff"]) == whole.update_states["knockon_cutoff"]
        end
        # the fourth saved state is at the fifth time point
        stored = read_state(file; time_index=4)
        @test stored.time == case.save_times[4] == case.times[5]
        @test stored.state == whole.states[:, 4]
        @test stored.update_states["knockon_cutoff"] == whole.update_states["knockon_cutoff"][:, 4]
        @test stored.saved_count == 4
        @test stored.segment == 1
        case = _checkpoint_case()
        resumed = @test_logs integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                       updates=(case.update(),), resume=stored)
        @test resumed.times == case.save_times[5:end]
        @test resumed.states == whole.states[:, 5:end]
        @test resumed.update_states["knockon_cutoff"] == whole.update_states["knockon_cutoff"][:, 5:end]
        @test resumed.settings["segment"]["index"] == 2
        @test resumed.settings["segment"]["saved_before"] == 4

        # the last saved state by default, by file name
        @test read_state(file).time == last(case.times)
        case = _checkpoint_case()
        done = @test_logs integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                    updates=(case.update(),), resume=file)
        @test isempty(done.times)
        # other save times are a changed setting
        case = _checkpoint_case()
        done = @test_logs (:warn, r"changed settings") integrate(case.system, case.u0, case.times;
                                                                 updates=(case.update(),), resume=file)
        @test done.settings["segment"]["changed_sections"] == ["time"]
        @test_throws ArgumentError read_state(file; time_index=8)

        # a saved state between two time points is not a state to resume from
        between = read_state(file; time_index=2)
        @test between.time == 1.5e-4
        @test_throws ArgumentError integrate(case.system, case.u0, case.times; resume=between)

        # one file read back is the solution written
        back = read_output(file)
        @test back.times == whole.times
        @test back.states == whole.states
        @test back.update_states == whole.update_states
        @test _same_statistics(back.statistics, whole.statistics)
        @test back.statistics.wall_time == whole.statistics.wall_time
        @test back.settings == whole.settings
        @test_throws ArgumentError read_output()
        rm(file)
    end

    @testset "changed setups" begin
        directory = _checkpoint_directory()
        case = _checkpoint_case(; with_density=false)
        checkpoint = CheckpointSettings("kinetic"; directory, interval=0)
        integrate(case.system, case.u0, case.times[1:5]; checkpoint)
        stored = read_checkpoint(checkpoint)

        # an unknown the stored state does not have starts from the setup's initial state
        extended = _checkpoint_case(; with_density=true)
        u0 = copy(extended.u0)
        for ir in 1:2
            unknown_view(u0, extended.layout, :hot_density, ir) .= 7.0
            unknown_view(u0, extended.layout, :f, ir) .= 0.0      # replaced by the stored values
        end
        solution = @test_logs (:warn, r"changed settings") integrate(extended.system, u0, extended.times[5:6];
                                                                     resume=stored)
        reference = _checkpoint_case(; with_density=false)
        plain = @test_logs (:warn, r"changed settings") integrate(reference.system, reference.u0,
                                                                  reference.times[5:6]; resume=stored)
        final = solution.states[:, end]
        distribution = stack(collect(unknown_view(final, extended.layout, :f, ir)) for ir in 1:2)
        for ir in 1:2
            # the same distribution from a larger linear system: equal to round-off
            @test unknown_view(final, extended.layout, :f, ir) ≈
                  unknown_view(plain.states[:, end], reference.layout, :f, ir) rtol = 1e-10
            # the added unknown is determined by its equation, not by its initial value
            @test only(unknown_view(final, extended.layout, :hot_density, ir)) ≈
                  density(extended.grid, distribution)[ir] rtol = 1e-10
        end
        @test "layout" in solution.settings["segment"]["changed_sections"]
        @test "system" in solution.settings["segment"]["changed_sections"]

        # a changed solver setting is one section
        case = _checkpoint_case(; with_density=false)
        changed = @test_logs (:warn, r"changed settings") integrate(
            case.system, case.u0, case.times[1:6]; solver=NewtonSolver(max_iterations=29),
            resume=read_checkpoint(Runes2P.previous_checkpoint_file(checkpoint)))
        @test changed.settings["segment"]["changed_sections"] == ["prescribed", "solver", "time"]

        # another momentum grid is refused until the conservative remap exists
        mom = MomentumGrid(uniform_p_faces(10, 0.0, 1.2), uniform_xi_faces(4))
        grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
        layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=1e20)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), LinearTerm(:f; coefficient=-1.0))])
        @test_throws Union{ArgumentError,DimensionMismatch} integrate(system, ones(length(layout)), case.times[5:6];
                                                                      resume=stored)

        # files of before the numbering hold no state to resume from
        old = tempname() * ".h5"
        h5open(old, "w") do h5
            attributes(h5)["format_version"] = 1
        end
        @test_throws ArgumentError read_state(old)
        rm(old)
    end
end
