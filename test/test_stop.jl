# Clean stop of a segment: the stop condition, a run in segments that
# stop on the wall-time budget and on a stop file, and the exit status for the batch
# script. The cases are those of test_checkpoint.jl.
using HDF5

# an interval update that creates a file at the time points from `time` on, as the batch
# script does when the warning signal of the batch system arrives
struct RequestStop <: IntervalUpdate
    file::String
    time::Float64
end
Runes2P.update_name(::RequestStop) = "request_stop"
Runes2P.apply_update!(update::RequestStop, u, layout, t) = (t >= update.time && touch(update.file); false)
Runes2P.update_state(::RequestStop) = [0.0]
Runes2P.restore_update!(update::RequestStop, state) = update

@testset "clean stop" begin
    @testset "stop condition" begin
        withenv("SLURM_JOB_END_TIME" => nothing) do
            stop = StopCondition()
            @test stop.end_time == Inf
            @test stop.stop_file == ""
            @test stop.slack == 300.0                 # the margin of GS2
            @test stop_reason(stop, 1e9, 1e9) === nothing
            before = time()
            stop = StopCondition(wall_budget=100, slack=5)
            @test before + 100 <= stop.end_time <= time() + 100
            @test StopCondition(end_time=50.0, wall_budget=100).end_time == 50.0
        end
        withenv("SLURM_JOB_END_TIME" => "1800000000") do
            @test StopCondition().end_time == 1800000000.0
            @test StopCondition(end_time=12.0).end_time == 12.0          # the argument counts
            @test StopCondition(wall_budget=100).end_time <= time() + 100  # the earlier end
        end
        withenv("SLURM_JOB_END_TIME" => "not a number") do
            @test StopCondition().end_time == Inf
        end
        @test_throws ArgumentError StopCondition(slack=-1)
        @test_throws ArgumentError StopCondition(wall_budget=0)

        # the margin: the longest interval, the longest write and the slack
        stop = StopCondition(end_time=1000.0, slack=10)
        @test stop_reason(stop, 50, 20; now=919.9) === nothing
        @test stop_reason(stop, 50, 20; now=920.0) == "wall time"
        @test stop_reason(stop, 0, 0; now=989.9) === nothing
        @test stop_reason(stop, 0, 0; now=990.0) == "wall time"
        @test stop_reason(nothing, 50, 20) === nothing

        file = tempname()
        stop = StopCondition(end_time=1000.0, stop_file=file, slack=10)
        @test stop_reason(stop, 0, 0; now=0.0) === nothing
        touch(file)
        @test stop_reason(stop, 0, 0; now=0.0) == "stop file"
        @test stop_reason(stop, 50, 20; now=990.0) == "stop file"
        rm(file)
        @test describe(stop) == Dict{String,Any}("type" => "StopCondition", "end_time" => 1000.0,
                                                 "stop_file" => file, "slack" => 10.0)
        @test (Runes2P.EXIT_FINISHED, Runes2P.EXIT_STOPPED, Runes2P.EXIT_STOPPED_WITHOUT_CHECKPOINT) == (0, 85, 86)
    end

    @testset "a run in segments that stop for time" begin
        case = _checkpoint_case()
        whole = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),))
        # a budget that is spent at once: every segment does one interval and stops
        checkpoint = CheckpointSettings("budget"; directory=_checkpoint_directory(), interval=1e6)
        segments = 0
        solution = nothing
        while true
            case = _checkpoint_case()
            solution = @test_logs integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                            updates=(case.update(),), checkpoint, resume=true,
                                            stop=StopCondition(wall_budget=1e-9, slack=0))
            segments += 1
            record = solution.settings["segment"]
            @test record["index"] == segments
            @test record["intervals"] == 1
            @test record["end_time"] == case.times[segments+1]
            @test record["checkpoint_written"] === true
            @test read_checkpoint(checkpoint).time == case.times[segments+1]
            @test read_settings(Runes2P.segment_file(checkpoint, segments))["segment"] == record
            if run_status(solution) === :finished
                @test record["status"] == "finished"
                @test !haskey(record, "stop_reason")
                @test exit_status(solution) == 0
                break
            end
            @test run_status(solution) === :stopped
            @test record["stop_reason"] == "wall time"
            @test exit_status(solution) == 85
            segments > 20 && break
        end
        @test segments == length(case.times) - 1
        joined = read_output(checkpoint)
        @test joined.times == whole.times
        @test joined.states == whole.states
        @test joined.update_states == whole.update_states
        @test _same_statistics(joined.statistics, whole.statistics)
        # the setup was the same in every segment
        @test solution.settings["segment"]["settings_changed"] === false
        @test solution.settings["setup_hash"] == whole.settings["setup_hash"]
        @test solution.settings["stop"]["type"] == "StopCondition"

        # a budget that lasts: the run finishes in one segment, with one checkpoint at the end
        case = _checkpoint_case()
        checkpoint = CheckpointSettings("enough"; directory=_checkpoint_directory(), interval=1e6)
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),),
                             checkpoint, resume=true, stop=StopCondition(wall_budget=1e6, slack=0))
        @test run_status(solution) === :finished
        @test solution.settings["segment"]["intervals"] == 8
        @test solution.states == whole.states
        @test !isfile(Runes2P.previous_checkpoint_file(checkpoint))
    end

    @testset "stop file" begin
        case = _checkpoint_case()
        whole = integrate(case.system, case.u0, case.times; save_times=case.save_times, updates=(case.update(),))
        directory = _checkpoint_directory()
        checkpoint = CheckpointSettings("file"; directory, interval=1e6)
        stop_file = joinpath(directory, "file.stop")
        stop() = StopCondition(end_time=Inf, stop_file=stop_file)
        # a stop file left by an earlier segment is removed and does not stop this one. The
        # request comes at the fourth time point, and the segment stops at the end of the
        # interval that starts there.
        touch(stop_file)
        case = _checkpoint_case()
        first_segment = integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                  updates=(case.update(), RequestStop(stop_file, case.times[4])), checkpoint,
                                  resume=true, stop=stop())
        @test run_status(first_segment) === :stopped
        @test first_segment.settings["segment"]["stop_reason"] == "stop file"
        @test first_segment.settings["segment"]["end_time"] == case.times[5]
        @test first_segment.settings["segment"]["intervals"] == 4
        @test first_segment.times == filter(<=(case.times[5]), case.save_times)
        @test exit_status(first_segment) == 85
        @test read_checkpoint(checkpoint).time == case.times[5]
        @test isfile(stop_file)

        case = _checkpoint_case()
        second_segment = integrate(case.system, case.u0, case.times; save_times=case.save_times,
                                   updates=(case.update(), RequestStop(stop_file, Inf)), checkpoint, resume=true,
                                   stop=stop())
        @test !isfile(stop_file)
        @test run_status(second_segment) === :finished
        @test second_segment.settings["segment"]["index"] == 2
        joined = read_output(checkpoint)
        @test joined.states == whole.states
        @test joined.update_states["knockon_cutoff"] == whole.update_states["knockon_cutoff"]

        # a request in the last interval does not stop the run: it ends there anyway
        case = _checkpoint_case()
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times,
                             updates=(case.update(), RequestStop(stop_file, case.times[end-1])), stop=stop())
        @test run_status(solution) === :finished
        @test solution.states == whole.states
        rm(stop_file)

        # without checkpoints a stopped segment has nothing to continue from
        case = _checkpoint_case()
        solution = integrate(case.system, case.u0, case.times; save_times=case.save_times,
                             updates=(case.update(), RequestStop(stop_file, case.times[2])), stop=stop())
        @test run_status(solution) === :stopped
        @test solution.settings["segment"]["end_time"] == case.times[3]
        @test solution.settings["segment"]["checkpoint_written"] === false
        @test exit_status(solution) == 86
        @test solution.states == whole.states[:, 1:length(solution.times)]
        rm(stop_file)
    end
end
