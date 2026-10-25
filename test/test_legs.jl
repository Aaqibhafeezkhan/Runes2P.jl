# Legs of a run: the time points and the save times from one arithmetic, the keyword forms
# of Leg and their errors, and the rule that a save time of a prescribed-step run is a
# time point.

# a dt schedule of ten legs for an injection cascade: 5 500 steps to 1 ms
function growing_dt_legs()
    [Leg(1e-9; steps = 1000, saves = 1), Leg(3e-9; steps = 1000, saves = 3),
        Leg(7e-9; steps = 1000, saves = 7),
        Leg(1e-8; steps = 1000, saves = 10), Leg(1e-8; steps = 100, saves = 10), Leg(
            1e-7; steps = 100, saves = 10),
        Leg(1e-6; steps = 100, saves = 10), Leg(1e-5; steps = 100, saves = 10), Leg(
            1e-4; steps = 100, saves = 10),
        Leg(1e-3 - 1.11131e-4; steps = 1000, saves = 10)]
end

@testset "legs of a run" begin
    @testset "forms and errors" begin
        leg = Leg(1e-9; steps = 1000, saves = 1)
        @test (leg.duration, leg.steps, leg.saves) == (1e-9, 1000, 1)
        leg = Leg(3e-9; dt = 3e-12, save_dt = 1e-9)
        @test (leg.steps, leg.saves) == (1000, 3)
        leg = Leg(1e-3; save_dt = 1e-5)
        @test (leg.steps, leg.saves) == (0, 100)
        @test_throws ArgumentError Leg(1e-9, 1000, 1)                    # positional
        @test_throws ArgumentError Leg(1e-9; steps = 1000)                 # one of a pair
        @test_throws ArgumentError Leg(1e-9; steps = 1000, save_dt = 1e-10)  # mixed forms
        @test_throws ArgumentError Leg(1e-9; dt = 1e-12)
        @test_throws ArgumentError Leg(1e-9)
        @test_throws ArgumentError Leg(0.0; steps = 10, saves = 1)
        @test_throws ArgumentError Leg(1e-9; steps = 0, saves = 1)
        @test_throws ArgumentError Leg(1e-9; steps = 10, saves = 11)
        @test_throws ArgumentError Leg(1e-9; steps = 10, saves = 0)
        @test_throws ArgumentError Leg(1e-9; dt = 3e-12, save_dt = 1e-10)    # dt does not divide
        @test_throws ArgumentError Leg(1e-9; dt = 1e-12, save_dt = 3e-10)    # save_dt does not divide
        @test_throws ArgumentError Leg(1e-9; dt = 1e-10, save_dt = 1e-12)    # save_dt shorter than dt
        @test_throws ArgumentError Leg(1e-9; save_dt = 3e-10)
        @test_throws TypeError Leg(1e-9; steps = 1000.0, saves = 1)          # a float count
        @test_throws ArgumentError time_points(Leg[])
    end

    @testset "time points of a schedule of ten legs" begin
        times, save_times = time_points(growing_dt_legs())
        @test length(times) == 5501
        @test times[1] == 0.0 && save_times[1] == 0.0
        @test last(times) ≈ 1e-3
        @test last(save_times) == last(times)
        @test length(save_times) == 1 + 1 + 3 + 7 + 10 * 6 + 10
        @test all(k -> times[k + 1] > times[k], 1:(length(times) - 1))
        @test issorted(save_times) && allunique(save_times)
        # every save time is a time point, bit for bit
        @test all(s -> s in times, save_times)
        # the leg ends are time points and save times (summed one by one: `cumsum` sums pairwise
        # and its floats differ)
        ends = accumulate(+, [leg.duration for leg in growing_dt_legs()])
        @test all(e -> e in times, ends)
        @test all(e -> e in save_times, ends)
        # a leg of 1000 steps with 3 saves saves at the points 333, 667 and 1000 of the leg
        @test save_times[3:5] == times[1001 .+ [333, 667, 1000]]
        # the same schedule in the dt form
        by_dt = [
            Leg(1e-9; dt = 1e-12, save_dt = 1e-9), Leg(3e-9; dt = 3e-12, save_dt = 1e-9),
            Leg(7e-9; dt = 7e-12, save_dt = 1e-9),
            Leg(1e-8; dt = 1e-11, save_dt = 1e-9), Leg(1e-8; dt = 1e-10, save_dt = 1e-9), Leg(
                1e-7; dt = 1e-9, save_dt = 1e-8),
            Leg(1e-6; dt = 1e-8, save_dt = 1e-7), Leg(1e-5; dt = 1e-7, save_dt = 1e-6), Leg(
                1e-4; dt = 1e-6, save_dt = 1e-5),
            Leg(1e-3 - 1.11131e-4; dt = (1e-3 - 1.11131e-4) / 1000, save_dt = (1e-3 -
                                                                               1.11131e-4) /
                                                                              10)]
        @test time_points(by_dt) == (times, save_times)
        # a start other than zero shifts everything
        shifted, shifted_saves = time_points(growing_dt_legs()[1:2]; start = 2.0)
        @test shifted[1] == 2.0 && last(shifted) ≈ 2.0 + 4e-9
        @test all(s -> s in shifted, shifted_saves)
    end

    @testset "controller legs" begin
        times, save_times = time_points([
            Leg(1e-3; save_dt = 1e-4), Leg(2e-3; save_dt = 1e-3)])
        @test times == [0.0, 1e-3, 1e-3 + 2e-3]
        @test length(save_times) == 1 + 10 + 2
        @test last(save_times) == last(times)
        @test save_times[2] ≈ 1e-4
    end

    @testset "a save time of a prescribed-step run is a time point" begin
        k = 3.0
        system = single_unknown_system(TimeDerivative(), LinearTerm(:x; coefficient = -k))
        times, save_times = time_points([
            Leg(0.5; steps = 5, saves = 1), Leg(0.5; steps = 10, saves = 2)])
        solution = integrate(system, [1.0], times; save_times)
        @test solution.times == save_times == [0.0, 0.5, 0.75, 1.0]
        @test solution.states[1, 2] ≈ (1 + k * 0.1)^-5 rtol = 1e-12
        # a save time between the time points (0.25, between 0.2 and 0.3) is rejected by
        # FixedSteps and StepDoubling
        @test_throws ArgumentError integrate(system, [1.0], times; save_times = [
            0.0, 0.25, 1.0])
        @test_throws ArgumentError integrate(
            system, [1.0], times; save_times = [0.0, 0.25, 1.0],
            stepper = StepDoubling(initial_step = 0.05))
        # a controller leg among them is rejected the same way, and the controller takes it
        controller_times, controller_saves = time_points([
            Leg(0.5; steps = 5, saves = 1), Leg(0.5; save_dt = 0.1)])
        @test_throws ArgumentError integrate(system, [1.0], controller_times; save_times = controller_saves)
        solution = integrate(
            system, [1.0], controller_times; save_times = controller_saves,
            stepper = StepController(initial_step = 0.05))
        @test solution.times == controller_saves
        @test all(t -> t in solution.statistics.time[solution.statistics.accepted], controller_saves[2:end])
    end
end
