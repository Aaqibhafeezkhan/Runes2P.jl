# Threads over radii: the residual, the stored Jacobian and the steps of two systems on 5
# radii at 4 Julia threads equal those at one thread bit for bit, a step that recovers from a
# DomainError of a trial state included. CI starts Julia with one thread, so each thread count
# runs in a subprocess (threads_over_radii_child.jl), which writes its results as raw bytes.

@testset "threads over radii" begin
    @testset "an error in a radius reaches the caller as itself" begin
        failing(ir) = ir == 3 ? throw(DomainError(-1.0, "radius 3")) : nothing
        @test_throws DomainError Runes2P._each_radius_threaded(failing, 5)
    end

    @testset "results at 1 and 4 threads bit for bit" begin
        child = joinpath(@__DIR__, "threads_over_radii_child.jl")
        project = dirname(Base.active_project())
        directories = Dict{Int, String}()
        for threads in (1, 4)
            directory = mktempdir()
            run(`$(Base.julia_cmd()) --startup-file=no --threads=$threads --project=$project $child $directory`)
            directories[threads] = directory
        end
        # each child ran with the threads it was given
        for threads in (1, 4)
            @test read(joinpath(directories[threads], "threads.txt"), String) == string(threads)
        end
        names = filter(!=("threads.txt"), readdir(directories[1]))
        @test names == filter(!=("threads.txt"), readdir(directories[4]))
        @test length(names) == 8
        for name in names
            @test read(joinpath(directories[1], name)) == read(joinpath(directories[4], name))
        end
        # the second system recovers from the DomainError by a smaller step at both counts
        @test startswith(read(joinpath(directories[4], "domain_error_status.txt"), String),
            "domain_error")
    end
end
