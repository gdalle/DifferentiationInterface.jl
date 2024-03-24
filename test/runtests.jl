## Imports

using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest

using Aqua: Aqua
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

# used for testing
using Chairmarks: Chairmarks
using DataFrames: DataFrames
using ForwardDiff: ForwardDiff
using ComponentArrays
using JLArrays
using StaticArrays

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    @testset verbose = true "Formal tests" begin
        @testset "Aqua" begin
            @info "Running Aqua.jl tests..."
            @time Aqua.test_all(DifferentiationInterface; ambiguities=false)
        end
        @testset "JuliaFormatter" begin
            @info "Running JuliaFormatter test..."
            @time @test JuliaFormatter.format(
                DifferentiationInterface; verbose=false, overwrite=false
            )
        end
        @testset "JET" begin
            @info "Running JET tests..."
            @time JET.test_package(DifferentiationInterface; target_defined_modules=true)
        end
    end

    @testset verbose = true "Trivial backends" begin
        @info "Running no backend tests..."
        @testset "No backend" begin
            @time include("nobackend.jl")
        end
        @info "Running zero backend tests..."
        @testset "Zero backend" begin
            @time include("zero.jl")
        end
    end

    @testset verbose = true "First order" begin
        @testset "ChainRules (reverse)" begin
            @info "Running ChainRules (reverse) tests..."
            @time include("chainrules_reverse.jl")
        end
        @testset "Diffractor (forward)" begin
            @info "Running Diffractor (forward) tests..."
            @time include("diffractor.jl")
        end
        @testset "Enzyme (forward)" begin
            @info "Running Enzyme (forward) tests..."
            @time include("enzyme_forward.jl")
        end
        @testset "Enzyme (reverse)" begin
            @info "Running Enzyme (reverse) tests..."
            @time include("enzyme_reverse.jl")
        end
        @testset "FastDifferentiation" begin
            @info "Running FastDifferentiation tests..."
            @time include("fastdifferentiation.jl")
        end
        @testset "FiniteDiff" begin
            @info "Running FiniteDiff tests..."
            @time include("finitediff.jl")
        end
        @testset "FiniteDifferences" begin
            @info "Running FiniteDifferences tests..."
            @time include("finitedifferences.jl")
        end
        @testset "ForwardDiff" begin
            @info "Running ForwardDiff tests..."
            @time include("forwarddiff.jl")
        end
        @testset "PolyesterForwardDiff" begin
            @info "Running PolyesterForwardDiff tests..."
            @time include("polyesterforwarddiff.jl")
        end
        @testset "ReverseDiff" begin
            @info "Running ReverseDiff tests..."
            @time include("reversediff.jl")
        end
        @testset "Tracker" begin
            @info "Running Tracker tests..."
            @time include("tracker.jl")
        end
        @testset "Zygote" begin
            @info "Running Zygote tests..."
            @time include("zygote.jl")
        end
    end

    @testset verbose = true "Second order" begin
        include("second_order.jl")
    end
end;
