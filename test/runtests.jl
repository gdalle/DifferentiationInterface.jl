## Imports

using Aqua: Aqua
using ChainRulesCore: ChainRulesCore
using Chairmarks: Chairmarks
using DataFrames: DataFrames
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using JET: JET
using JuliaFormatter: JuliaFormatter
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Random: Random
using Test
using Zygote: Zygote

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    @testset verbose = true "Formal tests" begin
        @testset "Aqua" begin
            @info "Running Aqua.jl tests..."
            Aqua.test_all(DifferentiationInterface; ambiguities=false)
        end
        @testset "JuliaFormatter" begin
            @info "Running JuliaFormatter test..."
            @test JuliaFormatter.format(
                DifferentiationInterface; verbose=false, overwrite=false
            )
        end
        @testset "JET" begin
            @info "Running JET tests..."
            JET.test_package(DifferentiationInterface; target_defined_modules=true)
        end
    end

    @testset verbose = true "Trivial backends" begin
        @testset "No backend" begin
            include("nobackend.jl")
        end
        @testset "Zero backend" begin
            include("zero.jl")
        end
    end

    @testset verbose = true "First order" begin
        @testset "ChainRules (reverse)" begin
            @info "Running ChainRules (reverse) tests..."
            include("chainrules_reverse.jl")
        end
        @testset "Diffractor (forward)" begin
            @info "Running Diffractor (forward) tests..."
            include("diffractor.jl")
        end
        @testset "Enzyme (forward)" begin
            @info "Running Enzyme (forward) tests..."
            include("enzyme_forward.jl")
        end
        @testset "Enzyme (reverse)" begin
            @info "Running Enzyme (reverse) tests..."
            include("enzyme_reverse.jl")
        end
        @testset "FastDifferentiation" begin
            @info "Running FastDifferentiation tests..."
            include("fastdifferentiation.jl")
        end
        @testset "FiniteDiff" begin
            @info "Running FiniteDiff tests..."
            include("finitediff.jl")
        end
        @testset "ForwardDiff" begin
            @info "Running ForwardDiff tests..."
            include("forwarddiff.jl")
        end
        @testset "PolyesterForwardDiff" begin
            @info "Running PolyesterForwardDiff tests..."
            include("polyesterforwarddiff.jl")
        end
        @testset "ReverseDiff" begin
            @info "Running ReverseDiff tests..."
            include("reversediff.jl")
        end
        @testset "Tracker" begin
            @info "Running Tracker tests..."
            include("tracker.jl")
        end
        @testset "Zygote" begin
            @info "Running Zygote tests..."
            include("zygote.jl")
        end
    end

    @testset verbose = true "Second order" begin
        @info "Running second order tests..."
        include("second_order.jl")
    end
end
