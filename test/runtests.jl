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

@time @testset verbose = true "DifferentiationInterface.jl" begin
    @testset verbose = true "Formal tests" begin
        @testset "Aqua" begin
            Aqua.test_all(DifferentiationInterface; ambiguities=false)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(
                DifferentiationInterface; verbose=false, overwrite=false
            )
        end
        @testset "JET" begin
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
            include("chainrules_reverse.jl")
        end
        @testset "Diffractor (forward)" begin
            include("diffractor.jl")
        end
        @testset "Enzyme (forward)" begin
            include("enzyme_forward.jl")
        end
        @testset "Enzyme (reverse)" begin
            include("enzyme_reverse.jl")
        end
        @testset "FastDifferentiation" begin
            include("fastdifferentiation.jl")
        end
        @testset "FiniteDiff" begin
            include("finitediff.jl")
        end
        @testset "ForwardDiff" begin
            include("forwarddiff.jl")
        end
        @testset "PolyesterForwardDiff" begin
            include("polyesterforwarddiff.jl")
        end
        @testset "ReverseDiff" begin
            include("reversediff.jl")
        end
        @testset "Tracker" begin
            include("tracker.jl")
        end
        @testset "Zygote" begin
            include("zygote.jl")
        end
    end

    @testset verbose = true "Second order" begin
        include("second_order.jl")
    end
end
