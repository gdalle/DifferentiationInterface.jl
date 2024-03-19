## Imports

using Aqua: Aqua
using ChainRulesCore: ChainRulesCore
using Chairmarks: Chairmarks
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

    @testset "No backend" begin
        include("nobackend.jl")
    end
    @testset "Zero backend" begin
        include("zero.jl")
    end

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
    @testset "Zygote" begin
        include("zygote.jl")
    end

    @testset "Second order" begin
        include("second_order.jl")
    end
end
