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
    @time "Aqua" @testset "Aqua" begin
        Aqua.test_all(DifferentiationInterface; ambiguities=false)
    end
    @time "JuliaFormatter" @testset "JuliaFormatter" begin
        @test JuliaFormatter.format(
            DifferentiationInterface; verbose=false, overwrite=false
        )
    end
    @time "JET" @testset "JET" begin
        JET.test_package(DifferentiationInterface; target_defined_modules=true)
    end

    @time "No backend" @testset "No backend" begin
        include("nobackend.jl")
    end
    @time "Zero backend" @testset "Zero backend" begin
        include("zero.jl")
    end

    @time "ChainRules (reverse)" @testset "ChainRules (reverse)" begin
        include("chainrules_reverse.jl")
    end
    @time "Diffractor" @testset "Diffractor (forward)" begin
        include("diffractor.jl")
    end
    @time "Enzyme (forward)" @testset "Enzyme (forward)" begin
        include("enzyme_forward.jl")
    end
    @time "Enzyme (reverse)" @testset "Enzyme (reverse)" begin
        include("enzyme_reverse.jl")
    end
    @time "FastDifferentiation" @testset "FastDifferentiation" begin
        include("fastdifferentiation.jl")
    end
    @time "FiniteDiff" @testset "FiniteDiff" begin
        include("finitediff.jl")
    end
    @time "FiniteDifferences" @testset "FiniteDifferences" begin
        include("finitedifferences.jl")
    end
    @time "ForwardDiff" @testset "ForwardDiff" begin
        include("forwarddiff.jl")
    end
    @time "PolyesterForwardDiff" @testset "PolyesterForwardDiff" begin
        include("polyesterforwarddiff.jl")
    end
    @time "ReverseDiff" @testset "ReverseDiff" begin
        include("reversediff.jl")
    end
    @time "Tracker" @testset "Tracker" begin
        include("tracker.jl")
    end
    @time "Zygote" @testset "Zygote" begin
        include("zygote.jl")
    end
end;
