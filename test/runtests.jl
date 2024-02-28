## Imports

using Aqua: Aqua
using DifferentiationInterface
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

using Diffractor: Diffractor
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

## Utils

include("utils.jl")

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(DifferentiationInterface; deps_compat=(; check_extras=false,))
    end
    @testset "JuliaFormatter" begin
        @test JuliaFormatter.format(
            DifferentiationInterface; verbose=false, overwrite=false
        )
    end
    @testset "JET" begin
        @test_skip JET.test_package(DifferentiationInterface; target_defined_modules=true)
    end

    @testset "Diffractor" begin
        @test_skip include("diffractor.jl")
    end
    @testset "Enzyme" begin
        include("enzyme.jl")
    end
    @testset "FiniteDiff" begin
        include("finitediff.jl")
    end
    @testset "ForwardDiff" begin
        include("forwarddiff.jl")
    end
    @testset "ReverseDiff" begin
        include("reversediff.jl")
    end
    @testset "Zygote" begin
        include("zygote.jl")
    end
end
