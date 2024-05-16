using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using Pkg
using SparseConnectivityTracer
using Test

DI_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterface")
if isdir(DI_PATH)
    Pkg.develop(; path=DI_PATH)
end

LOGGING = get(ENV, "CI", "false") == "false"

## Main tests

@testset verbose = true "DifferentiationInterfaceTest.jl" begin
    @testset verbose = true "Formal tests" begin
        @static if VERSION >= v"1.10"
            include("formal.jl")
        end
    end

    @testset verbose = false "Zero backends" begin
        include("zero_backends.jl")
    end

    @testset verbose = false "ForwardDiff" begin
        include("forwarddiff.jl")
    end
end;
