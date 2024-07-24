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

GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "All")

## Main tests

@testset verbose = true "DifferentiationInterfaceTest.jl" begin
    if GROUP == "Formalities" || GROUP == "All"
        @testset verbose = true "Formalities" begin
            include("formalities.jl")
        end
    end

    if GROUP == "Zero" || GROUP == "All"
        @testset verbose = false "Zero" begin
            include("zero.jl")
        end
    end

    if GROUP == "ForwardDiff" || GROUP == "All"
        @testset verbose = false "ForwardDiff" begin
            include("forwarddiff.jl")
        end
    end

    if GROUP == "Weird" || GROUP == "All"
        @testset verbose = false "Weird" begin
            include("weird.jl")
        end
    end
end;
