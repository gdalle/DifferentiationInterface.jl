using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using Pkg
using SparseConnectivityTracer
using Test

GROUP = get(ENV, "JULIA_DIT_TEST_GROUP", "All")

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
