"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ..DifferentiationInterface
using ..DifferentiationInterface:
    AutoZeroForward,
    AutoZeroReverse,
    ForwardMode,
    ReverseMode,
    SymbolicMode,
    MutationNotSupported,
    mode,
    mutation_behavior,
    inner,
    outer
using ADTypes
using ADTypes: AbstractADType
using DocStringExtensions
using Test: @testset, @test

include("utils.jl")
include("scenario.jl")
include("benchmark.jl")
include("call_count.jl")
include("default_scenarios.jl")
include("test_operators.jl")

export backend_string
export Scenario, default_scenarios, scen_string
export allocating, mutating, scalar_scalar, scalar_array, array_scalar, array_array
export BenchmarkData, record!
export test_operators

end
