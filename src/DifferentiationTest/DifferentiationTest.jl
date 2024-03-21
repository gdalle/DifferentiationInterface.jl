"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ..DifferentiationInterface
import ..DifferentiationInterface as DI
using ..DifferentiationInterface:
    ForwardMode,
    MutationNotSupported,
    ReverseMode,
    SymbolicMode,
    inner,
    mode,
    mutation_behavior,
    outer,
    zero!
using ADTypes
using ADTypes: AbstractADType
using DocStringExtensions
using Test: @testset, @test

include("scenario.jl")
include("benchmark.jl")
include("call_count.jl")
include("default_scenarios.jl")
include("test_operators.jl")
include("zero.jl")
include("printing.jl")

export backend_string
export Scenario, default_scenarios
export allocating, mutating, scalar_scalar, scalar_array, array_scalar, array_array
export BenchmarkData, record!
export test_operators
export AutoZeroForward, AutoZeroReverse

end
