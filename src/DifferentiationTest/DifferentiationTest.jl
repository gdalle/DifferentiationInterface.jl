"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ..DifferentiationInterface:
    AutoFastDifferentiation,
    AutoZeroForward,
    AutoZeroReverse,
    ForwardMode,
    ReverseMode,
    SymbolicMode,
    SecondOrder,
    mode
using ADTypes
using ADTypes: AbstractADType
using DocStringExtensions
using Test: @testset

include("utils.jl")
include("scenario.jl")
include("default_scenarios.jl")
include("test_operators.jl")
include("pretty.jl")

export Scenario, default_scenarios
export allocating, mutating
export scalar_scalar, scalar_array, array_scalar, array_array
export test_operators, parse_benchmark
export backend_string

end
