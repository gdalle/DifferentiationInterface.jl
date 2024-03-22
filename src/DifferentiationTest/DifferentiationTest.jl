"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ..DifferentiationInterface
import ..DifferentiationInterface as DI
using ..DifferentiationInterface:
    mode, supports_mutation, supports_pushforward, supports_pullback, myzero, myzero!
using ADTypes
using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using DocStringExtensions
using Test: @testset, @test

include("scenario.jl")
include("default_scenarios.jl")
include("zero.jl")
include("printing.jl")
include("test_operators.jl")

export Scenario, default_scenarios
export allocating, mutating, scalar_in, scalar_out, same_in_out
export AutoZeroForward, AutoZeroReverse
export backend_string
export test_operators

end
