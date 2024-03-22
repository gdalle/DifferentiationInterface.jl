"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ..DifferentiationInterface
import ..DifferentiationInterface as DI
using ..DifferentiationInterface:
    inner,
    mode,
    outer,
    supports_mutation,
    supports_pushforward,
    supports_pullback,
    supports_hvp,
    zero!,
    MutationBehavior,
    MutationSupported,
    MutationNotSupported
using ADTypes
using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using DocStringExtensions
using Test: @testset, @test

include("testtraits.jl")
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
export isallocating, ismutating
export BenchmarkData, record!
export test_operators
export AutoZeroForward, AutoZeroReverse

end
