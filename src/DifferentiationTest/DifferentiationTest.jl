"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ..DifferentiationInterface
import ..DifferentiationInterface as DI
using ..DifferentiationInterface:
    mode,
    myzero,
    supports_mutation,
    supports_pushforward,
    supports_pullback,
    myzero,
    myzero!
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
include("benchmark.jl")
include("call_count.jl")
include("test_operators.jl")

export Scenario, default_scenarios 
export BenchmarkData
export test_operators

end
