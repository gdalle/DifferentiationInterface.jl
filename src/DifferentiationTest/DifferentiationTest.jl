"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationTest

using ADTypes
using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using Base: get_extension
using ..DifferentiationInterface
import ..DifferentiationInterface as DI
using ..DifferentiationInterface:
    AutoTaped,
    inner,
    mode,
    mysimilar,
    mysimilar_random,
    myzero,
    myzero!!,
    outer,
    supports_mutation,
    supports_pushforward,
    supports_pullback
using DocStringExtensions
using Functors: @functor, fleaves, fmap
using LinearAlgebra: dot
using Test: @testset, @test

include("scenario.jl")
include("compatibility.jl")
include("scenarios_default.jl")
include("scenarios_weird_arrays.jl")
include("scenarios_nested.jl")
include("zero.jl")
include("printing.jl")
include("benchmark.jl")
include("test_call_count.jl")
include("test_error_free.jl")
include("test_differentiation.jl")

export Scenario, default_scenarios, weird_array_scenarios, nested_scenarios
export BenchmarkData, record!
export all_operators, test_differentiation

end
