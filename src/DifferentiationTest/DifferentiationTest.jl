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
    myisapprox,
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
using LinearAlgebra: Diagonal, dot
using Test: @testset, @test

include("scenarios/scenario.jl")
include("scenarios/default.jl")
include("scenarios/weird_arrays.jl")
include("scenarios/nested.jl")
include("scenarios/layer.jl")

include("utils/zero.jl")
include("utils/compatibility.jl")
include("utils/printing.jl")

include("tests/correctness.jl")
include("tests/call_count.jl")
include("tests/benchmark.jl")
include("tests/test.jl")

export Scenario
export default_scenarios
export weird_array_scenarios, nested_scenarios
export BenchmarkData, record!
export all_operators, test_differentiation

end
