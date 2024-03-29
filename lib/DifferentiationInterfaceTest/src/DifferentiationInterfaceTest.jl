"""
    DifferentiationInterfaceTest

Testing utilities for [`DifferentiationInterface`](@ref).
"""
module DifferentiationInterfaceTest

using ADTypes
using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using Base: get_extension
using Chairmarks: @be, Benchmark, Sample
using ComponentArrays: ComponentVector
using DifferentiationInterface
using DifferentiationInterface:
    AutoTapir,
    inner,
    mode,
    outer,
    supports_mutation,
    pushforward_performance,
    pullback_performance
using DocStringExtensions
import DifferentiationInterface as DI
using JET: @test_call, @test_opt
using JLArrays: jl
using LinearAlgebra: Diagonal, dot
using StaticArrays: SVector, SMatrix
using Test: @testset, @test

include("scenarios/scenario.jl")
include("scenarios/default.jl")
include("scenarios/static.jl")
include("scenarios/component.jl")
include("scenarios/gpu.jl")

include("utils/zero.jl")
include("utils/compatibility.jl")
include("utils/printing.jl")
include("utils/misc.jl")
include("utils/filter.jl")

include("tests/correctness.jl")
include("tests/type_stability.jl")
include("tests/call_count.jl")
include("tests/benchmark.jl")
include("tests/test.jl")

export all_operators
export Scenario
export default_scenarios
export static_scenarios, component_scenarios, gpu_scenarios
export BenchmarkData
export test_differentiation, benchmark_differentiation

end
