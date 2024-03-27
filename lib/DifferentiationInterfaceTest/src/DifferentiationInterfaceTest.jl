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
    AutoTaped,
    inner,
    mode,
    outer,
    supports_mutation,
    supports_pushforward,
    supports_pullback
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

include("utils/zero_backends.jl")
include("utils/compatibility.jl")
include("utils/printing.jl")
include("utils/misc.jl")

include("tests/correctness.jl")
include("tests/type_stability.jl")
include("tests/call_count.jl")
include("tests/benchmark.jl")
include("test_differentiation.jl")

export Scenario
export default_scenarios
export static_scenarios, component_scenarios, gpu_scenarios
export BenchmarkData, record!
export all_operators, test_differentiation

end
