"""
    DifferentiationInterfaceTest

Testing and benchmarking utilities for automatic differentiation in Julia.

# Exports

$(EXPORTS)
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
    backend_string,
    inner,
    mode,
    outer,
    supports_mutation,
    pushforward_performance,
    pullback_performance
using DifferentiationInterface: NoPullbackExtras, NoPushforwardExtras
using DocStringExtensions
import DifferentiationInterface as DI
using JET: @test_call, @test_opt
using JLArrays: jl
using LinearAlgebra: Diagonal, dot
using ProgressMeter: ProgressUnknown, next!
using SparseArrays: SparseArrays, nnz, SparseMatrixCSC
using StaticArrays: MMatrix, MVector, SMatrix, SVector
using Test: @testset, @test

include("scenarios/scenario.jl")
include("scenarios/default.jl")
include("scenarios/static.jl")
include("scenarios/component.jl")
include("scenarios/gpu.jl")

include("utils/zero_backends.jl")
include("utils/compatibility.jl")
include("utils/misc.jl")
include("utils/filter.jl")

include("tests/correctness.jl")
include("tests/type_stability.jl")
include("tests/sparsity.jl")
include("tests/benchmark.jl")
include("test_differentiation.jl")

export AbstractScenario
export PushforwardScenario,
    PullbackScenario,
    DerivativeScenario,
    GradientScenario,
    JacobianScenario,
    SecondDerivativeScenario,
    HVPScenario,
    HessianScenario
export default_scenarios
export static_scenarios, component_scenarios, gpu_scenarios
export test_differentiation, benchmark_differentiation

end
