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
    AbstractMode,
    ForwardMode,
    ForwardOrReverseMode,
    ReverseMode,
    SymbolicMode
using Chairmarks: @be, Benchmark, Sample
using Compat
using ComponentArrays: ComponentVector
using DataFrames: DataFrame
using DifferentiationInterface
using DifferentiationInterface:
    Batch,
    backend_str,
    inner,
    maybe_inner,
    maybe_dense_ad,
    mode,
    outer,
    twoarg_support,
    prepare_hvp_batched,
    prepare_pullback_batched,
    prepare_pushforward_batched,
    hvp_batched,
    hvp_batched!,
    pullback_batched,
    pullback_batched!,
    pushforward_batched,
    pushforward_batched!,
    pushforward_performance,
    pullback_performance
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    NoPullbackExtras,
    NoPushforwardExtras,
    SecondDerivativeExtras
using DocStringExtensions
import DifferentiationInterface as DI
using JET: JET
using JLArrays: JLArray, jl
using LinearAlgebra: Adjoint, Diagonal, Transpose, dot, parent
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, spdiagm
using StaticArrays: MArray, MMatrix, MVector, SArray, SMatrix, SVector
using Test: @testset, @test

include("scenarios/scenario.jl")
include("scenarios/default.jl")
include("scenarios/sparse.jl")
include("scenarios/static.jl")
include("scenarios/component.jl")
include("scenarios/gpu.jl")
include("scenarios/allocfree.jl")

include("utils/zero_backends.jl")
include("utils/misc.jl")
include("utils/filter.jl")

include("tests/correctness.jl")
@static if VERSION >= v"1.7"
    include("tests/type_stability.jl")
end
include("tests/sparsity.jl")
include("tests/benchmark.jl")
include("test_differentiation.jl")

export PushforwardScenario,
    PullbackScenario,
    DerivativeScenario,
    GradientScenario,
    JacobianScenario,
    SecondDerivativeScenario,
    HVPScenario,
    HessianScenario
export default_scenarios, sparse_scenarios
export static_scenarios, component_scenarios, gpu_scenarios
export test_differentiation, benchmark_differentiation
export DifferentiationBenchmarkDataRow

end
