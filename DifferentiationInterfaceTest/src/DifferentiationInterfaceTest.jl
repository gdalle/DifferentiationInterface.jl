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
using DataFrames: DataFrame
using DifferentiationInterface
using DifferentiationInterface:
    inner,
    inner,
    dense_ad,
    mode,
    outer,
    inplace_support,
    pushforward_performance,
    pullback_performance
using DifferentiationInterface:
    DerivativePrep,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    PullbackPrep,
    PushforwardPrep,
    NoPullbackPrep,
    NoPushforwardPrep,
    SecondDerivativePrep
using DocStringExtensions
import DifferentiationInterface as DI
using Functors: fmap
using JET: JET
using LinearAlgebra: Adjoint, Diagonal, Transpose, dot, parent
using PackageExtensionCompat: @require_extensions
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, spdiagm
using Test: @testset, @test

include("utils.jl")

include("scenarios/scenario.jl")
include("scenarios/modify.jl")
include("scenarios/default.jl")
include("scenarios/sparse.jl")
include("scenarios/allocfree.jl")
include("scenarios/extensions.jl")

include("tests/correctness_eval.jl")
@static if VERSION >= v"1.7"
    include("tests/type_stability_eval.jl")
end
include("tests/sparsity.jl")
include("tests/benchmark.jl")
include("tests/benchmark_eval.jl")
include("test_differentiation.jl")

function __init__()
    @require_extensions
end

export Scenario
export default_scenarios, sparse_scenarios
export test_differentiation, benchmark_differentiation
export DifferentiationBenchmarkDataRow
# extensions
export static_scenarios
export component_scenarios
export gpu_scenarios

end
