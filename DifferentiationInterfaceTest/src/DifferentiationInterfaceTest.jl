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
    AutoSparse,
    ForwardMode,
    ForwardOrReverseMode,
    ReverseMode,
    SymbolicMode,
    mode
using Chairmarks: @be, Benchmark, Sample
using DataFrames: DataFrame
using DifferentiationInterface
using DifferentiationInterface:
    inner,
    mode,
    outer,
    inplace_support,
    prepare!_derivative,
    prepare!_gradient,
    prepare!_hessian,
    prepare!_hvp,
    prepare!_jacobian,
    prepare!_pullback,
    prepare!_pushforward,
    prepare!_second_derivative,
    pushforward_performance,
    pullback_performance,
    unwrap
using DifferentiationInterface:
    DerivativePrep,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    PullbackPrep,
    PushforwardPrep,
    SecondDerivativePrep,
    MixedMode,
    SecondOrder,
    Rewrap
import DifferentiationInterface as DI
using DocStringExtensions
using Functors: fmap
using JET: JET
using LinearAlgebra: Adjoint, Diagonal, Transpose, dot, parent
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, spdiagm
import SparseMatrixColorings as SMC
using Test: @testset, @test

include("utils.jl")

include("scenarios/scenario.jl")
include("scenarios/modify.jl")
include("scenarios/default.jl")
include("scenarios/sparse.jl")
include("scenarios/allocfree.jl")
include("scenarios/extensions.jl")

include("tests/correctness_eval.jl")
include("tests/type_stability_eval.jl")
include("tests/sparsity.jl")
include("tests/benchmark.jl")
include("tests/benchmark_eval.jl")
include("test_differentiation.jl")

export Scenario
export default_scenarios, sparse_scenarios
export test_differentiation, benchmark_differentiation
export DifferentiationBenchmarkDataRow
# extensions
export static_scenarios
export component_scenarios
export gpu_scenarios

end
