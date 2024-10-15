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
    SecondOrder,
    Rewrap
import DifferentiationInterface as DI
using DocStringExtensions
using JET: @test_opt
using LinearAlgebra: Adjoint, Diagonal, Transpose, dot, parent
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays:
    SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, nnz, sparse, spdiagm
using Test: @testset, @test

"""
    FIRST_ORDER = [:pushforward, :pullback, :derivative, :gradient, :jacobian]

List of all first-order operators, to facilitate exclusion during tests.
"""
const FIRST_ORDER = [:pushforward, :pullback, :derivative, :gradient, :jacobian]

"""
    SECOND_ORDER = [:hvp, :second_derivative, :hessian]

List of all second-order operators, to facilitate exclusion during tests.
"""
const SECOND_ORDER = [:hvp, :second_derivative, :hessian]

const ALL_OPS = vcat(FIRST_ORDER, SECOND_ORDER)

include("utils.jl")

include("scenarios/scenario.jl")
include("scenarios/modify.jl")
include("scenarios/default.jl")
include("scenarios/sparse.jl")
include("scenarios/allocfree.jl")
include("scenarios/extensions.jl")

include("tests/correctness_eval.jl")
include("tests/type_stability_eval.jl")
include("tests/benchmark.jl")
include("tests/benchmark_eval.jl")
include("test_differentiation.jl")

export FIRST_ORDER, SECOND_ORDER
export Scenario
export default_scenarios, sparse_scenarios
export test_differentiation, benchmark_differentiation
export DifferentiationBenchmarkDataRow
# extensions
export static_scenarios
export component_scenarios
export gpu_scenarios

end
