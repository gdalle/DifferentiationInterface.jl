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
    Batch,
    inner,
    maybe_inner,
    maybe_dense_ad,
    mode,
    outer,
    twoarg_support,
    pushforward_performance,
    pullback_performance
using DifferentiationInterface:
    prepare_hvp_batched,
    prepare_hvp_batched_same_point,
    prepare_pullback_batched,
    prepare_pullback_batched_same_point,
    prepare_pushforward_batched,
    prepare_pushforward_batched_same_point,
    hvp_batched,
    hvp_batched!,
    pullback_batched,
    pullback_batched!,
    pushforward_batched,
    pushforward_batched!,
    value_and_pullback_batched,
    value_and_pullback_batched!,
    value_and_pushforward_batched,
    value_and_pushforward_batched!
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
using LinearAlgebra: Adjoint, Diagonal, Transpose, dot, parent
using PackageExtensionCompat: @require_extensions
using ProgressMeter: ProgressUnknown, next!
using Random: AbstractRNG, default_rng, rand!
using SparseArrays
using Test: @testset, @test

include("scenarios/scenario.jl")
include("scenarios/batchify.jl")
include("scenarios/default.jl")
include("scenarios/sparse.jl")
include("scenarios/allocfree.jl")
include("scenarios/extensions.jl")

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

function __init__()
    @require_extensions
end

export Scenario
export PushforwardScenario,
    PullbackScenario,
    DerivativeScenario,
    GradientScenario,
    JacobianScenario,
    SecondDerivativeScenario,
    HVPScenario,
    HessianScenario
export default_scenarios, sparse_scenarios
export test_differentiation, benchmark_differentiation
export DifferentiationBenchmarkDataRow
# extensions
export static_scenarios
export component_scenarios
export gpu_scenarios

end
