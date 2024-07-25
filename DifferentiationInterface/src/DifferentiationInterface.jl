"""
    DifferentiationInterface

An interface to various automatic differentiation backends in Julia.

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using ADTypes: ADTypes, AbstractADType
using ADTypes: mode, ForwardMode, ForwardOrReverseMode, ReverseMode, SymbolicMode
using ADTypes: AutoSparse, dense_ad
using ADTypes: coloring_algorithm, column_coloring, row_coloring
using ADTypes: sparsity_detector, jacobian_sparsity, hessian_sparsity
using ADTypes:
    AutoChainRules,
    AutoDiffractor,
    AutoEnzyme,
    AutoFastDifferentiation,
    AutoFiniteDiff,
    AutoFiniteDifferences,
    AutoForwardDiff,
    AutoPolyesterForwardDiff,
    AutoReverseDiff,
    AutoSymbolics,
    AutoTapir,
    AutoTracker,
    AutoZygote
using Compat
using DocStringExtensions
using FillArrays: OneElement
using LinearAlgebra: Symmetric, Transpose, dot, parent, transpose
using PackageExtensionCompat: @require_extensions
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse
using SparseMatrixColorings:
    GreedyColoringAlgorithm,
    color_groups,
    decompress_columns,
    decompress_columns!,
    decompress_rows,
    decompress_rows!,
    decompress_symmetric,
    decompress_symmetric!,
    symmetric_coloring_detailed,
    StarSet

abstract type Extras end

include("second_order/second_order.jl")

include("utils/traits.jl")
include("utils/basis.jl")
include("utils/batch.jl")
include("utils/check.jl")
include("utils/exceptions.jl")
include("utils/maybe.jl")
include("utils/printing.jl")

include("first_order/pushforward.jl")
include("first_order/pushforward_batched.jl")
include("first_order/pullback.jl")
include("first_order/pullback_batched.jl")
include("first_order/derivative.jl")
include("first_order/gradient.jl")
include("first_order/jacobian.jl")

include("second_order/second_derivative.jl")
include("second_order/hvp.jl")
include("second_order/hvp_batched.jl")
include("second_order/hessian.jl")

include("sparse/fallbacks.jl")
include("sparse/matrices.jl")
include("sparse/jacobian.jl")
include("sparse/hessian.jl")

include("misc/differentiate_with.jl")
include("misc/sparsity_detector.jl")
include("misc/from_primitive.jl")

struct ReactantBackend{B} <: ADTypes.AbstractADType
    backend::B
end

function __init__()
    @require_extensions
end

## Exported

export SecondOrder

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback

export value_and_derivative!, value_and_derivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

export pushforward!, pushforward
export pullback!, pullback

export derivative!, derivative
export gradient!, gradient
export jacobian!, jacobian

export second_derivative!, second_derivative
export value_derivative_and_second_derivative, value_derivative_and_second_derivative!
export hvp!, hvp
export hessian!, hessian
export value_gradient_and_hessian, value_gradient_and_hessian!

export prepare_pushforward, prepare_pushforward_same_point
export prepare_pullback, prepare_pullback_same_point
export prepare_hvp, prepare_hvp_same_point
export prepare_derivative, prepare_gradient, prepare_jacobian
export prepare_second_derivative, prepare_hessian

export check_available, check_twoarg, check_hessian

export DifferentiateWith
export DenseSparsityDetector

## Re-exported from ADTypes

export AutoChainRules
export AutoDiffractor
export AutoEnzyme
export AutoFastDifferentiation
export AutoFiniteDiff
export AutoFiniteDifferences
export AutoForwardDiff
export AutoPolyesterForwardDiff
export AutoReverseDiff
export AutoSymbolics
export AutoTapir
export AutoTracker
export AutoZygote

export AutoSparse

## Re-exported from SparseMatrixColorings

export GreedyColoringAlgorithm

## Public but not exported

@compat public inner
@compat public outer

end # module
