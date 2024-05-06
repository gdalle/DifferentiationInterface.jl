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
using ADTypes: coloring_algorithm, column_coloring, row_coloring, symmetric_coloring
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

abstract type Extras end

include("second_order/second_order.jl")

include("utils/traits.jl")
include("utils/basis.jl")
include("utils/printing.jl")
include("utils/chunk.jl")
include("utils/check.jl")
include("utils/exceptions.jl")

include("first_order/pushforward.jl")
include("first_order/pullback.jl")
include("first_order/derivative.jl")
include("first_order/gradient.jl")
include("first_order/jacobian.jl")

include("second_order/second_derivative.jl")
include("second_order/hvp.jl")
include("second_order/hessian.jl")

include("sparse/detector.jl")
include("sparse/matrices.jl")
include("sparse/coloring.jl")
include("sparse/compressed_matrix.jl")
include("sparse/fallbacks.jl")
include("sparse/jacobian.jl")
include("sparse/hessian.jl")

include("translation/differentiate_with.jl")

function __init__()
    @require_extensions
end

export SecondOrder

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback
export value_and_pullback!_split, value_and_pullback_split

export value_and_derivative!, value_and_derivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

export pushforward!, pushforward
export pullback!, pullback

export derivative!, derivative
export gradient!, gradient
export jacobian!, jacobian

export second_derivative!, second_derivative
export hvp!, hvp
export hessian!, hessian

export prepare_pushforward, prepare_pullback
export prepare_derivative, prepare_gradient, prepare_jacobian
export prepare_second_derivative, prepare_hvp, prepare_hessian

export check_available, check_twoarg, check_hessian

export DifferentiateWith

# Re-export backends from ADTypes
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

end # module
