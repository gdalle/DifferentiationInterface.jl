"""
    DifferentiationInterface

An interface to various automatic differentiation backends in Julia.

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using ADTypes:
    ADTypes,
    AbstractADType,
    AbstractForwardMode,
    AbstractFiniteDifferencesMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using ADTypes:
    AutoChainRules,
    AutoDiffractor,
    AutoEnzyme,
    AutoFiniteDiff,
    AutoFiniteDifferences,
    AutoForwardDiff,
    AutoPolyesterForwardDiff,
    AutoReverseDiff,
    AutoSparseFiniteDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff,
    AutoSparseReverseDiff,
    AutoSparseZygote,
    AutoTracker,
    AutoZygote
using DocStringExtensions
using FillArrays: OneElement
using LinearAlgebra: Symmetric, dot

abstract type Extras end

include("backends.jl")
include("second_order.jl")
include("traits.jl")
include("utils.jl")
include("printing.jl")

include("pushforward.jl")
include("pullback.jl")

include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")

include("second_derivative.jl")
include("hvp.jl")
include("hessian.jl")

include("check.jl")
include("sparse.jl")
include("chunk.jl")

export AutoChainRules,
    AutoDiffractor,
    AutoEnzyme,
    AutoFiniteDiff,
    AutoFiniteDifferences,
    AutoForwardDiff,
    AutoPolyesterForwardDiff,
    AutoReverseDiff,
    AutoSparseFiniteDiff,
    AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff,
    AutoSparseReverseDiff,
    AutoSparseZygote,
    AutoTracker,
    AutoZygote

export AutoFastDifferentiation,
    AutoSparseFastDifferentiation, AutoSymbolics, AutoSparseSymbolics, AutoTapir
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

end # module
