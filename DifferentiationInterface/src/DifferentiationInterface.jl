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

"""
    AutoFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl).

!!! danger
    This backend is experimental, use at your own risk.
"""
struct AutoFastDifferentiation <: AbstractSymbolicDifferentiationMode end

"""
    AutoSparseFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl) leveraging sparsity.

!!! danger
    This backend is experimental, use at your own risk.
"""
struct AutoSparseFastDifferentiation <: AbstractSymbolicDifferentiationMode end

"""
    AutoTapir

Chooses [Tapir.jl](https://github.com/withbayes/Tapir.jl).

!!! danger
    This backend is experimental, use at your own risk.
"""
struct AutoTapir <: AbstractReverseMode end

abstract type Extras end

include("second_order.jl")
include("traits.jl")
include("utils.jl")

include("pushforward.jl")
include("pullback.jl")

include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")

include("second_derivative.jl")
include("hvp.jl")
include("hessian.jl")

include("backends.jl")

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

export AutoFastDifferentiation, AutoSparseFastDifferentiation, AutoTapir
export SecondOrder

export value_and_pushforward!!, value_and_pushforward
export value_and_pullback!!, value_and_pullback

export value_and_derivative!!, value_and_derivative
export value_and_gradient!!, value_and_gradient
export value_and_jacobian!!, value_and_jacobian

export pushforward!!, pushforward
export pullback!!, pullback

export derivative!!, derivative
export gradient!!, gradient
export jacobian!!, jacobian

export second_derivative!!, second_derivative
export hvp!!, hvp
export hessian!!, hessian

export prepare_pushforward, prepare_pullback
export prepare_derivative, prepare_gradient, prepare_jacobian
export prepare_second_derivative, prepare_hvp, prepare_hessian

export check_available, check_mutation, check_hessian

function __init__()
    Base.Experimental.register_error_hint(StackOverflowError) do io, exc, argtypes, kwargs
        f_name = string(exc.f)
        if (
            f_name == "mode" ||
            contains(f_name, "pushforward") ||
            contains(f_name, "pullback") ||
            contains(f_name, "derivative") ||
            contains(f_name, "gradient") ||
            contains(f_name, "jacobian") ||
            contains(f_name, "hvp") ||
            contains(f_name, "hessian")
        )
            for T in argtypes
                if T <: AbstractADType
                    print(
                        io,
                        """\n
                        HINT: One of DifferentiationInterface's functions is missing a method, which causes an endless loop of `pullback` calling `pushforward` and vice-versa.
                        Some possible fixes:
                        - switch to another backend
                        - if you don't want to switch, load the package extension corresponding to backend `$T`
                        - if the package is already loaded, define the method `$f_name` for the right combination of argument types
                        """,
                    )
                    return nothing
                end
            end
        end
    end
end

end # module