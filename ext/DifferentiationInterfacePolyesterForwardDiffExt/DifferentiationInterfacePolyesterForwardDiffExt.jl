module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoPolyesterForwardDiff, AutoForwardDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff: Chunk
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!

## Primitives

function DI.value_and_pushforward!(
    dy::Union{Number,AbstractArray},
    ::AutoPolyesterForwardDiff{C},
    f,
    x,
    dx,
    extras::Nothing=nothing,
) where {C}
    return DI.value_and_pushforward!(
        dy, AutoForwardDiff{C,Nothing}(nothing), f, x, dx, extras
    )
end

## Utilities

function DI.value_and_gradient!(
    grad::AbstractArray,
    ::AutoPolyesterForwardDiff{C},
    f,
    x::AbstractArray,
    extras::Nothing=nothing,
) where {C}
    y = f(x)
    threaded_gradient!(f, grad, x, Chunk{C}())
    return y, grad
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix,
    ::AutoPolyesterForwardDiff{C},
    f,
    x::AbstractArray,
    extras::Nothing=nothing,
) where {C}
    y = f(x)
    threaded_jacobian!(f, jac, x, Chunk{C}())
    return y, jac
end

end # module
