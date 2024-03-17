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
    extras::Nothing,
) where {C}
    return DI.value_and_pushforward!(
        dy, AutoForwardDiff{C,Nothing}(nothing), f, x, dx, extras
    )
end

function DI.value_and_pushforward!(
    y::AbstractArray,
    dy::AbstractArray,
    ::AutoPolyesterForwardDiff{C},
    f!,
    x,
    dx,
    extras::Nothing,
) where {C}
    return DI.value_and_pushforward!(
        y, dy, AutoForwardDiff{C,Nothing}(nothing), f!, x, dx, extras
    )
end

## Utilities

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoPolyesterForwardDiff{C}, f, x::AbstractArray, extras::Nothing
) where {C}
    y = f(x)
    threaded_gradient!(f, grad, x, Chunk{C}())
    return y, grad
end

function DI.gradient!(
    grad::AbstractArray, ::AutoPolyesterForwardDiff{C}, f, x::AbstractArray, extras::Nothing
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return grad
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, ::AutoPolyesterForwardDiff{C}, f, x::AbstractArray, extras::Nothing
) where {C}
    y = f(x)
    threaded_jacobian!(f, jac, x, Chunk{C}())
    return y, jac
end

function DI.jacobian!(
    jac::AbstractMatrix, ::AutoPolyesterForwardDiff{C}, f, x::AbstractArray, extras::Nothing
) where {C}
    threaded_jacobian!(f, jac, x, Chunk{C}())
    return jac
end

function DI.value_and_jacobian!(
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoPolyesterForwardDiff{C},
    f!,
    x::AbstractArray,
    extras::Nothing,
) where {C}
    f!(y, x)
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return y, jac
end

end # module
