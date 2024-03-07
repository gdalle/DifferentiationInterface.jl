module DifferentiationInterfacePolyesterForwardDiffExt

using DifferentiationInterface: ForwardDiffBackend, PolyesterForwardDiffBackend
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff: Chunk
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!

## Backend construction

"""
$(SIGNATURES)
"""
function DI.PolyesterForwardDiffBackend(C::Integer; custom::Bool=true)
    return PolyesterForwardDiffBackend{custom,C}()
end

## Primitives

function DI.value_and_pushforward!(
    dy, ::PolyesterForwardDiffBackend{custom}, f, x, dx
) where {custom}
    return DI.value_and_pushforward!(dy, ForwardDiffBackend{custom}(), f, x, dx)
end

## Utilities

function DI.value_and_gradient!(
    grad::AbstractArray, ::PolyesterForwardDiffBackend{true,C}, f, x::AbstractArray
) where {C}
    y = f(x)
    threaded_gradient!(f, grad, x, Chunk{C}())
    return y, grad
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, ::PolyesterForwardDiffBackend{true,C}, f, x::AbstractArray
) where {C}
    y = f(x)
    threaded_jacobian!(f, jac, x, Chunk{C}())
    return y, jac
end

end # module
