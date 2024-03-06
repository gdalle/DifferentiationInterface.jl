module DifferentiationInterfaceReverseDiffExt

using DifferentiationInterface: ReverseDiffBackend
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using LinearAlgebra: mul!
using ReverseDiff: gradient, gradient!, jacobian, jacobian!

## Primitives

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pullback!(
    dx, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    res = DiffResults.DiffResult(zero(Y), dx)
    res = gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy .* DiffResults.gradient(res)
    return y, dx
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pullback!(
    dx, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    res = DiffResults.DiffResult(similar(dy), similar(dy, length(dy), length(x)))
    res = jacobian!(res, f, x)
    y = DiffResults.value(res)
    J = DiffResults.jacobian(res)
    mul!(vec(dx), transpose(J), vec(dy))
    return y, dx
end

## Special cases (TODO: use DiffResults)

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient(::ReverseDiffBackend, f, x::AbstractArray)
    y = f(x)
    grad = gradient(f, x)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient!(
    grad::AbstractArray, ::ReverseDiffBackend, f, x::AbstractArray
)
    y = f(x)
    gradient!(grad, f, x)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(::ReverseDiffBackend, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, jac
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(
    jac::AbstractMatrix, ::ReverseDiffBackend, f, x::AbstractArray
)
    y = f(x)
    jacobian!(jac, f, x)
    return y, jac
end

end # module
