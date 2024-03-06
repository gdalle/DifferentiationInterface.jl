module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface: ForwardDiffBackend
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff:
    Dual,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: mul!

## Primitives

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:Real,Y<:Real}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:Real,Y<:AbstractArray}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:AbstractArray,Y<:Real}
    T = typeof(Tag(f, X))  # TODO: unsure
    xdual = Dual{T}.(x, dx)  # TODO: allocation
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:AbstractArray,Y<:AbstractArray}
    T = typeof(Tag(f, X))  # TODO: unsure
    xdual = Dual{T}.(x, dx)  # TODO: allocation
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

## Special cases (TODO: use DiffResults)

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_derivative(::ForwardDiffBackend, f, x::Number)
    y = f(x)
    der = derivative(f, x)
    return y, der
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_multiderivative(::ForwardDiffBackend, f, x::Number)
    y = f(x)
    multider = derivative(f, x)
    return y, multider
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_multiderivative!(
    multider::AbstractArray, ::ForwardDiffBackend, f, x::Number
)
    y = f(x)
    derivative!(multider, f, x)
    return y, multider
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient(::ForwardDiffBackend, f, x::AbstractArray)
    y = f(x)
    grad = gradient(f, x)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient!(
    grad::AbstractArray, ::ForwardDiffBackend, f, x::AbstractArray
)
    y = f(x)
    gradient!(grad, f, x)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(::ForwardDiffBackend, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, jac
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(
    jac::AbstractMatrix, ::ForwardDiffBackend, f, x::AbstractArray
)
    y = f(x)
    jacobian!(jac, f, x)
    return y, jac
end

end # module
