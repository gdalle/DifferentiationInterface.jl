module DifferentiationInterfaceForwardDiffExt

using ADTypes: AutoForwardDiff
using DifferentiationInterface: CustomImplem
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

function DI.value_and_pushforward!(
    _dy::Y, ::AutoForwardDiff, f, x::X, dx
) where {X<:Real,Y<:Real}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::AutoForwardDiff, f, x::X, dx
) where {X<:Real,Y<:AbstractArray}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

function DI.value_and_pushforward!(
    _dy::Y, ::AutoForwardDiff, f, x::X, dx
) where {X<:AbstractArray,Y<:Real}
    T = typeof(Tag(f, X))  # TODO: unsure
    xdual = Dual{T}.(x, dx)  # TODO: allocation
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::AutoForwardDiff, f, x::X, dx
) where {X<:AbstractArray,Y<:AbstractArray}
    T = typeof(Tag(f, X))  # TODO: unsure
    xdual = Dual{T}.(x, dx)  # TODO: allocation
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

## Utilities (TODO: use DiffResults)

function DI.value_and_derivative(::CustomImplem, ::AutoForwardDiff, f, x::Number)
    y = f(x)
    der = derivative(f, x)
    return y, der
end

function DI.value_and_multiderivative(::CustomImplem, ::AutoForwardDiff, f, x::Number)
    y = f(x)
    multider = derivative(f, x)
    return y, multider
end

function DI.value_and_multiderivative!(
    ::CustomImplem, multider::AbstractArray, ::AutoForwardDiff, f, x::Number
)
    y = f(x)
    derivative!(multider, f, x)
    return y, multider
end

function DI.value_and_gradient(::CustomImplem, ::AutoForwardDiff, f, x::AbstractArray)
    y = f(x)
    grad = gradient(f, x)
    return y, grad
end

function DI.value_and_gradient!(
    ::CustomImplem, grad::AbstractArray, ::AutoForwardDiff, f, x::AbstractArray
)
    y = f(x)
    gradient!(grad, f, x)
    return y, grad
end

function DI.value_and_jacobian(::CustomImplem, ::AutoForwardDiff, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, jac
end

function DI.value_and_jacobian!(
    ::CustomImplem, jac::AbstractMatrix, ::AutoForwardDiff, f, x::AbstractArray
)
    y = f(x)
    jacobian!(jac, f, x)
    return y, jac
end

end # module
