module DifferentiationInterfaceFiniteDiffExt

using DifferentiationInterface: FiniteDiffBackend
import DifferentiationInterface as DI
using DocStringExtensions
using FiniteDiff:
    finite_difference_derivative,
    finite_difference_gradient,
    finite_difference_gradient!,
    finite_difference_jacobian
using LinearAlgebra: dot, mul!

const DEFAULT_INPLACE = Val{false}  # see https://docs.sciml.ai/FiniteDiff/stable/#f-Definitions

## Stepping

struct Step{F,X,D}
    f::F
    x::X
    d::D
end

(step::Step)(t::Number) = step.f(step.x + t * step.d)

## Backend construction

"""
$(SIGNATURES)
"""
function DI.FiniteDiffBackend(
    ::Type{fdtype}=Val{:central}; custom::Bool=true
) where {fdtype}
    return FiniteDiffBackend{custom,fdtype}()
end

## Primitives

function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend{custom,fdtype}, f, x::X, dx
) where {X<:Number,Y<:Number,custom,fdtype}
    y = f(x)
    der = finite_difference_derivative(f, x, fdtype, eltype(dy), y)
    new_dy = der * dx
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend{custom,fdtype}, f, x::X, dx
) where {X<:Number,Y<:AbstractArray,custom,fdtype}
    y = f(x)
    finite_difference_gradient!(dy, f, x, fdtype, eltype(dy), DEFAULT_INPLACE, y)
    dy .*= dx
    return y, dy
end

function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend{custom,fdtype}, f, x::X, dx
) where {X<:AbstractArray,Y<:Number,custom,fdtype}
    y = f(x)
    g = finite_difference_gradient(f, x, fdtype, eltype(dy), DEFAULT_INPLACE, y)
    new_dy = dot(g, dx)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend{custom,fdtype}, f, x::X, dx
) where {X<:AbstractArray,Y<:AbstractArray,custom,fdtype}
    y = f(x)
    J = finite_difference_jacobian(f, x, fdtype, eltype(dy))
    mul!(vec(dy), J, vec(dx))
    return y, dy
end

## Utilities

function DI.value_and_derivative(
    ::FiniteDiffBackend{true,fdtype}, f, x::Number
) where {fdtype}
    y = f(x)
    der = finite_difference_derivative(f, x, fdtype, eltype(y), y)
    return y, der
end

function DI.value_and_multiderivative!(
    multider::AbstractArray, ::FiniteDiffBackend{true,fdtype}, f, x::Number
) where {fdtype}
    y = f(x)
    finite_difference_gradient!(multider, f, x, fdtype, eltype(y), DEFAULT_INPLACE, y)
    return y, multider
end

function DI.value_and_multiderivative(
    ::FiniteDiffBackend{true,fdtype}, f, x::Number
) where {fdtype}
    y = f(x)
    multider = finite_difference_gradient(f, x, fdtype, eltype(y), DEFAULT_INPLACE, y)
    return y, multider
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::FiniteDiffBackend{true,fdtype}, f, x::AbstractArray
) where {fdtype}
    y = f(x)
    finite_difference_gradient!(grad, f, x, fdtype, eltype(y), DEFAULT_INPLACE, y)
    return y, grad
end

function DI.value_and_gradient(
    ::FiniteDiffBackend{true,fdtype}, f, x::AbstractArray
) where {fdtype}
    y = f(x)
    grad = finite_difference_gradient(f, x, fdtype, eltype(y), DEFAULT_INPLACE, y)
    return y, grad
end

function DI.value_and_jacobian(
    ::FiniteDiffBackend{true,fdtype}, f, x::AbstractArray
) where {fdtype}
    y = f(x)
    jac = finite_difference_jacobian(f, x, fdtype, eltype(y))
    return y, jac
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::FiniteDiffBackend{true}, f, x::AbstractArray
)
    y, new_jac = DI.value_and_jacobian(backend, f, x)
    jac .= new_jac
    return y, jac
end

end # module
