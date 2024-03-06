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

const DEFAULT_FDTYPE = Val{:central}
const DEFAULT_INPLACE = Val{false}  # see https://docs.sciml.ai/FiniteDiff/stable/#f-Definitions

## Primitives

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx
) where {X<:Number,Y<:Number}
    y = f(x)
    der = finite_difference_derivative(f, x, DEFAULT_FDTYPE, eltype(dy), y)
    new_dy = der * dx
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx
) where {X<:Number,Y<:AbstractArray}
    y = f(x)
    finite_difference_gradient!(dy, f, x, DEFAULT_FDTYPE, eltype(dy), DEFAULT_INPLACE, y)
    dy .*= dx
    return y, dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx
) where {X<:AbstractArray,Y<:Number}
    y = f(x)
    g = finite_difference_gradient(f, x, DEFAULT_FDTYPE, eltype(dy), DEFAULT_INPLACE, y)
    new_dy = dot(g, dx)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx
) where {X<:AbstractArray,Y<:AbstractArray}
    y = f(x)
    J = finite_difference_jacobian(f, x, DEFAULT_FDTYPE, eltype(dy))
    mul!(vec(dy), J, vec(dx))
    return y, dy
end

## Special cases

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_derivative(::FiniteDiffBackend, f, x::Number)
    y = f(x)
    der = finite_difference_derivative(f, x, DEFAULT_FDTYPE, eltype(y), y)
    return y, der
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_multiderivative!(
    multider::AbstractArray, ::FiniteDiffBackend, f, x::Number
)
    y = f(x)
    finite_difference_gradient!(
        multider, f, x, DEFAULT_FDTYPE, eltype(y), DEFAULT_INPLACE, y
    )
    return y, multider
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_multiderivative(::FiniteDiffBackend, f, x::Number)
    y = f(x)
    multider = finite_difference_gradient(
        f, x, DEFAULT_FDTYPE, eltype(y), DEFAULT_INPLACE, y
    )
    return y, multider
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient!(
    grad::AbstractArray, ::FiniteDiffBackend, f, x::AbstractArray
)
    y = f(x)
    finite_difference_gradient!(grad, f, x, DEFAULT_FDTYPE, eltype(y), DEFAULT_INPLACE, y)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient(::FiniteDiffBackend, f, x::AbstractArray)
    y = f(x)
    grad = finite_difference_gradient(f, x, DEFAULT_FDTYPE, eltype(y), DEFAULT_INPLACE, y)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(::FiniteDiffBackend, f, x::AbstractArray)
    y = f(x)
    jac = finite_difference_jacobian(f, x, DEFAULT_FDTYPE, eltype(y))
    return y, jac
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::FiniteDiffBackend, f, x::AbstractArray
)
    y, new_jac = DI.value_and_jacobian(backend, f, x)
    jac .= new_jac
    return y, jac
end

end # module
