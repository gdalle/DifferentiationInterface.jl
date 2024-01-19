module DifferentiationInterfaceFiniteDiffExt

using DifferentiationInterface
using DocStringExtensions
using FiniteDiff
using LinearAlgebra

## Pushforward

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:Number}
    new_dy = FiniteDiff.finite_difference_derivative(f, x) * dx
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:AbstractArray}
    new_dy = FiniteDiff.finite_difference_derivative(f, x)
    dy .= new_dy .* dx
    return dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Number}
    g = FiniteDiff.finite_difference_gradient(f, x)
    new_dy = dot(g, dx)
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    J = FiniteDiff.finite_difference_jacobian(f, x)
    mul!(dy, J, dx)
    return dy
end

## Pullback

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    dx::X, ::FiniteDiffBackend, f, x::X, dy::Y
) where {X<:Real,Y<:Real}
    new_dx = dy * FiniteDiff.finite_difference_derivative(f, x)
    return new_dx
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    dx::X, ::FiniteDiffBackend, f, x::X, dy::Y
) where {X<:Real,Y<:AbstractArray}
    new_dx = dot(dy, FiniteDiff.finite_difference_derivative(f, x))
    return new_dx
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    dx::X, ::FiniteDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    g = FiniteDiff.finite_difference_gradient(f, x)
    dx .= g .* dy
    return dx
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    dx::X, ::FiniteDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    J = FiniteDiff.finite_difference_jacobian(f, x)
    mul!(dx, transpose(J), dy)
    return dx
end

end
