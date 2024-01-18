module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DocStringExtensions
using ForwardDiff
using LinearAlgebra

## JVP

"""
$(TYPEDSIGNATURES)

JVP for a scalar -> scalar function: derivative multiplied by the tangent.
"""
function DifferentiationInterface.jvp!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:Real}
    new_dy = ForwardDiff.derivative(f, x) * dx
    return new_dy
end

"""
$(TYPEDSIGNATURES)

JVP for a scalar -> vector function: derivative vector multiplied componentwise by the tangent.
"""
function DifferentiationInterface.jvp!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:AbstractArray}
    ForwardDiff.derivative!(dy, f, x)
    dy .*= dx
    return dy
end

"""
$(TYPEDSIGNATURES)

JVP for a vector -> scalar function: dot product between the gradient vector and the tangent vector.
"""
function DifferentiationInterface.jvp!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Real}
    g = ForwardDiff.gradient(f, x)  # TODO: allocates
    new_dy = dot(g, dx)
    return new_dy
end

"""
$(TYPEDSIGNATURES)

JVP for a vector -> vector function: Jacobian matrix multiplied by the tangent vector.
"""
function DifferentiationInterface.jvp!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ForwardDiff.jacobian(f, x)  # TODO: allocates
    mul!(dy, J, dx)
    return dy
end

## VJP

"""
$(TYPEDSIGNATURES)

VJP for a scalar -> scalar function: derivative multiplied by the cotangent.
"""
function DifferentiationInterface.vjp!(
    dx::X, ::ForwardDiffBackend, f, x::X, dy::Y
) where {X<:Real,Y<:Real}
    new_dx = dy * ForwardDiff.derivative(f, x)
    return new_dx
end

"""
$(TYPEDSIGNATURES)

VJP for a scalar -> vector function: dot product between the derivative vector and the cotangent vector.
"""
function DifferentiationInterface.vjp!(
    dx::X, ::ForwardDiffBackend, f, x::X, dy::Y
) where {X<:Real,Y<:AbstractArray}
    new_dx = dot(dy, ForwardDiff.derivative(f, x))  # TODO: allocates
    return new_dx
end

"""
$(TYPEDSIGNATURES)

VJP for a vector -> scalar function: gradient vector multiplied componentwise by the cotangent.
"""
function DifferentiationInterface.vjp!(
    dx::X, ::ForwardDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    ForwardDiff.gradient!(dx, f, x)
    dx .*= dy
    return dx
end

"""
$(TYPEDSIGNATURES)

VJP for a vector -> vector function: transposed Jacobian matrix multiplied by the cotangent vector.
"""
function DifferentiationInterface.vjp!(
    dx::X, ::ForwardDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ForwardDiff.jacobian(f, x)  # TODO: allocates
    mul!(dx, transpose(J), dy)
    return dx
end

end
