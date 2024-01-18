module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DocStringExtensions
using ForwardDiff
using LinearAlgebra

## Jacobian-vector products

"""
$(TYPEDSIGNATURES)

JVP for a scalar -> scalar function: derivative multiplied by the tangent.
"""
function DifferentiationInterface.jvp!(
    dys::Tuple{Y}, ::ForwardDiffBackend, f, xs::Tuple{X}, dxs::Tuple{X}
) where {X<:Real,Y<:Real}
    x, dx, dy = only(xs), only(dxs), only(dys)
    new_dy = ForwardDiff.derivative(f, x) * dx
    return (new_dy,)
end

"""
$(TYPEDSIGNATURES)

JVP for a scalar -> vector function: vector of derivatives multiplied componentwise by the tangent.
"""
function DifferentiationInterface.jvp!(
    dys::Tuple{Y}, ::ForwardDiffBackend, f, xs::Tuple{X}, dxs::Tuple{X}
) where {X<:Real,Y<:AbstractArray}
    x, dx, dy = only(xs), only(dxs), only(dys)
    ForwardDiff.derivative!(dy, f, x)
    dy .*= dx
    return dys
end

"""
$(TYPEDSIGNATURES)

JVP for a vector -> scalar function: dot product between the gradient vector and the vector of tangents.
"""
function DifferentiationInterface.jvp!(
    dys::Tuple{Y}, ::ForwardDiffBackend, f, xs::Tuple{X}, dxs::Tuple{X}
) where {X<:AbstractArray,Y<:Real}
    x, dx, dy = only(xs), only(dxs), only(dys)
    g = ForwardDiff.gradient(f, x)  # allocates
    new_dy = dot(g, dx)
    return (new_dy,)
end

"""
$(TYPEDSIGNATURES)

JVP for a vector -> vector function: Jacobian matrix multiplied by the vector of tangents.
"""
function DifferentiationInterface.jvp!(
    dys::Tuple{Y}, ::ForwardDiffBackend, f, xs::Tuple{X}, dxs::Tuple{X}
) where {X<:AbstractArray,Y<:AbstractArray}
    x, dx, dy = only(xs), only(dxs), only(dys)
    J = ForwardDiff.jacobian(f, x)  # allocates
    mul!(dy, J, dx)
    return dys
end

## Vector-Jacobian products

"""
$(TYPEDSIGNATURES)

VJP for a scalar -> scalar function: derivative multiplied by the cotangent.
"""
function DifferentiationInterface.vjp!(
    dxs::Tuple{X}, ::ForwardDiffBackend, f, xs::Tuple{X}, dys::Tuple{Y}
) where {X<:Real,Y<:Real}
    x, dx, dy = only(xs), only(dxs), only(dys)
    new_dx = dy * ForwardDiff.derivative(f, x)
    return (new_dx,)
end

"""
$(TYPEDSIGNATURES)

VJP for a scalar -> vector function: dot product between the vector of derivatives and the vector of cotangents.
"""
function DifferentiationInterface.vjp!(
    dxs::Tuple{X}, ::ForwardDiffBackend, f, xs::Tuple{X}, dys::Tuple{Y}
) where {X<:Real,Y<:AbstractArray}
    x, dx, dy = only(xs), only(dxs), only(dys)
    new_dx = dot(dy, ForwardDiff.derivative(f, x))
    return (new_dx,)
end

"""
$(TYPEDSIGNATURES)

VJP for a vector -> scalar function: gradient vector multiplied componentwise by the cotangent.
"""
function DifferentiationInterface.vjp!(
    dxs::Tuple{X}, ::ForwardDiffBackend, f, xs::Tuple{X}, dys::Tuple{Y}
) where {X<:AbstractArray,Y<:Real}
    x, dx, dy = only(xs), only(dxs), only(dys)
    ForwardDiff.gradient!(dx, f, x)
    dx .*= dy
    return dxs
end

"""
$(TYPEDSIGNATURES)

VJP for a vector -> vector function: transposed Jacobian matrix multiplied by the vector of tangents.
"""
function DifferentiationInterface.vjp!(
    dxs::Tuple{X}, ::ForwardDiffBackend, f, xs::Tuple{X}, dys::Tuple{X}
) where {X<:AbstractArray,Y<:AbstractArray}
    x, dx, dy = only(xs), only(dxs), only(dys)
    J = ForwardDiff.jacobian(f, x)  # allocates
    mul!(dx, transpose(J), dy)
    return dxs
end

end
