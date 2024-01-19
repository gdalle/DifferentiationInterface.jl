module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DocStringExtensions
using ForwardDiff
using LinearAlgebra

## JVP

"""
$(TYPEDSIGNATURES)

JVP for a vector -> scalar function: dot product between the gradient vector and the tangent vector.
"""
function DifferentiationInterface.pushforward!(
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
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ForwardDiff.jacobian(f, x)  # TODO: allocates
    mul!(dy, J, dx)
    return dy
end

end
