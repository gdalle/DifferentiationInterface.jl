module DifferentiationInterfaceReverseDiffExt

using DifferentiationInterface
using DocStringExtensions
using ReverseDiff
using LinearAlgebra

## VJP

"""
$(TYPEDSIGNATURES)

VJP for a vector -> scalar function: gradient vector multiplied componentwise by the cotangent.
"""
function DifferentiationInterface.pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    ReverseDiff.gradient!(dx, f, x)
    dx .*= dy
    return dx
end

"""
$(TYPEDSIGNATURES)

VJP for a vector -> vector function: transposed Jacobian matrix multiplied by the cotangent vector.
"""
function DifferentiationInterface.pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ReverseDiff.jacobian(f, x)  # TODO: allocates
    mul!(dx, transpose(J), dy)
    return dx
end

end
