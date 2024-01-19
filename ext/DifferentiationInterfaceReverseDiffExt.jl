module DifferentiationInterfaceReverseDiffExt

using DifferentiationInterface
using DocStringExtensions
using ReverseDiff
using LinearAlgebra

"""
$(TYPEDSIGNATURES)
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
"""
function DifferentiationInterface.pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ReverseDiff.jacobian(f, x)
    mul!(dx, transpose(J), dy)
    return dx
end

end
