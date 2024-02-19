module DifferentiationInterfaceReverseDiffExt

using DifferentiationInterface
using DocStringExtensions
using ReverseDiff
using LinearAlgebra

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    res = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy * DiffResults.gradient(res)
    return y, dx
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    res = DiffResults.JacobianResult(x)
    ReverseDiff.jacobian!(res, f, x)
    y = DiffResults.value(res)
    J = DiffResults.jacobian(res)
    mul!(dx, transpose(J), dy)
    return y, dx
end

end # module
