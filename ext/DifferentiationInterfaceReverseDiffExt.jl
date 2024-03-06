module DifferentiationInterfaceReverseDiffExt

using DifferentiationInterface
using DiffResults: DiffResults
using DocStringExtensions
using LinearAlgebra: mul!
using ReverseDiff: gradient!, jacobian!

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    res = DiffResults.DiffResult(zero(Y), dx)
    res = gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy .* DiffResults.gradient(res)
    return y, dx
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    res = DiffResults.DiffResult(similar(dy), similar(dy, length(dy), length(x)))
    res = jacobian!(res, f, x)
    y = DiffResults.value(res)
    J = DiffResults.jacobian(res)
    mul!(vec(dx), transpose(J), vec(dy))
    return y, dx
end

end # module
