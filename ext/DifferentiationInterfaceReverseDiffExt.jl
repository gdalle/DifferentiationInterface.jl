module DifferentiationInterfaceReverseDiffExt

using DifferentiationInterface
import DifferentiationInterface: value_and_pullback!

using DiffResults: DiffResults
using ReverseDiff: gradient!, jacobian!
using LinearAlgebra: mul!

function DifferentiationInterface.value_and_pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Real}
    res = DiffResults.DiffResult(zero(Y), dx)
    res = gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy .* DiffResults.gradient(res)
    return y, dx
end

function DifferentiationInterface.value_and_pullback!(
    dx::X, ::ReverseDiffBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    res = DiffResults.DiffResult(similar(dy), similar(dy, length(dy), length(x)))
    res = jacobian!(res, f, x)
    y = DiffResults.value(res)
    J = DiffResults.jacobian(res)
    mul!(dx, transpose(J), dy)
    return y, dx
end

end # module
