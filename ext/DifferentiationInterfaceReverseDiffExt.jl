module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using LinearAlgebra: mul!
using ReverseDiff: gradient, gradient!, jacobian, jacobian!

## Limitations

DI.handles_input_type(::AutoReverseDiff, ::Type{<:Number}) = false

## Primitives

function DI.value_and_pullback!(
    dx, ::AutoReverseDiff, f, x::X, dy::Y, extras::Nothing=nothing
) where {X<:AbstractArray,Y<:Real}
    res = DiffResults.DiffResult(zero(Y), dx)
    res = gradient!(res, f, x)
    y = DiffResults.value(res)
    dx .= dy .* DiffResults.gradient(res)
    return y, dx
end

function DI.value_and_pullback!(
    dx, ::AutoReverseDiff, f, x::X, dy::Y, extras::Nothing=nothing
) where {X<:AbstractArray,Y<:AbstractArray}
    res = DiffResults.DiffResult(similar(dy), similar(dy, length(dy), length(x)))
    res = jacobian!(res, f, x)
    y = DiffResults.value(res)
    J = DiffResults.jacobian(res)
    mul!(vec(dx), transpose(J), vec(dy))
    return y, dx
end

## Utilities (TODO: use DiffResults)

function DI.value_and_gradient(
    ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    grad = gradient(f, x)
    return y, grad
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    gradient!(grad, f, x)
    return y, grad
end

function DI.value_and_jacobian(
    ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    jac = jacobian(f, x)
    return y, jac
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, ::AutoReverseDiff, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    jacobian!(jac, f, x)
    return y, jac
end

end # module
