module DifferentiationInterfaceFiniteDiffExt

using ADTypes: AutoFiniteDiff
import DifferentiationInterface as DI
using DocStringExtensions
using FiniteDiff:
    finite_difference_derivative,
    finite_difference_gradient,
    finite_difference_gradient!,
    finite_difference_jacobian
using LinearAlgebra: dot, mul!

# see https://docs.sciml.ai/FiniteDiff/stable/#f-Definitions
const FUNCTION_INPLACE = Val{true}
const FUNCTION_NOT_INPLACE = Val{false}

## Primitives

function DI.value_and_pushforward!(
    _dy::Number, ::AutoFiniteDiff{fdtype}, f, x, dx, extras::Nothing
) where {fdtype}
    y = f(x)
    step(t::Number)::Number = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(step, zero(eltype(dx)), fdtype, eltype(y), y)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::AbstractArray, ::AutoFiniteDiff{fdtype}, f, x, dx, extras::Nothing
) where {fdtype}
    y = f(x)
    step(t::Number)::AbstractArray = f(x .+ t .* dx)
    finite_difference_gradient!(
        dy, step, zero(eltype(dx)), fdtype, eltype(y), FUNCTION_NOT_INPLACE, y
    )
    return y, dy
end

## Utilities

function DI.value_and_derivative(
    ::AutoFiniteDiff{fdtype}, f, x::Number, extras::Nothing
) where {fdtype}
    y = f(x)
    der = finite_difference_derivative(f, x, fdtype, eltype(y), y)
    return y, der
end

function DI.value_and_multiderivative!(
    multider::AbstractArray, ::AutoFiniteDiff{fdtype}, f, x::Number, extras::Nothing
) where {fdtype}
    y = f(x)
    finite_difference_gradient!(multider, f, x, fdtype, eltype(y), FUNCTION_NOT_INPLACE, y)
    return y, multider
end

function DI.value_and_multiderivative(
    ::AutoFiniteDiff{fdtype}, f, x::Number, extras::Nothing
) where {fdtype}
    y = f(x)
    multider = finite_difference_gradient(f, x, fdtype, eltype(y), FUNCTION_NOT_INPLACE, y)
    return y, multider
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoFiniteDiff{fdtype}, f, x::AbstractArray, extras::Nothing
) where {fdtype}
    y = f(x)
    finite_difference_gradient!(grad, f, x, fdtype, eltype(y), FUNCTION_NOT_INPLACE, y)
    return y, grad
end

function DI.value_and_gradient(
    ::AutoFiniteDiff{fdtype}, f, x::AbstractArray, extras::Nothing
) where {fdtype}
    y = f(x)
    grad = finite_difference_gradient(f, x, fdtype, eltype(y), FUNCTION_NOT_INPLACE, y)
    return y, grad
end

function DI.value_and_jacobian(
    ::AutoFiniteDiff{fdtype}, f, x::AbstractArray, extras::Nothing
) where {fdtype}
    y = f(x)
    jac = finite_difference_jacobian(f, x, fdtype, eltype(y))
    return y, jac
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::AutoFiniteDiff, f, x::AbstractArray, extras::Nothing
)
    y, new_jac = DI.value_and_jacobian(backend, f, x, extras)
    jac .= new_jac
    return y, jac
end

end # module
