module DifferentiationInterfaceForwardDiffExt

using ADTypes: AutoForwardDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff:
    Chunk,
    Dual,
    GradientConfig,
    JacobianConfig,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: mul!

## Pushforward

function DI.value_and_pushforward!(
    _dy::Y, ::AutoForwardDiff, f, x::X, dx, extras::Nothing=nothing
) where {X<:Real,Y<:Real}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::AutoForwardDiff, f, x::X, dx, extras::Nothing=nothing
) where {X<:Real,Y<:AbstractArray}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

function DI.value_and_pushforward!(
    _dy::Y, ::AutoForwardDiff, f, x::X, dx, extras::Nothing=nothing
) where {X<:AbstractArray,Y<:Real}
    T = typeof(Tag(f, eltype(X)))
    xdual = Dual{T}.(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Y, ::AutoForwardDiff, f, x::X, dx, extras::Nothing=nothing
) where {X<:AbstractArray,Y<:AbstractArray}
    T = typeof(Tag(f, eltype(X)))
    xdual = Dual{T}.(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

## Derivative

function DI.derivative(::AutoForwardDiff, f, x::Number, extras::Nothing)
    return derivative(f, x)
end

## Multiderivative

function DI.multiderivative!(
    multider::AbstractArray, ::AutoForwardDiff, f, x::Number, extras::Nothing
)
    derivative!(multider, f, x)
    return multider
end

function DI.multiderivative(::AutoForwardDiff, f, x::Number, extras::Nothing)
    return derivative(f, x)
end

## Gradient

function DI.value_and_gradient!(
    grad::AbstractArray, backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_gradient!(grad, backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoForwardDiff, f, x::AbstractArray, extras::GradientConfig
)
    result = DiffResults.DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, f, x, extras)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_gradient(backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.value_and_gradient(
    ::AutoForwardDiff, f, x::AbstractArray, extras::GradientConfig
)
    result = DiffResults.GradientResult(x)
    result = gradient!(result, f, x, extras)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.gradient!(
    grad::AbstractArray, backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.gradient!(grad, backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.gradient!(
    grad::AbstractArray, ::AutoForwardDiff, f, x::AbstractArray, extras::GradientConfig
)
    gradient!(grad, f, x, extras)
    return grad
end

function DI.gradient(backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing)
    return DI.gradient(backend, f, x, DI.prepare_gradient(backend, f, x))
end

function DI.gradient(
    ::AutoForwardDiff, f, x::AbstractArray, extras::Union{Nothing,GradientConfig}
)
    return gradient(f, x, extras)
end

## Jacobian

function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_jacobian!(jac, backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, ::AutoForwardDiff, f, x::AbstractArray, extras::JacobianConfig
)
    y = f(x)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, f, x, extras)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.value_and_jacobian(backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.value_and_jacobian(
    ::AutoForwardDiff, f, x::AbstractArray, extras::JacobianConfig
)
    return f(x), jacobian(f, x, extras)
end

function DI.jacobian!(
    jac::AbstractMatrix, backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing
)
    return DI.jacobian!(jac, backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.jacobian!(
    jac::AbstractMatrix, ::AutoForwardDiff, f, x::AbstractArray, extras::JacobianConfig
)
    jacobian!(jac, f, x, extras)
    return jac
end

function DI.jacobian(backend::AutoForwardDiff, f, x::AbstractArray, extras::Nothing)
    return DI.jacobian(backend, f, x, DI.prepare_jacobian(backend, f, x))
end

function DI.jacobian(::AutoForwardDiff, f, x::AbstractArray, extras::JacobianConfig)
    return jacobian(f, x, extras)
end

## Preparation

function DI.prepare_gradient(::AutoForwardDiff{nothing}, f, x::AbstractArray)
    return GradientConfig(f, x, Chunk(x))
end

function DI.prepare_gradient(::AutoForwardDiff{C}, f, x::AbstractArray) where {C}
    return GradientConfig(f, x, Chunk{C}())
end

function DI.prepare_jacobian(::AutoForwardDiff{nothing}, f, x::AbstractArray)
    return JacobianConfig(f, x, Chunk(x))
end

function DI.prepare_jacobian(::AutoForwardDiff{C}, f, x::AbstractArray) where {C}
    return JacobianConfig(f, x, Chunk{C}())
end

end # module
