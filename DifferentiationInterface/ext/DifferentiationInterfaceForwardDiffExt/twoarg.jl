DI.prepare_pushforward(f!, ::AnyAutoForwardDiff, y, x) = NoPushforwardExtras()

function DI.value_and_pushforward(f!, y, ::AnyAutoForwardDiff, x, dx, ::NoPushforwardExtras)
    T = tag_type(f!, x)
    xdual = make_dual(T, x, dx)
    ydual = make_dual(T, y, similar(y))
    f!(ydual, xdual)
    y = myvalue!(T, y, ydual)
    dy = myderivative(T, ydual)
    return y, dy
end

## Derivative

struct ForwardDiffTwoArgDerivativeExtras{C} <: DerivativeExtras
    config::C
end

function DI.prepare_derivative(f!, ::AnyAutoForwardDiff, y::AbstractArray, x::Number)
    return ForwardDiffTwoArgDerivativeExtras(DerivativeConfig(f!, y, x))
end

function DI.value_and_derivative(
    f!,
    y::AbstractArray,
    ::AnyAutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    result = DiffResult(y, similar(y))
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AnyAutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    result = DiffResult(y, der)
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!,
    y::AbstractArray,
    ::AnyAutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    der = derivative(f!, y, x, extras.config)
    return der
end

function DI.derivative!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AnyAutoForwardDiff,
    x::Number,
    extras::ForwardDiffTwoArgDerivativeExtras,
)
    der = derivative!(der, f!, y, x, extras.config)
    return der
end

## Jacobian

struct ForwardDiffTwoArgJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(
    f!, backend::AnyAutoForwardDiff, y::AbstractArray, x::AbstractArray
)
    return ForwardDiffTwoArgJacobianExtras(
        JacobianConfig(f!, y, x, choose_chunk(backend, x))
    )
end

function DI.value_and_jacobian(
    f!,
    y::AbstractArray,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    jac = jacobian(f!, y, x, extras.config)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    result = DiffResult(y, jac)
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!,
    y::AbstractArray,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    jac = jacobian(f!, y, x, extras.config)
    return jac
end

function DI.jacobian!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffTwoArgJacobianExtras,
)
    jac = jacobian!(jac, f!, y, x, extras.config)
    return jac
end
