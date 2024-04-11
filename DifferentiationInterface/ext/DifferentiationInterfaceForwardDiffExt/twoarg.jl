DI.prepare_pushforward(f!, ::AnyAutoForwardDiff, y, x) = NoPushforwardExtras()

function DI.value_and_pushforward!(
    f!, y, dy, ::AnyAutoForwardDiff, x, dx, ::NoPushforwardExtras
)
    T = tag_type(f!, x)
    xdual = make_dual(T, x, dx)
    ydual = make_dual(T, y, dy)
    f!(ydual, xdual)
    y = myvalue!(T, y, ydual)
    dy = myderivative!(T, dy, ydual)
    return y, dy
end

## Derivative

struct ForwardDiffMutatingDerivativeExtras{C} <: DerivativeExtras
    config::C
end

function DI.prepare_derivative(f!, ::AnyAutoForwardDiff, y::AbstractArray, x::Number)
    return ForwardDiffMutatingDerivativeExtras(DerivativeConfig(f!, y, x))
end

function DI.value_and_derivative!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AnyAutoForwardDiff,
    x::Number,
    extras::ForwardDiffMutatingDerivativeExtras,
)
    result = DiffResult(y, der)
    result = derivative!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AnyAutoForwardDiff,
    x::Number,
    extras::ForwardDiffMutatingDerivativeExtras,
)
    der = derivative!(der, f!, y, x, extras.config)
    return der
end

## Jacobian

struct ForwardDiffMutatingJacobianExtras{C} <: JacobianExtras
    config::C
end

function DI.prepare_jacobian(
    f!, backend::AnyAutoForwardDiff, y::AbstractArray, x::AbstractArray
)
    return ForwardDiffMutatingJacobianExtras(
        JacobianConfig(f!, y, x, choose_chunk(backend, x))
    )
end

function DI.value_and_jacobian!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffMutatingJacobianExtras,
)
    result = DiffResult(y, jac)
    result = jacobian!(result, f!, y, x, extras.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoForwardDiff,
    x::AbstractArray,
    extras::ForwardDiffMutatingJacobianExtras,
)
    jac = jacobian!(jac, f!, y, x, extras.config)
    return jac
end
