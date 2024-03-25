function DI.value_and_pushforward!!(f!, y, dy, ::AutoForwardDiff, x, dx, extras::Nothing)
    T = tag_type(f!, x)
    xdual = make_dual(T, x, dx)
    ydual = make_dual(T, y, dy)
    f!(ydual, xdual)
    y = my_value!!(T, y, ydual)
    dy = my_derivative!!(T, dy, ydual)
    return y, dy
end

## Derivative

function DI.prepare_derivative(f!, ::AutoForwardDiff, y::AbstractArray, x::Number)
    return DerivativeConfig(f!, y, x)
end

function DI.value_and_derivative!!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    ::AutoForwardDiff,
    x::Number,
    config::DerivativeConfig,
)
    result = DiffResult(y, der)
    result = derivative!(result, f!, y, x, config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

## Jacobian

function DI.prepare_jacobian(
    f!, backend::AutoForwardDiff, y::AbstractArray, x::AbstractArray
)
    return JacobianConfig(f!, y, x, choose_chunk(backend, x))
end

function DI.value_and_jacobian!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoForwardDiff,
    x::AbstractArray,
    config::JacobianConfig,
)
    result = DiffResult(y, jac)
    result = jacobian!(result, f!, y, x, config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end
