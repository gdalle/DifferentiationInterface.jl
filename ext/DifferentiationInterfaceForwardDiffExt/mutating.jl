## Pushforward

function DI.value_and_pushforward!(
    y::AbstractArray, dy::AbstractArray, ::AutoForwardDiff, f!, x::Real, dx, extras::Nothing
)
    T = tag_type(f!, x)
    xdual = Dual{T}(x, dx)
    ydual = Dual{T}.(y, dy)
    f!(ydual, xdual)
    y .= value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

function DI.value_and_pushforward!(
    y::AbstractArray,
    dy::AbstractArray,
    ::AutoForwardDiff,
    f!,
    x::AbstractArray,
    dx,
    extras::Nothing,
)
    T = tag_type(f!, x)
    xdual = Dual{T}.(x, dx)
    ydual = Dual{T}.(y, dy)
    f!(ydual, xdual)
    y .= value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

## Multiderivative

function DI.prepare_multiderivative(::AutoForwardDiff, f!, x::Number, y::AbstractArray)
    return DerivativeConfig(f!, y, x)
end

function DI.value_and_multiderivative!(
    y::AbstractArray,
    multider::AbstractArray,
    ::AutoForwardDiff,
    f!,
    x::Number,
    config::DerivativeConfig,
)
    result = DiffResults.DiffResult(y, multider)
    result = derivative!(result, f!, y, x, config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

## Jacobian

function DI.prepare_jacobian(
    backend::AutoForwardDiff, f!, x::AbstractArray, y::AbstractArray
)
    return JacobianConfig(f!, y, x, choose_chunk(backend, x))
end

function DI.value_and_jacobian!(
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AutoForwardDiff,
    f!,
    x::AbstractArray,
    config::JacobianConfig,
)
    result = DiffResults.DiffResult(y, jac)
    result = jacobian!(result, f!, y, x, config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end
