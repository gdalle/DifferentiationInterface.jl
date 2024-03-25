## Pushforward

function DI.value_and_pushforward(f, ::AutoForwardDiff, x, dx, extras::Nothing)
    T = tag_type(f, x)
    xdual = make_dual(T, x, dx)
    ydual = f(xdual)
    y = myvalue(T, ydual)
    new_dy = myderivative(T, ydual)
    return y, new_dy
end

## Gradient

function DI.prepare_gradient(f, backend::AutoForwardDiff, x::AbstractArray)
    return GradientConfig(f, x, choose_chunk(backend, x))
end

function DI.value_and_gradient!!(
    f, grad::AbstractArray, ::AutoForwardDiff, x::AbstractArray, config::GradientConfig
)
    result = DiffResult(zero(eltype(x)), grad)
    result = gradient!(result, f, x, config)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function DI.value_and_gradient(
    f, backend::AutoForwardDiff, x::AbstractArray, config::GradientConfig
)
    grad = similar(x)
    return DI.value_and_gradient!!(f, grad, backend, x, config)
end

function DI.gradient!!(
    f, grad::AbstractArray, ::AutoForwardDiff, x::AbstractArray, config::GradientConfig
)
    return gradient!(grad, f, x, config)
end

function DI.gradient(f, ::AutoForwardDiff, x::AbstractArray, config::GradientConfig)
    return gradient(f, x, config)
end

## Jacobian

function DI.prepare_jacobian(f, backend::AutoForwardDiff, x::AbstractArray)
    return JacobianConfig(f, x, choose_chunk(backend, x))
end

function DI.value_and_jacobian!!(
    f, jac::AbstractMatrix, ::AutoForwardDiff, x::AbstractArray, config::JacobianConfig
)
    y = f(x)
    result = DiffResult(y, jac)
    result = jacobian!(result, f, x, config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian(
    f, ::AutoForwardDiff, x::AbstractArray, config::JacobianConfig
)
    return f(x), jacobian(f, x, config)
end

function DI.jacobian!!(
    f, jac::AbstractMatrix, ::AutoForwardDiff, x::AbstractArray, config::JacobianConfig
)
    return jacobian!(jac, f, x, config)
end

function DI.jacobian(f, ::AutoForwardDiff, x::AbstractArray, config::JacobianConfig)
    return jacobian(f, x, config)
end
