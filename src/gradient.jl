## Allocating

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient(
    f, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    return value_and_gradient_aux(f, backend, x, extras, supports_pullback(backend))
end

function value_and_gradient_aux(f, backend, x, extras, ::PullbackSupported)
    return value_and_pullback(f, backend, x, true, extras)
end

function value_and_gradient_aux(f, backend, x::Number, extras, ::PullbackNotSupported)
    return value_and_pushforward(f, backend, x, one(x), extras)
end

function value_and_gradient_aux(
    f, backend, x::AbstractArray, extras, ::PullbackNotSupported
)
    y = f(x)
    grad = map(CartesianIndices(x)) do j
        dx_j = basisarray(backend, x, j)
        last(value_and_pushforward(f, backend, x, dx_j, extras))
    end
    return y, grad
end

"""
    value_and_gradient!!(f, grad, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient!!(
    f, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    return value_and_gradient_aux!!(f, grad, backend, x, extras, supports_pullback(backend))
end

function value_and_gradient_aux!!(f, grad, backend, x, extras, ::PullbackSupported)
    return value_and_pullback!!(f, grad, backend, x, true, extras)
end

function value_and_gradient_aux!!(
    f, grad, backend, x::Number, extras, ::PullbackNotSupported
)
    return value_and_pushforward(f, backend, x, one(x), extras)
end

function value_and_gradient_aux!!(
    f, grad, backend, x::AbstractArray, extras, ::PullbackNotSupported
)
    y = f(x)
    map!(grad, CartesianIndices(x)) do j
        dx_j = basisarray(backend, x, j)
        pushforward(f, backend, x, dx_j, extras)
    end
    return y, grad
end

"""
    gradient(f, backend, x, [extras]) -> grad
"""
function gradient(f, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x))
    return last(value_and_gradient(f, backend, x, extras))
end

"""
    gradient!!(f, grad, backend, x, [extras]) -> grad
"""
function gradient!!(
    f, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    return last(value_and_gradient!!(f, grad, backend, x, extras))
end
