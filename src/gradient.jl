## Allocating

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient(
    f::F, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
) where {F}
    return value_and_gradient_aux(f, backend, x, extras, supports_pullback(backend))
end

function value_and_gradient_aux(f::F, backend, x, extras, ::PullbackSupported) where {F}
    return value_and_pullback(f, backend, x, one(myeltype(x)), extras)
end

function value_and_gradient_aux(
    f::F, backend, x::Number, extras, ::PullbackNotSupported
) where {F}
    return value_and_derivative(f, backend, x, extras)
end

function value_and_gradient_aux(
    f::F, backend, x::AbstractArray, extras, ::PullbackNotSupported
) where {F}
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
    f::F, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
) where {F}
    return value_and_gradient_aux!!(f, grad, backend, x, extras, supports_pullback(backend))
end

function value_and_gradient_aux!!(
    f::F, grad, backend, x, extras, ::PullbackSupported
) where {F}
    return value_and_pullback!!(f, grad, backend, x, one(myeltype(grad)), extras)
end

function value_and_gradient_aux!!(
    f::F, grad, backend, x::Number, extras, ::PullbackNotSupported
) where {F}
    return value_and_derivative(f, backend, x, extras)
end

function value_and_gradient_aux!!(
    f::F, grad, backend, x::AbstractArray, extras, ::PullbackNotSupported
) where {F}
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
function gradient(
    f::F, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
) where {F}
    return last(value_and_gradient(f, backend, x, extras))
end

"""
    gradient!!(f, grad, backend, x, [extras]) -> grad
"""
function gradient!!(
    f::F, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
) where {F}
    return last(value_and_gradient!!(f, grad, backend, x, extras))
end
