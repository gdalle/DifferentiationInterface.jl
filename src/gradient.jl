"""
    value_and_gradient!(f, grad, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient!(f::F, grad, backend::AbstractADType, x) where {F}
    return value_and_gradient_aux!(f, grad, backend, x, supports_pullback(backend))
end

function value_and_gradient_aux!(f::F, grad, backend, x, ::PullbackSupported) where {F}
    return value_and_pullback!(f, grad, backend, x, one(myeltype(grad)))
end

function value_and_gradient_aux!(
    f::F, grad, backend, x::Number, ::PullbackNotSupported
) where {F}
    return value_and_derivative(f, backend, x)
end

function value_and_gradient_aux!(
    f::F, grad, backend, x::AbstractArray, ::PullbackNotSupported
) where {F}
    y = f(x)
    for j in CartesianIndices(x)
        dx_j = basisarray(backend, x, j)
        _, grad[j] = value_and_pushforward(f, backend, x, dx_j)
    end
    return y, grad
end

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient(f::F, backend::AbstractADType, x) where {F}
    return value_and_gradient_aux(f, backend, x, supports_pullback(backend))
end

function value_and_gradient_aux(f::F, backend, x, ::PullbackSupported) where {F}
    return value_and_pullback(f, backend, x, one(myeltype(x)))
end

function value_and_gradient_aux(f::F, backend, x::Number, ::PullbackNotSupported) where {F}
    return value_and_derivative(f, backend, x)
end

function value_and_gradient_aux(
    f::F, backend, x::AbstractArray, ::PullbackNotSupported
) where {F}
    y = f(x)
    grad = map(CartesianIndices(x)) do j
        dx_j = basisarray(backend, x, j)
        last(value_and_pushforward(f, backend, x, dx_j))
    end
    return y, grad
end
