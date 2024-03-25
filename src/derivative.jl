## Allocating

"""
    value_and_derivative(f, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative(
    f::F, backend::AbstractADType, x::Number, extras=prepare_derivative(f, backend, x)
) where {F}
    return value_and_derivative_aux(f, backend, x, extras, supports_pushforward(backend))
end

function value_and_derivative_aux(
    f::F, backend, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward(f, backend, x, one(x), extras)
end

function value_and_derivative_aux(
    f::F, backend, x, extras, ::PushforwardNotSupported
) where {F}
    y = f(x)
    if y isa Number
        return value_and_pullback(f, backend, x, one(y))
    else
        der = map(CartesianIndices(y)) do i
            dy_i = basisarray(backend, y, i)
            last(value_and_pullback(f, backend, x, dy_i, extras))
        end
        return y, der
    end
end

"""
    value_and_derivative!!(f, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!!(
    f::F, der, backend::AbstractADType, x::Number, extras=prepare_derivative(f, backend, x)
) where {F}
    return value_and_derivative_aux!!(
        f, der, backend, x, extras, supports_pushforward(backend)
    )
end

function value_and_derivative_aux!!(
    f::F, der, backend, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward!!(f, der, backend, x, one(x), extras)
end

function value_and_derivative_aux!!(
    f::F, _der::Number, backend, x, extras, ::PushforwardNotSupported
) where {F}
    return value_and_pullback(f, backend, x, one(x), extras)
end

function value_and_derivative_aux!!(
    f::F, der::AbstractArray, backend, x, extras, ::PushforwardNotSupported
) where {F}
    y = f(x)
    map!(der, CartesianIndices(y)) do i
        dy_i = basisarray(backend, y, i)
        pullback(f, backend, x, dy_i, extras)
    end
    return y, der
end

"""
    derivative(f, backend, x, [extras]) -> der
"""
function derivative(
    f::F, backend::AbstractADType, x::Number, extras=prepare_derivative(f, backend, x)
) where {F}
    return last(value_and_derivative(f, backend, x, extras))
end

"""
    derivative!!(f, der, backend, x, [extras]) -> der
"""
function derivative!!(
    f::F, der, backend::AbstractADType, x::Number, extras=prepare_derivative(f, backend, x)
) where {F}
    return last(value_and_derivative!!(f, der, backend, x, extras))
end

## Mutating

"""
    value_and_derivative!!(f!, y, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!!(
    f!::F,
    y,
    der,
    backend::AbstractADType,
    x::Number,
    extras=prepare_derivative(f!, backend, y, x),
) where {F}
    return value_and_derivative_aux!!(
        f!, y, der, backend, x, extras, supports_pushforward(backend)
    )
end

function value_and_derivative_aux!!(
    f!::F, y, der, backend, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward!!(f!, y, der, backend, x, one(x), extras)
end

function value_and_derivative_aux!!(
    f!::F,
    y::AbstractArray,
    der::AbstractArray,
    backend,
    x,
    extras,
    ::PushforwardNotSupported,
) where {F}
    f!(y, x)
    map!(der, CartesianIndices(y)) do i
        dy_i = basisarray(backend, y, i)
        last(value_and_pullback!!(f!, y, der[i], backend, x, dy_i, extras))
    end
    return y, der
end
