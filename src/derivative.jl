"""
    value_and_derivative(backend, f, x, [extras]) -> (y, der)

Compute the primal value `y = f(x)` and the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function value_and_derivative(
    backend::AbstractADType, f::F, x::Number, extras=prepare_derivative(backend, f, x)
) where {F}
    return value_and_derivative_aux(backend, f, x, extras, supports_pushforward(backend))
end

function value_and_derivative_aux(
    backend, f::F, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward(backend, f, x, one(x), extras)
end

function value_and_derivative_aux(
    backend, f::F, x, extras, ::PushforwardNotSupported
) where {F}
    return value_and_pullback(backend, f, x, one(x), extras)
end

"""
    derivative(backend, f, x, [extras]) -> der

Compute the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function derivative(
    backend::AbstractADType, f::F, x::Number, extras=prepare_derivative(backend, f, x)
) where {F}
    return derivative_aux(backend, f, x, extras, supports_pushforward(backend))
end

function derivative_aux(backend, f::F, x, extras, ::PushforwardSupported) where {F}
    return pushforward(backend, f, x, one(x), extras)
end

function derivative_aux(backend, f::F, x, extras, ::PushforwardNotSupported) where {F}
    return pullback(backend, f, x, one(x), extras)
end
