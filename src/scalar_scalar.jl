"""
    value_and_derivative(backend, f, x, [extras]) -> (y, der)

Compute the primal value `y = f(x)` and the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function value_and_derivative(backend::AbstractADType, f, x::Number, extras=nothing)
    return value_and_derivative(backend, f, x, extras, autodiff_mode(backend))
end

function value_and_derivative(backend::AbstractADType, f, x::Number, extras, ::ForwardMode)
    # don't use derivative extras for a pushforward
    return value_and_pushforward(backend, f, x, one(x))
end

function value_and_derivative(backend::AbstractADType, f, x::Number, extras, ::ReverseMode)
    # don't use derivative extras for a pullback
    return value_and_pullback(backend, f, x, one(x))
end

"""
    derivative(backend, f, x, [extras]) -> der

Compute the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function derivative(backend::AbstractADType, f, x::Number, args...)
    return last(value_and_derivative(backend, f, x, args...))
end
