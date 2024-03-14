"""
    value_and_derivative(backend, f, x, [extras]) -> (y, der)

Compute the primal value `y = f(x)` and the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function value_and_derivative end

function value_and_derivative(backend::AbstractADType, f, x::Number, extras, ::ForwardMode)
    return value_and_pushforward(backend, f, x, one(x), extras)
end

function value_and_derivative(backend::AbstractADType, f, x::Number, extras, ::ReverseMode)
    return value_and_pullback(backend, f, x, one(x), extras)
end

"""
    derivative(backend, f, x, [extras]) -> der

Compute the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function derivative end

function derivative(backend::AbstractADType, f, x::Number, extras, ::ForwardMode)
    return pushforward(backend, f, x, one(x), extras)
end

function derivative(backend::AbstractADType, f, x::Number, extras, ::ReverseMode)
    return pullback(backend, f, x, one(x), extras)
end
