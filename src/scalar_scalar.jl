"""
    value_and_derivative(backend, f, x) -> (y, der)

Compute the primal value `y = f(x)` and the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function value_and_derivative end

function value_and_derivative(backend::AbstractForwardBackend, f, x::Number)
    return value_and_pushforward!(one(x), backend, f, x, one(x))
end

function value_and_derivative(backend::AbstractReverseBackend, f, x::Number)
    return value_and_pullback!(one(x), backend, f, x, one(x))
end

"""
    derivative(backend, f, x) -> der

Compute the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function derivative(backend::AbstractBackend, f, x::Number)
    return last(value_and_derivative(backend, f, x))
end
