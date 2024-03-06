"""
    value_and_derivative(backend, f, x) -> (y, der)

Compute the derivative `der` of a scalar-to-scalar function and return it with the primal output.
"""
function value_and_derivative end

function value_and_derivative(backend::AbstractForwardBackend, f, x::Number)
    return value_and_pushforward!(one(x), backend, f, x, one(x))
end

function value_and_derivative(backend::AbstractReverseBackend, f, x::Number)
    return value_and_pullback!(one(x), backend, f, x, one(x))
end
