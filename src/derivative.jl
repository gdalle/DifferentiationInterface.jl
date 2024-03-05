function value_and_derivative(backend::AbstractForwardBackend, f, x::Number)
    return value_and_pushforward(backend, f, x, one(x))
end

function value_and_derivative(backend::AbstractReverseBackend, f, x::Number)
    y = f(x)
    return value_and_pullback(backend, f, x, one(y))
end
