"""
    value_and_derivative(backend, f, x) -> (y, der)

Compute the primal value `y = f(x)` and the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function value_and_derivative(implem::AbstractImplem, backend::AbstractADType, f, x::Number)
    return value_and_derivative(implem, autodiff_mode(backend), backend, f, x)
end

function value_and_derivative(
    ::AbstractImplem, ::ForwardMode, backend::AbstractADType, f, x::Number
)
    return value_and_pushforward!(one(x), backend, f, x, one(x))
end

function value_and_derivative(
    ::AbstractImplem, ::ReverseMode, backend::AbstractADType, f, x::Number
)
    return value_and_pullback!(one(x), backend, f, x, one(x))
end

"""
    derivative(backend, f, x) -> der

Compute the derivative `der = f'(x)` of a scalar-to-scalar function.
"""
function derivative(implem::AbstractImplem, backend::AbstractADType, f, x::Number)
    return last(value_and_derivative(implem, backend, f, x))
end
