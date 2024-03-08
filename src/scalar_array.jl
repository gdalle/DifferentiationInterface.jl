"""
    value_and_multiderivative!(multider, backend, f, x) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function value_and_multiderivative! end

function value_and_multiderivative!(
    multider::AbstractArray, backend::AbstractForwardBackend, f, x::Number
)
    return value_and_pushforward!(multider, backend, f, x, one(x))
end

function value_and_multiderivative!(
    multider::AbstractArray, backend::AbstractReverseBackend, f, x::Number
)
    y = f(x)
    for i in eachindex(IndexCartesian(), y)
        dy_i = basisarray(backend, y, i)
        _, multider[i] = value_and_pullback!(multider[i], backend, f, x, dy_i)
    end
    return y, multider
end

"""
    value_and_multiderivative(backend, f, x) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function value_and_multiderivative(backend::AbstractBackend, f, x::Number)
    multider = similar(f(x))
    return value_and_multiderivative!(multider, backend, f, x)
end
