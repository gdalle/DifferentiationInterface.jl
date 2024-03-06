"""
    value_and_multiderivative!(multider, backend, f, x) -> (y, multider)

Compute the derivative of a scalar-to-array function inside `multider` and return it with the primal output. 
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

Call [`value_and_multiderivative!`](@ref) after allocating memory for the multiderivative.
"""
function value_and_multiderivative(backend::AbstractBackend, f, x::Number)
    multider = similar(f(x))
    return value_and_multiderivative!(multider, backend, f, x)
end
