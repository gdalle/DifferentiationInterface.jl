"""
    value_and_multiderivative!(multider, backend, f, x, [extras]) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function value_and_multiderivative!(
    multider::AbstractArray, backend::AbstractADType, f, x::Number, extras, ::ForwardMode
)
    return value_and_pushforward!(multider, backend, f, x, one(x), extras)
end

function value_and_multiderivative!(
    multider::AbstractArray, backend::AbstractADType, f, x::Number, extras, ::ReverseMode
)
    y = f(x)
    for i in eachindex(IndexCartesian(), multider)
        dy_i = basisarray(backend, multider, i)
        multider[i] = pullback(backend, f, x, dy_i, extras)
    end
    return y, multider
end

"""
    value_and_multiderivative(backend, f, x, [extras]) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function value_and_multiderivative(
    backend::AbstractADType, f, x::Number, extras, ::ForwardMode
)
    return value_and_pushforward(backend, f, x, one(x), extras)
end

function value_and_multiderivative(
    backend::AbstractADType, f, x::Number, extras, ::ReverseMode
)
    multider = similar(f(x))
    return value_and_multiderivative!(multider, backend, f, x, extras)
end

"""
    multiderivative!(multider, backend, f, x, [extras]) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function multiderivative!(
    multider::AbstractArray, backend::AbstractADType, f, x::Number, extras, ::ForwardMode
)
    return pushforward!(multider, backend, f, x, one(x), extras)
end

function multiderivative!(
    multider::AbstractArray, backend::AbstractADType, f, x::Number, extras, ::ReverseMode
)
    return last(value_and_multiderivative!(multider, backend, f, x, extras))
end

"""
    multiderivative(backend, f, x, [extras]) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function multiderivative(backend::AbstractADType, f, x::Number, extras, ::ForwardMode)
    return pushforward(backend, f, x, one(x), extras)
end

function multiderivative(backend::AbstractADType, f, x::Number, extras, ::ReverseMode)
    return last(value_and_multiderivative(backend, f, x, extras))
end
