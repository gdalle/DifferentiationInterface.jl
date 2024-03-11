"""
    value_and_multiderivative!(multider, backend, f, x, [extras]) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function value_and_multiderivative!(
    multider::AbstractArray,
    backend::AbstractADType,
    f,
    x::Number,
    extras=nothing,
    implem::AbstractImplem=CustomImplem(),
)
    return value_and_multiderivative!(
        multider, backend, f, x, extras, implem, autodiff_mode(backend)
    )
end

function value_and_multiderivative!(
    multider::AbstractArray,
    backend::AbstractADType,
    f,
    x::Number,
    extras,
    ::AbstractImplem,
    ::ForwardMode,
)
    return value_and_pushforward!(multider, backend, f, x, one(x), extras)
end

function value_and_multiderivative!(
    multider::AbstractArray,
    backend::AbstractADType,
    f,
    x::Number,
    extras,
    ::AbstractImplem,
    ::ReverseMode,
)
    y = f(x)
    for i in eachindex(IndexCartesian(), y)
        dy_i = basisarray(backend, y, i)
        multider[i] = pullback!(multider[i], backend, f, x, dy_i, extras)
    end
    return y, multider
end

"""
    value_and_multiderivative(backend, f, x, [extras]) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function value_and_multiderivative(backend::AbstractADType, f, x::Number, args...)
    multider = similar(f(x))
    return value_and_multiderivative!(multider, backend, f, x, args...)
end

"""
    multiderivative!(multider, backend, f, x, [extras]) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function multiderivative!(
    multider::AbstractArray, backend::AbstractADType, f, x::Number, args...
)
    return last(value_and_multiderivative!(multider, backend, f, x, args...))
end

"""
    multiderivative(backend, f, x, [extras]) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function multiderivative(backend::AbstractADType, f, x::Number, args...)
    return last(value_and_multiderivative(backend, f, x, args...))
end
