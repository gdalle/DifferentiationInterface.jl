"""
    value_and_multiderivative!(multider, backend, f, x) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function value_and_multiderivative!(
    multider::AbstractArray, backend::AbstractADType, f, x::Number
)
    return value_and_multiderivative!(Val{:fallback}(), multider, backend, f, x)
end

function value_and_multiderivative!(
    implem::Val{:fallback}, multider::AbstractArray, backend::AbstractADType, f, x::Number
)
    return value_and_multiderivative!(
        implem, autodiff_mode(backend), multider, backend, f, x
    )
end

function value_and_multiderivative!(
    ::Val{:fallback},
    ::Val{:forward},
    multider::AbstractArray,
    backend::AbstractADType,
    f,
    x::Number,
)
    return value_and_pushforward!(multider, backend, f, x, one(x))
end

function value_and_multiderivative!(
    ::Val{:fallback},
    ::Val{:reverse},
    multider::AbstractArray,
    backend::AbstractADType,
    f,
    x::Number,
)
    y = f(x)
    for i in eachindex(IndexCartesian(), y)
        dy_i = basisarray(backend, y, i)
        multider[i] = pullback!(multider[i], backend, f, x, dy_i)
    end
    return y, multider
end

"""
    value_and_multiderivative(backend, f, x) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function value_and_multiderivative(backend::AbstractADType, f, x::Number)
    return value_and_multiderivative(Val{:fallback}(), backend, f, x)
end

function value_and_multiderivative(
    implem::Val{:fallback}, backend::AbstractADType, f, x::Number
)
    multider = similar(f(x))
    return value_and_multiderivative!(implem, multider, backend, f, x)
end

"""
    multiderivative!(multider, backend, f, x) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider` if possible.
"""
function multiderivative!(multider::AbstractArray, backend::AbstractADType, f, x::Number)
    return multiderivative!(Val{:fallback}(), multider, backend, f, x)
end

function multiderivative!(
    implem::Val{:fallback}, multider::AbstractArray, backend::AbstractADType, f, x::Number
)
    return last(value_and_multiderivative!(implem, multider, backend, f, x))
end

"""
    multiderivative(backend, f, x) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function multiderivative(backend::AbstractADType, f, x::Number)
    return multiderivative(Val{:fallback}(), backend, f, x)
end

function multiderivative(implem::Val{:fallback}, backend::AbstractADType, f, x::Number)
    return last(value_and_multiderivative(implem, backend, f, x))
end
