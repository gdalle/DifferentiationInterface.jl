"""
    value_and_multiderivative!(multider, backend, f, x, [extras]) -> (y, multider)
    value_and_multiderivative!(y, multider, backend, f!, x, [extras]) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider`.
"""
function value_and_multiderivative!(
    multider::AbstractArray,
    backend::AbstractADType,
    f::F,
    x::Number,
    extras=prepare_multiderivative(backend, f, x),
) where {F}
    return value_and_multiderivative_aux!(
        multider, backend, f, x, extras, supports_pushforward(backend)
    )
end

function value_and_multiderivative!(
    y::AbstractArray,
    multider::AbstractArray,
    backend::AbstractADType,
    f::F,
    x::Number,
    extras=prepare_multiderivative(backend, f, x, y),
) where {F}
    return value_and_multiderivative_aux!(
        y, multider, backend, f, x, extras, supports_pushforward(backend)
    )
end

function value_and_multiderivative_aux!(
    multider, backend, f::F, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward!(multider, backend, f, x, one(x), extras)
end

function value_and_multiderivative_aux!(
    y, multider, backend, f!::F, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward!(y, multider, backend, f!, x, one(x), extras)
end

function value_and_multiderivative_aux!(
    multider, backend, f::F, x, extras, ::PushforwardNotSupported
) where {F}
    y = f(x)
    for i in eachindex(IndexCartesian(), multider)
        dy_i = basisarray(backend, multider, i)
        multider[i] = pullback(backend, f, x, dy_i, extras)
    end
    return y, multider
end

function value_and_multiderivative_aux!(
    y, multider, backend, f!::F, x, extras, ::PushforwardNotSupported
) where {F}
    for i in eachindex(IndexCartesian(), multider)
        dy_i = basisarray(backend, multider, i)
        y, multider[i] = value_and_pullback!(y, multider[i], backend, f!, x, dy_i, extras)
    end
    return y, multider
end

"""
    value_and_multiderivative(backend, f, x, [extras]) -> (y, multider)

Compute the primal value `y = f(x)` and the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function value_and_multiderivative(
    backend::AbstractADType, f::F, x::Number, extras=prepare_multiderivative(backend, f, x)
) where {F}
    return value_and_multiderivative_aux(
        backend, f, x, extras, supports_pushforward(backend)
    )
end

function value_and_multiderivative_aux(
    backend, f::F, x, extras, ::PushforwardSupported
) where {F}
    return value_and_pushforward(backend, f, x, one(x), extras)
end

function value_and_multiderivative_aux(
    backend, f::F, x, extras, ::PushforwardNotSupported
) where {F}
    multider = similar(f(x))
    return value_and_multiderivative!(multider, backend, f, x, extras)
end

"""
    multiderivative!(multider, backend, f, x, [extras]) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function, overwriting `multider`.
"""
function multiderivative!(
    multider::AbstractArray,
    backend::AbstractADType,
    f::F,
    x::Number,
    extras=prepare_multiderivative(backend, f, x),
) where {F}
    return multiderivative_aux!(
        multider, backend, f, x, extras, supports_pushforward(backend)
    )
end

function multiderivative_aux!(
    multider, backend, f::F, x, extras, ::PushforwardSupported
) where {F}
    return pushforward!(multider, backend, f, x, one(x), extras)
end

function multiderivative_aux!(
    multider, backend, f::F, x, extras, ::PushforwardNotSupported
) where {F}
    return last(value_and_multiderivative!(multider, backend, f, x, extras))
end

"""
    multiderivative(backend, f, x, [extras]) -> multider

Compute the (array-valued) derivative `multider = f'(x)` of a scalar-to-array function.
"""
function multiderivative(
    backend::AbstractADType, f::F, x::Number, extras=prepare_multiderivative(backend, f, x)
) where {F}
    return multiderivative_aux(backend, f, x, extras, supports_pushforward(backend))
end

function multiderivative_aux(backend, f::F, x, extras, ::PushforwardSupported) where {F}
    return pushforward(backend, f, x, one(x), extras)
end

function multiderivative_aux(backend, f::F, x, extras, ::PushforwardNotSupported) where {F}
    return last(value_and_multiderivative(backend, f, x, extras))
end
