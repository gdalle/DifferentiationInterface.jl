"""
    value_and_pushforward!(dy, backend, f, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(y, dy, backend, f!, x, dx, [extras]) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a forward mode backend.
"""
function value_and_pushforward!(
    dy, backend::AbstractADType, f, x, dx, extras=prepare_pushforward(backend, f, x)
)
    return value_and_pushforward_aux!(dy, backend, f, x, dx, extras, mode(backend))
end

function value_and_pushforward!(
    y, dy, backend::AbstractADType, f!, x, dx, extras=prepare_pushforward(backend, f!, x, y)
)
    return value_and_pushforward_aux!(y, dy, backend, f!, x, dx, extras, mode(backend))
end

## Forward mode (true pushforward)

function value_and_pushforward_aux!(
    dy, backend::AbstractADType, f, x, dx, extras, ::ForwardMode
)
    return error(
        "You need to load the package for `$backend` or implement `DifferentiationInterface.value_and_pushforward!` in forward mode",
    )
end

function value_and_pushforward_aux!(
    y, dy, backend::AbstractADType, f, x, dx, extras, ::ForwardMode
)
    return error(
        "You need to load the package for `$backend` or implement `DifferentiationInterface.value_and_pushforward!` in forward mode",
    )
end

## Reverse mode (fake pushforward based on pullback)

function value_and_pushforward_aux!(
    _dy::Number, backend::AbstractADType, f, x::Number, dx::Number, extras, ::ReverseMode
)
    return value_and_pullback(backend, f, x, dx, extras)
end

function value_and_pushforward_aux!(
    dy::AbstractArray,
    backend::AbstractADType,
    f,
    x::Number,
    dx::Number,
    extras,
    ::ReverseMode,
)
    y = f(x)
    for i in eachindex(IndexCartesian(), dy)
        v_i = dx * basisarray(backend, dy, i)
        dy[i] = pullback!(dy[i], backend, f, x, v_i, extras)
    end
    return y, dy
end

function value_and_pushforward_aux!(
    y::AbstractArray,
    dy::AbstractArray,
    backend::AbstractADType,
    f!,
    x::Number,
    dx::Number,
    extras,
    ::ReverseMode,
)
    for i in eachindex(IndexCartesian(), dy)
        v_i = dx * basisarray(backend, dy, i)
        y, dy[i] = value_and_pullback!(y, dy[i], backend, f!, x, v_i, extras)
    end
    return y, dy
end

function value_and_pushforward_aux!(
    dy::Number,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    dx::AbstractArray,
    extras,
    ::ReverseMode,
)
    y, grad = value_and_pullback(backend, f, x, one(dy), extras)  # allocates
    return y, dot(grad, dx)
end

"""
    value_and_pushforward(backend, f, x, dx, [extras]) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function value_and_pushforward(
    backend::AbstractADType, f, x, dx, extras=prepare_pushforward(backend, f, x)
)
    dy = mysimilar(f(x))
    return value_and_pushforward!(dy, backend, f, x, dx, extras)
end

"""
    pushforward!(dy, backend, f, x, dx, [extras]) -> dy

Compute the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.
"""
function pushforward!(
    dy, backend::AbstractADType, f, x, dx, extras=prepare_pushforward(backend, f, x)
)
    return last(value_and_pushforward!(dy, backend, f, x, dx, extras))
end

"""
    pushforward(backend, f, x, dx, [extras]) -> dy

Compute the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function pushforward(
    backend::AbstractADType, f, x, dx, extras=prepare_pushforward(backend, f, x)
)
    return last(value_and_pushforward(backend, f, x, dx, extras))
end
