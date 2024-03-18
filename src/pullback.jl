"""
    value_and_pullback!(dx, backend, f, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(y, dx, backend, f!, x, dy, [extras]) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a reverse mode backend.
"""
function value_and_pullback!(
    dx, backend::AbstractADType, f, x, dy, extras=prepare_pullback(backend, f, x)
)
    return value_and_pullback_aux!(dx, backend, f, x, dy, extras, mode(backend))
end

function value_and_pullback!(
    y, dx, backend::AbstractADType, f!, x, dy, extras=prepare_pullback(backend, f!, x, y)
)
    return value_and_pullback_aux!(y, dx, backend, f!, x, dy, extras, mode(backend))
end

## Reverse mode (true pullback)

function value_and_pullback_aux!(
    dx, backend::AbstractADType, f, x, dy, extras, ::ReverseMode
)
    return error(
        "You need to load the package for `$backend` or implement `DifferentiationInterface.value_and_pullback!` in reverse mode",
    )
end

function value_and_pullback_aux!(
    y, dx, backend::AbstractADType, f!, x, dy, extras, ::ReverseMode
)
    return error(
        "You need to load the package for `$backend` or implement `DifferentiationInterface.value_and_pullback!` in reverse mode",
    )
end

## Forward mode (fake pullback based on pushforward)

function value_and_pullback_aux!(
    _dx::Number, backend::AbstractADType, f, x::Number, dy::Number, extras, ::ForwardMode
)
    return value_and_pushforward(backend, f, x, dy, extras)
end

function value_and_pullback_aux!(
    dx::Number,
    backend::AbstractADType,
    f,
    x::Number,
    dy::AbstractArray,
    extras,
    ::ForwardMode,
)
    y, multider = value_and_pushforward(backend, f, x, one(dx), extras)
    return y, dot(multider, dy)
end

function value_and_pullback_aux!(
    y::AbstractArray,
    dx::Number,
    backend::AbstractADType,
    f!,
    x::Number,
    dy::AbstractArray,
    extras,
    ::ForwardMode,
)
    multider = similar(y)
    y, multider = value_and_pushforward!(y, multider, backend, f!, x, one(dx), extras)
    return y, dot(multider, dy)
end

function value_and_pullback_aux!(
    dx::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    dy::Number,
    extras,
    ::ForwardMode,
)
    y = f(x)
    for j in eachindex(IndexCartesian(), dx)
        v_j = dy * basisarray(backend, dx, j)
        dx[j] = pushforward!(dx[j], backend, f, x, v_j, extras)
    end
    return y, dx
end

"""
    value_and_pullback(backend, f, x, dy, [extras]) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function value_and_pullback(
    backend::AbstractADType, f, x, dy, extras=prepare_pullback(backend, f, x)
)
    dx = mysimilar(x)
    return value_and_pullback!(dx, backend, f, x, dy, extras)
end

"""
    pullback!(dx, backend, f, x, dy, [extras]) -> dx

Compute the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.
"""
function pullback!(
    dx, backend::AbstractADType, f, x, dy, extras=prepare_pullback(backend, f, x)
)
    return last(value_and_pullback!(dx, backend, f, x, dy, extras))
end

"""
    pullback(backend, f, x, dy, [extras]) -> dx

Compute the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function pullback(backend::AbstractADType, f, x, dy, extras=prepare_pullback(backend, f, x))
    return last(value_and_pullback(backend, f, x, dy, extras))
end
