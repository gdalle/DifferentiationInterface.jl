"""
    value_and_pushforward!(dy, backend, f, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(y, dy, backend, f!, x, dx, [extras]) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a forward mode backend.
"""
function value_and_pushforward!(dy, backend::AbstractADType, f, x, dx)
    return value_and_pushforward!(dy, backend, f, x, dx, prepare_pushforward(backend, f, x))
end

function value_and_pushforward!(y, dy, backend::AbstractADType, f, x, dx)
    return value_and_pushforward!(
        y, dy, backend, f, x, dx, prepare_pushforward(backend, f, x, y)
    )
end

function value_and_pushforward!(dy, backend::AbstractADType, f, x, dx, extras)
    throw(
        ArgumentError(
            "The backend `$backend` is not available.
            You may need to load the right package extension, implement `DifferentiationInterface.value_and_pushforward!`, or choose another backend.",
        ),
    )
end

function value_and_pushforward!(y, dy, backend::AbstractADType, f, x, dx, extras)
    throw(
        ArgumentError(
            "The backend `$backend` is not available.
            You may need to load the right package extension, implement `DifferentiationInterface.value_and_pushforward!`, or choose another backend.",
        ),
    )
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
