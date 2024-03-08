"""
    value_and_pushforward!(dy, backend, f, x, dx) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a forward mode backend.
"""
function value_and_pushforward!(dy, backend::AbstractADType, f, x, dx)
    return error(
        "Backend $backend is not loaded or does not support this type combination: `typeof(x) = $(typeof(x))` and `typeof(y) = $(typeof(dy))`",
    )
end

"""
    value_and_pushforward(backend, f, x, dx) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function value_and_pushforward(backend::AbstractADType, f, x, dx)
    dy = mysimilar(f(x))
    return value_and_pushforward!(dy, backend, f, x, dx)
end

"""
    pushforward!(dy, backend, f, x, dx) -> dy

Compute the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.
"""
function pushforward!(dy, backend::AbstractADType, f, x, dx)
    return last(value_and_pushforward!(dy, backend, f, x, dx))
end

"""
    pushforward(backend, f, x, dx) -> dy

Compute the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function pushforward(backend::AbstractADType, f, x, dx)
    return last(value_and_pushforward(backend, f, x, dx))
end
