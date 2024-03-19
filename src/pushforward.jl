"""
    value_and_pushforward!(dy, backend, f, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(y, dy, backend, f!, x, dx, [extras]) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a forward mode backend.
"""
function value_and_pushforward!(dy, backend::AbstractADType, f::F, x, dx) where {F}
    return value_and_pushforward!(dy, backend, f, x, dx, prepare_pushforward(backend, f, x))
end

function value_and_pushforward!(y, dy, backend::AbstractADType, f::F, x, dx) where {F}
    return value_and_pushforward!(
        y, dy, backend, f, x, dx, prepare_pushforward(backend, f, x, y)
    )
end

"""
    value_and_pushforward(backend, f, x, dx, [extras]) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function value_and_pushforward(
    backend::AbstractADType, f::F, x, dx, extras=prepare_pushforward(backend, f, x)
) where {F}
    dy = mysimilar(f(x))
    return value_and_pushforward!(dy, backend, f, x, dx, extras)
end

"""
    pushforward!(dy, backend, f, x, dx, [extras]) -> dy

Compute the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.
"""
function pushforward!(
    dy, backend::AbstractADType, f::F, x, dx, extras=prepare_pushforward(backend, f, x)
) where {F}
    return last(value_and_pushforward!(dy, backend, f, x, dx, extras))
end

"""
    pushforward(backend, f, x, dx, [extras]) -> dy

Compute the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function pushforward(
    backend::AbstractADType, f::F, x, dx, extras=prepare_pushforward(backend, f, x)
) where {F}
    return last(value_and_pushforward(backend, f, x, dx, extras))
end
