"""
    value_and_pushforward!(dy, backend::AbstractForwardBackend, f, x, dx) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`, overwriting `dy` if possible.

!!! info "Interface requirement"
    This is the only required implementation for an [`AbstractForwardBackend`](@ref).
"""
function value_and_pushforward!(dy, backend::AbstractForwardBackend, f, x, dx)
    return error(
        "Backend $backend is not loaded or does not support this type combination."
    )
end

"""
    value_and_pushforward(backend::AbstractForwardBackend, f, x, dx) -> (y, dy)

Compute the primal value `y = f(x)` and the Jacobian-vector product `dy = ∂f(x) * dx`.
"""
function value_and_pushforward(backend::AbstractForwardBackend, f, x, dx)
    dy = mysimilar(f(x))
    return value_and_pushforward!(dy, backend, f, x, dx)
end
