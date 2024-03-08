"""
    value_and_pullback!(dx, backend::AbstractReverseBackend, f, x, dy) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.

!!! info "Interface requirement"
    This is the only required implementation for an [`AbstractReverseBackend`](@ref).
"""
function value_and_pullback!(dx, backend::AbstractReverseBackend, f, x, dy)
    return error(
        "Backend $backend is not loaded or does not support this type combination."
    )
end

"""
    value_and_pullback(backend::AbstractReverseBackend, f, x, dy) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function value_and_pullback(backend::AbstractReverseBackend, f, x, dy)
    dx = mysimilar(x)
    return value_and_pullback!(dx, backend, f, x, dy)
end
