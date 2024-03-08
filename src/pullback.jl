"""
    value_and_pullback!(dx, backend, f, x, dy) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a reverse mode backend.
"""
function value_and_pullback!(dx, backend::AbstractADType, f, x, dy)
    return error(
        "Backend $backend is not loaded or does not support this type combination: `typeof(x) = $(typeof(x))` and `typeof(y) = $(typeof(dy))`",
    )
end

"""
    value_and_pullback(backend, f, x, dy) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function value_and_pullback(backend::AbstractADType, f, x, dy)
    dx = mysimilar(x)
    return value_and_pullback!(dx, backend, f, x, dy)
end

"""
    pullback!(dx, backend, f, x, dy) -> dx

Compute the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.
"""
function pullback!(dx, backend::AbstractADType, f, x, dy)
    return last(value_and_pullback!(dx, backend, f, x, dy))
end

"""
    pullback(backend, f, x, dy) -> dx

Compute the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function pullback(backend::AbstractADType, f, x, dy)
    return last(value_and_pullback(backend, f, x, dy))
end
