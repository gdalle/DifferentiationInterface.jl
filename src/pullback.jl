"""
    value_and_pullback!(dx, backend, f, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(y, dx, backend, f!, x, dy, [extras]) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a reverse mode backend.
"""
function value_and_pullback!(dx, backend::AbstractADType, f::F, x, dy) where {F}
    return value_and_pullback!(dx, backend, f, x, dy, prepare_pullback(backend, f, x))
end

function value_and_pullback!(y, dx, backend::AbstractADType, f::F, x, dy) where {F}
    return value_and_pullback!(y, dx, backend, f, x, dy, prepare_pullback(backend, f, x, y))
end

"""
    value_and_pullback(backend, f, x, dy, [extras]) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function value_and_pullback(
    backend::AbstractADType, f::F, x, dy, extras=prepare_pullback(backend, f, x)
) where {F}
    dx = mysimilar(x)
    return value_and_pullback!(dx, backend, f, x, dy, extras)
end

"""
    pullback!(dx, backend, f, x, dy, [extras]) -> dx

Compute the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.
"""
function pullback!(
    dx, backend::AbstractADType, f::F, x, dy, extras=prepare_pullback(backend, f, x)
) where {F}
    return last(value_and_pullback!(dx, backend, f, x, dy, extras))
end

"""
    pullback(backend, f, x, dy, [extras]) -> dx

Compute the vector-Jacobian product `dx = ∂f(x)' * dy`.
"""
function pullback(
    backend::AbstractADType, f::F, x, dy, extras=prepare_pullback(backend, f, x)
) where {F}
    return last(value_and_pullback(backend, f, x, dy, extras))
end
