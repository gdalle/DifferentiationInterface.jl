"""
    value_and_pullback!(dx, backend, f, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(y, dx, backend, f!, x, dy, [extras]) -> (y, dx)

Compute the primal value `y = f(x)` and the vector-Jacobian product `dx = ∂f(x)' * dy`, overwriting `dx` if possible.

!!! info "Interface requirement"
    This is the only required implementation for a reverse mode backend.
"""
function value_and_pullback!(dx, backend::AbstractADType, f, x, dy)
    return value_and_pullback!(dx, backend, f, x, dy, prepare_pullback(backend, f, x))
end

function value_and_pullback!(y, dx, backend::AbstractADType, f, x, dy)
    return value_and_pullback!(y, dx, backend, f, x, dy, prepare_pullback(backend, f, x, y))
end

function value_and_pullback!(dx, backend::AbstractADType, f, x, dy, extras)
    throw(
        ArgumentError(
            "The backend `$backend` is not available.
            You may need to load the right package extension, implement `DifferentiationInterface.value_and_pullback!`, or choose another backend.",
        ),
    )
end

function value_and_pullback!(y, dx, backend::AbstractADType, f, x, dy, extras)
    throw(
        ArgumentError(
            "The backend `$backend` is not available.
            You may need to load the right package extension, implement `DifferentiationInterface.value_and_pullback!`, or choose another backend.",
        ),
    )
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
