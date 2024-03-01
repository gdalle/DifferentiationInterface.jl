"""
    value_and_pullback!(dx, backend, f, x, dy[, stuff]) -> (y, dx)

Compute a vector-Jacobian product inside `dx` and return it and the primal output.

# Arguments

- `y`: primal output
- `dx`: tangent, might be modified
- `backend`: reverse-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function value_and_pullback!(dx, backend::AbstractBackend, f, x::X, dy::Y) where {X,Y}
    return error("No package extension loaded for backend $backend.")
end

"""
    pullback!(dx, backend, f, x, dy[, stuff])

Compute a vector-Jacobian product inside `dx` and return it.

See [`value_and_pullback!`](@ref).
"""
function pullback!(dx, backend, f, x, dy)
    _, dx = value_and_pullback!(dx, backend, f, x, dy)
    return dx
end
