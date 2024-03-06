"""
    value_and_pullback!(dx, backend, f, x, dy[, stuff]) -> (y, dx)

Compute the vector-Jacobian product inside `dx` and return it with the primal output.

!!! info "Interface requirement"
    This is the only required implementation for a reverse mode backend.

# Arguments

- `y`: primal output
- `dx`: tangent, might be modified
- `backend`: reverse-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function value_and_pullback!(dx, backend::AbstractBackend, f, x, dy)
    return error("No package extension loaded for backend $backend.")
end

"""
    value_and_pullback(backend, f, x, dy[, stuff]) -> (y, dx)

Call [`value_and_pullback!`](@ref) after allocating memory for the vector-Jacobian product.
"""
function value_and_pullback(backend::AbstractBackend, f, x, dy)
    dx = mysimilar(x)
    return value_and_pullback!(dx, backend, f, x, dy)
end
