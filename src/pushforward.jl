"""
    value_and_pushforward!(dy, backend, f, x, dx[, stuff]) -> (y, dy)

Compute the Jacobian-vector product inside `dy` and return it with the primal output.

!!! info "Interface requirement"
    This is the only required implementation for a forward mode backend.

# Arguments

- `y`: primal output
- `dy`: cotangent, might be modified
- `backend`: forward-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dx`: tangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function value_and_pushforward!(dy, backend::AbstractBackend, f, x, dx)
    return error("No package extension loaded for backend $backend.")
end

"""
    value_and_pushforward(backend, f, x, dx[, stuff]) -> (y, dy)

Call [`value_and_pushforward!`](@ref) after allocating memory for the Jacobian-vector product.
"""
function value_and_pushforward(backend::AbstractBackend, f, x, dx)
    dy = mysimilar(f(x))
    return value_and_pushforward!(dy, backend, f, x, dx)
end
