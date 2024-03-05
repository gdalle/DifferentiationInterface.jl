"""
    value_and_pushforward!(dy, backend, f, x, dx[, stuff]) -> (y, dy)

Compute a Jacobian-vector product inside `dy` and return it with the primal output.

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

Call [`value_and_pushforward!`](@ref) after allocating space for the Jacobian-vector product.
"""
function value_and_pushforward(backend::AbstractBackend, f, x, dx)
    dy = f(x)
    return value_and_pushforward!(dy, backend, f, x, dx)
end

"""
    pushforward!(dy, backend, f, x, dx[, stuff]) -> dy

Call [`value_and_pushforward!`](@ref) without returning the primal output.
"""
function pushforward!(dy, backend, f, x, dx)
    _, dy = value_and_pushforward!(dy, backend, f, x, dx)
    return dy
end

"""
    pushforward(backend, f, x, dx[, stuff]) -> dy

Call [`value_and_pushforward`](@ref) without returning the primal output.
"""
function pushforward(backend::AbstractBackend, f, x, dx)
    _, dy = value_and_pushforward(backend, f, x, dx)
    return dy
end
