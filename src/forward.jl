"""
    value_and_pushforward!(dy, backend, f, x, dx[, stuff]) -> (y, dy)

Compute a Jacobian-vector product inside `dy` and return it and the primal output.

# Arguments

- `y`: primal output
- `dy`: cotangent, might be modified
- `backend`: forward-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dx`: tangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function value_and_pushforward! end

"""
    pushforward!(dy, backend, f, x, dx[, stuff])

Compute a Jacobian-vector product inside `dy` and return it.

See [`value_and_pushforward!`](@ref).
"""
function pushforward!(dy, backend, f, x, dx, stuff)
    _, dy = value_and_pushforward!(dy, backend, f, x, dx, stuff)
    return dy
end
