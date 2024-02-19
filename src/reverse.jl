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
function value_and_pullback! end

"""
    pullback!(dx, backend, f, x, dy[, stuff])

Compute a vector-Jacobian product inside `dx` and return it.

# Arguments

- `dx`: tangent, might be modified
- `backend`: reverse-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function pullback!(dx, backend, f, x, dy, stuff)
    _, dx = value_and_pullback!(dx, backend, f, x, dy, stuff)
    return dx
end
