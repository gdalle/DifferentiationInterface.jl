const DOC_JACOBIAN_SHAPE = "For a function `f: ℝⁿ → ℝᵐ`, `J` is returned as a `m × n` matrix."

## In-place mutating functions

"""
    value_and_jacobian!(J, backend, f, x[, stuff])

Compute the Jacobian inside the pre-allocated matrix `J`.
$DOC_JACOBIAN_SHAPE
Returns the primal output of the computation `f(x)` and the corresponding Jacobian `J`.

See [`value_and_jacobian`](@ref), [`jacobian!`](@ref) and [`jacobian`](@ref).
"""
function value_and_jacobian!(J::AbstractMatrix, backend::AbstractForwardBackend, f, x)
    y = f(x)
    for (i, dy) in Iterators.enumerate(eachcol(J))
        dx = unitvector(backend, x, i)
        pushforward!(dy, backend, f, x, dx) # mutate J in-place
    end
    return y, J
end

function value_and_jacobian!(J::AbstractMatrix, backend::AbstractReverseBackend, f, x)
    y = f(x)
    for (i, dx) in Iterators.enumerate(eachrow(J))
        dy = unitvector(backend, y, i)
        pullback!(dx, backend, f, x, dy) # mutate J in-place
    end
    return y, J
end

"""
    jacobian!(J, backend, f, x[, stuff])

Compute the Jacobian of `f` at `x` inside the pre-allocated matrix `J` and return `J`.
$DOC_JACOBIAN_SHAPE

See [`value_and_jacobian!`](@ref), [`value_and_jacobian`](@ref) and [`jacobian`](@ref).
"""
function jacobian!(J::AbstractMatrix, backend::AbstractBackend, f, x)
    _, J = value_and_jacobian!(J, backend, f, x)
    return J
end

## Allocating functions

"""
    value_and_jacobian(backend, f, x[, stuff])

Return the primal output of the computation `f(x)` and the corresponding Jacobian `J`.
$DOC_JACOBIAN_SHAPE

See [`value_and_jacobian!`](@ref), [`jacobian!`](@ref) and [`jacobian`](@ref).
"""
function value_and_jacobian(backend::AbstractBackend, f, x)
    y = f(x)
    J = allocate_jacobian_buffer(x, y)
    return y, J = value_and_jacobian!(J, backend, f, x)
end

function allocate_jacobian_buffer(x, y)
    # The type of a derivative is the type julia promotes dy/dx to
    T = typeof(one(eltype(X)) / one(eltype(Y)))
    # For a function f: ℝⁿ → ℝᵐ , a matrix of size (m, n) is returned
    return Matrix{T}(undef, length(y), length(x))
end

"""
    jacobian(backend, f, x[, stuff])

Return the Jacobian `J` of function `f` at `x`.
$DOC_JACOBIAN_SHAPE

See [`value_and_jacobian`](@ref), [`value_and_jacobian!`](@ref) and [`jacobian!`](@ref).
"""
function jacobian(backend::AbstractBackend, f, x)
    _, J = value_and_jacobian(backend, f, x)
    return J
end
