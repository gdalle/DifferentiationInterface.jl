const DOC_JACOBIAN_SHAPE = "For a function `f: ℝⁿ → ℝᵐ`, `J` is returned as a `m × n` matrix."

## In-place mutating functions

"""
    value_and_jacobian!(J, backend, f, x[, stuff])

Compute the Jacobian inside the pre-allocated matrix `J`.
$DOC_JACOBIAN_SHAPE
Returns the primal output of the computation `f(x)` and the corresponding Jacobian `J`.

See [`value_and_jacobian`](@ref), [`jacobian!`](@ref) and [`jacobian`](@ref).
"""
function value_and_jacobian!(J::AbstractMatrix, backend, f, x)
    y = f(x)
    nx, ny = length(x), length(y)
    size(J) != (ny, nx) && throw(
        DimensionMismatch("Size of Jacobian buffer doesn't match expected size ($ny, $nx)"),
    )
    return y, J = _value_and_jacobian!(J, backend, f, x, y)
end

function _value_and_jacobian!(J, backend::AbstractReverseBackend, f, x, y)
    for i in axes(J, 1)
        dy = unitvector(backend, y, i)
        Jrow = reshapeview(J, (i, :), x) # view onto i-th row of J, reshaped to match x
        pullback!(Jrow, backend, f, x, dy)
    end
    return y, J
end

function _value_and_jacobian!(J, backend::AbstractForwardBackend, f, x, y)
    for i in axes(J, 2)
        dx = unitvector(backend, x, i)
        Jcol = reshapeview(J, (:, i), y) # view onto i-th column of J, reshaped to match y
        pushforward!(Jcol, backend, f, x, dx)
    end
    return y, J
end

# Special case for scalar x since pullback! assumes it can't mutate dx.
function _value_and_jacobian!(J, backend::AbstractReverseBackend, f, x::Real, y)
    dx = one(x) # place-holder for dispatch as it won't be mutated
    for i in axes(J, 1)
        dy = unitvector(backend, y, i)
        J[i] = pullback!(dx, backend, f, x, dy) # J is of shape (length(x), 1)
    end
    return y, J
end

# Special case for scalar y since pushforward! assumes it can't mutate dy.
function _value_and_jacobian!(J, backend::AbstractForwardBackend, f, x, y::Real)
    dy = one(y) # place-holder for dispatch as it won't be mutated
    for i in axes(J, 2)
        dx = unitvector(backend, x, i)
        J[i] = pushforward!(dy, backend, f, x, dx) # J is of shape (1, length(x))
    end
    return y, J
end

reshapeview(A, inds, B) = reshape(view(A, inds...), size(B)...)

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
    T = typeof(one(eltype(y)) / one(eltype(x)))
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
