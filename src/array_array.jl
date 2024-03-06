"""
    value_and_jacobian!(jac, backend, f, x[, stuff]) -> (y, jac)

Compute the Jacobian inside `jac` and return it with the primal output.

## Notes

For a function `f: ℝⁿ → ℝᵐ`, `jac` is an `m × n` matrix.
If the input or output is a higher-order array, it is flattened in the usual way (with `vec`).
"""
function value_and_jacobian!(
    jac::AbstractMatrix, backend::AbstractBackend, f, x::AbstractArray
)
    y = f(x)
    nx, ny = length(x), length(y)
    size(jac) != (ny, nx) && throw(
        DimensionMismatch("Size of Jacobian buffer doesn't match expected size ($ny, $nx)"),
    )
    return y, jac = _value_and_jacobian!(jac, backend, f, x, y)
end

function _value_and_jacobian!(
    jac::AbstractMatrix,
    backend::AbstractForwardBackend,
    f,
    x::AbstractArray,
    y::AbstractArray,
)
    for j in eachindex(IndexCartesian(), x)
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshapeview(jac, (:, j), y) # view onto i-th column of J, reshaped to match y
        pushforward!(jac_col_j, backend, f, x, dx_j)
    end
    return y, jac
end

function _value_and_jacobian!(
    jac::AbstractMatrix,
    backend::AbstractReverseBackend,
    f,
    x::AbstractArray,
    y::AbstractArray,
)
    y = f(x)
    for i in eachindex(IndexCartesian(), y)
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshapeview(jac, (i, :), x) # view onto i-th row of J, reshaped to match x
        pullback!(jac_row_i, backend, f, x, dy_i)
    end
    return y, J
end

reshapeview(A, inds, B) = reshape(view(A, inds...), size(B)...)
