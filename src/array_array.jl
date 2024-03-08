const JAC_NOTES = """
## Notes

Regardless of the shape of `x` and `y`, if `x` has length `n` and `y` has length `m`, then `jac` is expected to be a `m × n` matrix.
This function acts as if the input and output had been flattened with `vec`. 
"""

"""
    value_and_jacobian!(jac, backend, f, x) -> (y, jac)

Compute the primal value `y = f(x)` and the Jacobian matrix `jac = ∂f(x)` of an array-to-array function, overwriting `jac` if possible.

$JAC_NOTES
"""
function value_and_jacobian!(
    jac::AbstractMatrix, backend::AbstractADType, f, x::AbstractArray
)
    return value_and_jacobian!(Val{:fallback}(), jac, backend, f, x)
end

function value_and_jacobian!(
    implem::Val{:fallback},
    jac::AbstractMatrix,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    return value_and_jacobian!(implem, autodiff_mode(backend), jac, backend, f, x)
end

function check_jac(jac::AbstractMatrix, x::AbstractArray, y::AbstractArray)
    nx, ny = length(x), length(y)
    size(jac) != (ny, nx) && throw(
        DimensionMismatch("Size of Jacobian buffer doesn't match expected size ($ny, $nx)"),
    )
    return nothing
end

function value_and_jacobian!(
    ::Val{:fallback},
    ::Val{:forward},
    jac::AbstractMatrix,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    y = f(x)
    check_jac(jac, x, y)
    for (k, j) in enumerate(eachindex(IndexCartesian(), x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        pushforward!(jac_col_j, backend, f, x, dx_j)
    end
    return y, jac
end

function value_and_jacobian!(
    ::Val{:fallback},
    ::Val{:reverse},
    jac::AbstractMatrix,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    y = f(x)
    check_jac(jac, x, y)
    for (k, i) in enumerate(eachindex(IndexCartesian(), y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        pullback!(jac_row_i, backend, f, x, dy_i)
    end
    return y, jac
end

"""
    value_and_jacobian(backend, f, x) -> (y, jac)

Compute the primal value `y = f(x)` and the Jacobian matrix `jac = ∂f(x)` of an array-to-array function.

$JAC_NOTES 
"""
function value_and_jacobian(backend::AbstractADType, f, x::AbstractArray)
    return value_and_jacobian(Val{:fallback}(), backend, f, x)
end

function value_and_jacobian(
    implem::Val{:fallback}, backend::AbstractADType, f, x::AbstractArray
)
    y = f(x)
    T = promote_type(eltype(x), eltype(y))
    jac = similar(y, T, length(y), length(x))
    return value_and_jacobian!(implem, jac, backend, f, x)
end

"""
    jacobian!(jac, backend, f, x) -> jac

Compute the Jacobian matrix `jac = ∂f(x)` of an array-to-array function, overwriting `jac` if possible.

$JAC_NOTES
"""
function jacobian!(jac::AbstractMatrix, backend::AbstractADType, f, x::AbstractArray)
    return jacobian!(Val{:fallback}(), jac, backend, f, x)
end

function jacobian!(
    implem::Val{:fallback},
    jac::AbstractMatrix,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    return last(value_and_jacobian!(implem, jac, backend, f, x))
end

"""
    jacobian(backend, f, x) -> jac

Compute the Jacobian matrix `jac = ∂f(x)` of an array-to-array function.

$JAC_NOTES
"""
function jacobian(backend::AbstractADType, f, x::AbstractArray)
    return jacobian(Val{:fallback}(), backend, f, x)
end

function jacobian(implem::Val{:fallback}, backend::AbstractADType, f, x::AbstractArray)
    return last(value_and_jacobian(implem, backend, f, x))
end
