"""
    value_and_jacobian!!(f, jac, backend, x, [extras]) -> (y, jac)
    value_and_jacobian!!(f!, y, jac, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian!!(
    f::F,
    jac::AbstractMatrix,
    backend::AbstractADType,
    x::AbstractArray,
    extras=prepare_jacobian(f, backend, x),
) where {F}
    return value_and_jacobian_aux!!(
        f, jac, backend, x, extras, supports_pushforward(backend)
    )
end

function value_and_jacobian!!(
    f!::F,
    y::AbstractArray,
    jac::AbstractMatrix,
    backend::AbstractADType,
    x::AbstractArray,
    extras=prepare_jacobian(f!, backend, y, x),
) where {F}
    return value_and_jacobian_aux!!(
        f!, y, jac, backend, x, extras, supports_pushforward(backend)
    )
end

## Forward mode

function value_and_jacobian_aux!!(
    f::F, jac, backend, x, extras, ::PushforwardSupported
) where {F}
    y = f(x)
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        value_and_pushforward!!(f, jac_col_j, backend, x, dx_j, extras)
    end
    return y, jac
end

function value_and_jacobian_aux!!(
    f!::F, y, jac, backend, x, extras, ::PushforwardSupported
) where {F}
    f!(y, x)
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basisarray(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        value_and_pushforward!!(f!, y, jac_col_j, backend, x, dx_j, extras)
    end
    return y, jac
end

## Reverse mode

function value_and_jacobian_aux!!(
    f::F, jac, backend, x, extras, ::PushforwardNotSupported
) where {F}
    y = f(x)
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        value_and_pullback!!(f, jac_row_i, backend, x, dy_i, extras)
    end
    return y, jac
end

function value_and_jacobian_aux!!(
    f!::F, y, jac, backend, x, extras, ::PushforwardNotSupported
) where {F}
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basisarray(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        value_and_pullback!!(f!, y, jac_row_i, backend, x, dy_i, extras)
    end
    return y, jac
end

"""
    value_and_jacobian(f, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian(
    f::F, backend::AbstractADType, x::AbstractArray, extras=prepare_jacobian(f, backend, x)
) where {F}
    return value_and_jacobian_aux(f, backend, x, extras, supports_pushforward(backend))
end

## Forward mode

function value_and_jacobian_aux(f::F, backend, x, extras, ::PushforwardSupported) where {F}
    y = f(x)
    jac = stack(CartesianIndices(x); dims=2) do j
        dx_j = basisarray(backend, x, j)
        jac_col_j = last(value_and_pushforward(f, backend, x, dx_j, extras))
        vec(jac_col_j)
    end
    return y, jac
end

## Reverse mode

function value_and_jacobian_aux(
    f::F, backend, x, extras, ::PushforwardNotSupported
) where {F}
    y = f(x)
    jac = stack(CartesianIndices(y); dims=1) do i
        dy_i = basisarray(backend, y, i)
        jac_row_i = last(value_and_pullback(f, backend, x, dy_i, extras))
        vec(jac_row_i)
    end
    return y, jac
end

"""
    jacobian!!(f, jac, backend, x, [extras]) -> jac
"""
function jacobian!!(
    f::F,
    jac::AbstractMatrix,
    backend::AbstractADType,
    x::AbstractArray,
    extras=prepare_jacobian(f, backend, x),
) where {F}
    return last(value_and_jacobian!!(f, jac, backend, x, extras))
end

"""
    jacobian(f, backend, x, [extras]) -> jac
"""
function jacobian(
    f::F, backend::AbstractADType, x::AbstractArray, extras=prepare_jacobian(f, backend, x)
) where {F}
    return last(value_and_jacobian(f, backend, x, extras))
end
