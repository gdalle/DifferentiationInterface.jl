## Preparation

"""
    JacobianExtras

Abstract type for additional information needed by Jacobian operators.
"""
abstract type JacobianExtras <: Extras end

struct NoJacobianExtras <: JacobianExtras end

struct PushforwardJacobianExtras{E<:PushforwardExtras} <: JacobianExtras
    pushforward_extras::E
end

struct PullbackJacobianExtras{E<:PullbackExtras} <: JacobianExtras
    pullback_extras::E
end

"""
    prepare_jacobian(f, backend, x) -> extras
    prepare_jacobian(f!, backend, y, x) -> extras

Create an `extras` object subtyping [`JacobianExtras`](@ref) that can be given to Jacobian operators.
"""
function prepare_jacobian(f, backend::AbstractADType, x)
    return prepare_jacobian_aux(f, backend, x, pushforward_performance(backend))
end

function prepare_jacobian(f!, backend::AbstractADType, y, x)
    return prepare_jacobian_aux(f!, backend, y, x, pushforward_performance(backend))
end

function prepare_jacobian_aux(f, backend, x, ::PushforwardFast)
    return PushforwardJacobianExtras(prepare_pushforward(f, backend, x))
end

function prepare_jacobian_aux(f!, backend, y, x, ::PushforwardFast)
    return PushforwardJacobianExtras(prepare_pushforward(f!, backend, y, x))
end

function prepare_jacobian_aux(f, backend, x, ::PushforwardSlow)
    return PullbackJacobianExtras(prepare_pullback(f, backend, x))
end

function prepare_jacobian_aux(f!, backend, y, x, ::PushforwardSlow)
    return PullbackJacobianExtras(prepare_pullback(f!, backend, y, x))
end

## Allocating

"""
    value_and_jacobian(f, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian(
    f, backend::AbstractADType, x, extras::JacobianExtras=prepare_jacobian(f, backend, x)
)
    return value_and_jacobian_aux(f, backend, x, extras)
end

function value_and_jacobian_aux(
    f, backend, x::AbstractArray, extras::PushforwardJacobianExtras
)
    y = f(x)
    jac = stack(CartesianIndices(x); dims=2) do j
        dx_j = basis(backend, x, j)
        jac_col_j = pushforward(f, backend, x, dx_j, extras.pushforward_extras)
        vec(jac_col_j)
    end
    return y, jac
end

function value_and_jacobian_aux(
    f, backend, x::AbstractArray, extras::PullbackJacobianExtras
)
    y, pullbackfunc = value_and_pullback_split(f, backend, x, extras.pullback_extras)
    jac = stack(CartesianIndices(y); dims=1) do i
        dy_i = basis(backend, y, i)
        jac_row_i = pullbackfunc(dy_i)
        vec(jac_row_i)
    end
    return y, jac
end

"""
    value_and_jacobian!!(f, jac, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian!!(
    f,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f, backend, x),
)
    return value_and_jacobian_aux!!(f, jac, backend, x, extras)
end

function value_and_jacobian_aux!!(
    f, jac::AbstractMatrix, backend, x::AbstractArray, extras::PushforwardJacobianExtras
)
    y = f(x)
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basis(backend, x, j)
        jac_col_j_old = reshape(view(jac, :, k), size(y))
        jac_col_j_new = pushforward!!(
            f, jac_col_j_old, backend, x, dx_j, extras.pushforward_extras
        )
        # this allocates
        copyto!(jac_col_j_old, jac_col_j_new)
    end
    return y, jac
end

function value_and_jacobian_aux!!(
    f, jac::AbstractMatrix, backend, x::AbstractArray, extras::PullbackJacobianExtras
)
    y, pullbackfunc!! = value_and_pullback!!_split(f, backend, x, extras.pullback_extras)
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basis(backend, y, i)
        jac_row_i_old = reshape(view(jac, k, :), size(x))
        jac_row_i_new = pullbackfunc!!(jac_row_i_old, dy_i)
        # this allocates
        copyto!(jac_row_i_old, jac_row_i_new)
    end
    return y, jac
end

"""
    jacobian(f, backend, x, [extras]) -> jac
"""
function jacobian(
    f, backend::AbstractADType, x, extras::JacobianExtras=prepare_jacobian(f, backend, x)
)
    return value_and_jacobian(f, backend, x, extras)[2]
end

"""
    jacobian!!(f, jac, backend, x, [extras]) -> jac
"""
function jacobian!!(
    f,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f, backend, x),
)
    return value_and_jacobian!!(f, jac, backend, x, extras)[2]
end

## Mutating

"""
    value_and_jacobian!!(f!, y, jac, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian!!(
    f!,
    y,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f!, backend, y, x),
)
    return value_and_jacobian_aux!!(f!, y, jac, backend, x, extras)
end

function value_and_jacobian_aux!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PushforwardJacobianExtras,
)
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basis(backend, x, j)
        jac_col_j_old = reshape(view(jac, :, k), size(y))
        jac_col_j_new = pushforward!!(
            f!, y, jac_col_j_old, backend, x, dx_j, extras.pushforward_extras
        )
        # this allocates
        copyto!(jac_col_j_old, jac_col_j_new)
    end
    return y, jac
end

function value_and_jacobian_aux!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PullbackJacobianExtras,
)
    y, pullbackfunc!! = value_and_pullback!!_split!!(
        f!, y, backend, x, extras.pullback_extras
    )
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basis(backend, y, i)
        jac_row_i_old = reshape(view(jac, k, :), size(x))
        jac_row_i_new = pullbackfunc!!(y, jac_row_i_old, dy_i)
        # this allocates
        copyto!(jac_row_i_old, jac_row_i_new)
    end
    return y, jac
end

"""
    jacobian!!(f!, y, jac, backend, x, [extras]) -> jac
"""
function jacobian!!(
    f!,
    y,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f!, backend, y, x),
)
    return value_and_jacobian!!(f!, y, jac, backend, x, extras)[2]
end
