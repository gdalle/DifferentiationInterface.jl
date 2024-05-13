## Docstrings

"""
    prepare_jacobian(f,     backend, x) -> extras
    prepare_jacobian(f!, y, backend, x) -> extras

Create an `extras` object subtyping [`JacobianExtras`](@ref) that can be given to Jacobian operators.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_jacobian end

"""
    value_and_jacobian(f,     backend, x, [extras]) -> (y, jac)
    value_and_jacobian(f!, y, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian end

"""
    value_and_jacobian!(f,     jac, backend, x, [extras]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, backend, x, [extras]) -> (y, jac)
"""
function value_and_jacobian! end

"""
    jacobian(f,     backend, x, [extras]) -> jac
    jacobian(f!, y, backend, x, [extras]) -> jac
"""
function jacobian end

"""
    jacobian!(f,     jac, backend, x, [extras]) -> jac
    jacobian!(f!, y, jac, backend, x, [extras]) -> jac
"""
function jacobian! end

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

function prepare_jacobian(f::F, backend::AbstractADType, x) where {F}
    return prepare_jacobian_aux(f, backend, x, pushforward_performance(backend))
end

function prepare_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return prepare_jacobian_aux(f!, y, backend, x, pushforward_performance(backend))
end

function prepare_jacobian_aux(f::F, backend, x, ::PushforwardFast) where {F}
    dx = basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f, backend, x, dx)
    return PushforwardJacobianExtras(pushforward_extras)
end

function prepare_jacobian_aux(f!::F, y, backend, x, ::PushforwardFast) where {F}
    dx = basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f!, y, backend, x, dx)
    return PushforwardJacobianExtras(pushforward_extras)
end

function prepare_jacobian_aux(f::F, backend, x, ::PushforwardSlow) where {F}
    y = f(x)
    dy = basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f, backend, x, dy)
    return PullbackJacobianExtras(pullback_extras)
end

function prepare_jacobian_aux(f!::F, y, backend, x, ::PushforwardSlow) where {F}
    dy = basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f!, y, backend, x, dy)
    return PullbackJacobianExtras(pullback_extras)
end

## One argument

function value_and_jacobian(
    f::F, backend::AbstractADType, x, extras::JacobianExtras=prepare_jacobian(f, backend, x)
) where {F}
    return value_and_jacobian_onearg_aux(f, backend, x, extras)
end

function value_and_jacobian_onearg_aux(
    f::F, backend, x::AbstractArray, extras::PushforwardJacobianExtras
) where {F}
    pushforward_extras_same = prepare_pushforward_same_point(
        f,
        backend,
        x,
        basis(backend, x, first(CartesianIndices(x))),
        extras.pushforward_extras,
    )
    y = f(x)  # TODO: remove
    jac = stack(CartesianIndices(x); dims=2) do j
        dx_j = basis(backend, x, j)
        jac_col_j = pushforward(f, backend, x, dx_j, pushforward_extras_same)
        vec(jac_col_j)
    end
    return y, jac
end

function value_and_jacobian_onearg_aux(
    f::F, backend, x::AbstractArray, extras::PullbackJacobianExtras
) where {F}
    pullback_extras_same = prepare_pullback_same_point(
        f, backend, x, basis(backend, y, first(CartesianIndices(y))), extras.pullback_extras
    )
    y = f(x)  # TODO: remove
    jac = stack(CartesianIndices(y); dims=1) do i
        dy_i = basis(backend, y, i)
        jac_row_i = pullback(f, backend, x, dy_i, pullback_extras_same)
        vec(jac_row_i)
    end
    return y, jac
end

function value_and_jacobian!(
    f::F,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f, backend, x),
) where {F}
    return value_and_jacobian_onearg_aux!(f, jac, backend, x, extras)
end

function value_and_jacobian_onearg_aux!(
    f::F, jac::AbstractMatrix, backend, x::AbstractArray, extras::PushforwardJacobianExtras
) where {F}
    pushforward_extras_same = prepare_pushforward_same_point(
        f,
        backend,
        x,
        basis(backend, x, first(CartesianIndices(x))),
        extras.pushforward_extras,
    )
    y = f(x)  # TODO: remove
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basis(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        pushforward!(f, jac_col_j, backend, x, dx_j, pushforward_extras_same)
    end
    return y, jac
end

function value_and_jacobian_onearg_aux!(
    f::F, jac::AbstractMatrix, backend, x::AbstractArray, extras::PullbackJacobianExtras
) where {F}
    pullback_extras_same = prepare_pullback_same_point(
        f, backend, x, basis(backend, y, first(CartesianIndices(y))), extras.pullback_extras
    )
    y = f(x)  # TODO: remove
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basis(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        pullback!(f, jac_row_i, backend, x, dy_i, pullback_extras_same)
    end
    return y, jac
end

function jacobian(
    f::F, backend::AbstractADType, x, extras::JacobianExtras=prepare_jacobian(f, backend, x)
) where {F}
    return value_and_jacobian(f, backend, x, extras)[2]
end

function jacobian!(
    f::F,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f, backend, x),
) where {F}
    return value_and_jacobian!(f, jac, backend, x, extras)[2]
end

## Two arguments

function value_and_jacobian(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f!, y, backend, x),
) where {F}
    return value_and_jacobian_twoarg_aux(f!, y, backend, x, extras)
end

function value_and_jacobian_twoarg_aux(
    f!::F, y, backend, x::AbstractArray, extras::PushforwardJacobianExtras
) where {F}
    pushforward_extras_same = prepare_pushforward_same_point(
        f!,
        y,
        backend,
        x,
        basis(backend, x, first(CartesianIndices(x))),
        extras.pushforward_extras,
    )
    jac = stack(CartesianIndices(x); dims=2) do j
        dx_j = basis(backend, x, j)
        jac_col_j = pushforward(f!, y, backend, x, dx_j, pushforward_extras_same)
        vec(jac_col_j)
    end
    f!(y, x)  # TODO: remove
    return y, jac
end

function value_and_jacobian_twoarg_aux(
    f!::F, y, backend, x::AbstractArray, extras::PullbackJacobianExtras
) where {F}
    pullback_extras_same = prepare_pullback_same_point(
        f!,
        y,
        backend,
        x,
        basis(backend, y, first(CartesianIndices(y))),
        extras.pullback_extras,
    )
    jac = stack(CartesianIndices(y); dims=1) do i
        dy_i = basis(backend, y, i)
        jac_row_i = pullback(f!, y, backend, x, dy_i, pullback_extras_same)
        vec(jac_row_i)
    end
    f!(y, x)  # TODO: remove
    return y, jac
end

function value_and_jacobian!(
    f!::F,
    y,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f!, y, backend, x),
) where {F}
    return value_and_jacobian_twoarg_aux!(f!, y, jac, backend, x, extras)
end

function value_and_jacobian_twoarg_aux!(
    f!::F,
    y,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PushforwardJacobianExtras,
) where {F}
    pushforward_extras_same = prepare_pushforward_same_point(
        f!,
        y,
        backend,
        x,
        basis(backend, x, first(CartesianIndices(x))),
        extras.pushforward_extras,
    )
    for (k, j) in enumerate(CartesianIndices(x))
        dx_j = basis(backend, x, j)
        jac_col_j = reshape(view(jac, :, k), size(y))
        pushforward!(f!, y, jac_col_j, backend, x, dx_j, pushforward_extras_same)
    end
    f!(y, x)  # TODO: remove
    return y, jac
end

function value_and_jacobian_twoarg_aux!(
    f!::F, y, jac::AbstractMatrix, backend, x::AbstractArray, extras::PullbackJacobianExtras
) where {F}
    pullback_extras_same = prepare_pullback_same_point(
        f!,
        y,
        backend,
        x,
        basis(backend, y, first(CartesianIndices(y))),
        extras.pullback_extras,
    )
    for (k, i) in enumerate(CartesianIndices(y))
        dy_i = basis(backend, y, i)
        jac_row_i = reshape(view(jac, k, :), size(x))
        pullback!(f!, y, jac_row_i, backend, x, dy_i, pullback_extras_same)
    end
    f!(y, x)  # TODO: remove
    return y, jac
end

function jacobian(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f!, y, backend, x),
) where {F}
    return value_and_jacobian(f!, y, backend, x, extras)[2]
end

function jacobian!(
    f!::F,
    y,
    jac,
    backend::AbstractADType,
    x,
    extras::JacobianExtras=prepare_jacobian(f!, y, backend, x),
) where {F}
    return value_and_jacobian!(f!, y, jac, backend, x, extras)[2]
end
