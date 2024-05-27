## Docstrings

"""
    prepare_hessian(f, backend, x) -> extras

Create an `extras` object that can be given to [`hessian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hessian end

"""
    hessian(f, backend, x, [extras]) -> hess

Compute the Hessian matrix of the function `f` at point `x`.
"""
function hessian end

"""
    hessian!(f, hess, backend, x, [extras]) -> hess

Compute the Hessian matrix of the function `f` at point `x`, overwriting `hess`.
"""
function hessian! end

## Preparation

"""
    HessianExtras

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianExtras <: Extras end

struct NoHessianExtras <: HessianExtras end

struct HVPHessianExtras{E<:HVPExtras} <: HessianExtras
    hvp_extras::E
end

function prepare_hessian(f::F, backend::AbstractADType, x) where {F}
    v = basis(backend, x, first(CartesianIndices(x)))
    hvp_extras = prepare_hvp(f, backend, x, v)
    return HVPHessianExtras(hvp_extras)
end

## One argument

function hessian(
    f::F, backend::AbstractADType, x, extras::HessianExtras=prepare_hessian(f, backend, x)
) where {F}
    new_backend = SecondOrder(backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian(f, new_backend, x, new_extras)
end

function hessian(
    f::F, backend::SecondOrder, x, extras::HessianExtras=prepare_hessian(f, backend, x)
) where {F}
    hvp_extras_same = prepare_hvp_same_point(
        f, backend, x, basis(backend, x, first(CartesianIndices(x))), extras.hvp_extras
    )
    hess = stack(vec(CartesianIndices(x))) do j
        hess_col_j = hvp(f, backend, x, basis(backend, x, j), hvp_extras_same)
        vec(hess_col_j)
    end
    return hess
end

function hessian!(
    f::F,
    hess,
    backend::AbstractADType,
    x,
    extras::HessianExtras=prepare_hessian(f, backend, x),
) where {F}
    new_backend = SecondOrder(backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian!(f, hess, new_backend, x, new_extras)
end

function hessian!(
    f::F,
    hess,
    backend::SecondOrder,
    x,
    extras::HessianExtras=prepare_hessian(f, backend, x),
) where {F}
    hvp_extras_same = prepare_hvp_same_point(
        f, backend, x, basis(backend, x, first(CartesianIndices(x))), extras.hvp_extras
    )
    for (k, j) in enumerate(CartesianIndices(x))
        hess_col_j = reshape(view(hess, :, k), size(x))
        hvp!(f, hess_col_j, backend, x, basis(backend, x, j), hvp_extras_same)
    end
    return hess
end
