## Docstrings

"""
    prepare_hessian(f, backend, x) -> extras

Create an `extras` object subtyping [`HessianExtras`](@ref) that can be given to Hessian operators.
"""
function prepare_hessian end

"""
    hessian(f, backend, x, [extras]) -> hess
"""
function hessian end

"""
    hessian!(f, hess, backend, x, [extras]) -> hess
"""
function hessian! end

## Preparation

"""
    HessianExtras

Abstract type for additional information needed by Hessian operators.
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
    hess = stack(vec(CartesianIndices(x))) do j
        hess_col_j = hvp(f, backend, x, basis(backend, x, j), extras.hvp_extras)
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
    for (k, j) in enumerate(CartesianIndices(x))
        hess_col_j = reshape(view(hess, :, k), size(x))
        hvp!(f, hess_col_j, backend, x, basis(backend, x, j), extras.hvp_extras)
    end
    return hess
end
