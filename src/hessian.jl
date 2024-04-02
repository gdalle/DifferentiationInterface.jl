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

"""
    prepare_hessian([other_extras], f, backend, x) -> extras

Create an `extras` object subtyping [`HessianExtras`](@ref) that can be given to Hessian operators.
"""
function prepare_hessian(::Extras, f_or_f!, backend::AbstractADType, args...)
    return prepare_hessian(f_or_f!, backend, args...)
end

function prepare_hessian(f, backend::AbstractADType, x)
    return HVPHessianExtras(prepare_hvp(f, backend, x))
end

function prepare_hvp(extras::HVPHessianExtras, f_or_f!, backend::AbstractADType, args...)
    return extras.hvp_extras
end

## Allocating

"""
    hessian(f, backend, x, [extras]) -> hess
"""
function hessian(
    f, backend::AbstractADType, x, extras::HessianExtras=prepare_hessian(f, backend, x)
)
    new_backend = SecondOrder(backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian(f, new_backend, x, new_extras)
end

function hessian(
    f, backend::SecondOrder, x, extras::HessianExtras=prepare_hessian(f, backend, x)
)
    new_extras = prepare_hvp(extras, f, backend, x)
    hess = stack(vec(CartesianIndices(x))) do j
        hess_col_j = hvp(f, backend, x, basis(backend, x, j), new_extras)
        vec(hess_col_j)
    end
    return hess
end

"""
    hessian!!(f, hess, backend, x, [extras]) -> hess
"""
function hessian!!(
    f,
    hess,
    backend::AbstractADType,
    x,
    extras::HessianExtras=prepare_hessian(f, backend, x),
)
    new_backend = SecondOrder(backend)
    new_extras = prepare_hessian(f, new_backend, x)
    return hessian!!(f, hess, new_backend, x, new_extras)
end

function hessian!!(
    f, hess, backend::SecondOrder, x, extras::HessianExtras=prepare_hessian(f, backend, x)
)
    new_extras = prepare_hvp(extras, f, backend, x)
    for (k, j) in enumerate(CartesianIndices(x))
        hess_col_j_old = reshape(view(hess, :, k), size(x))
        hess_col_j_new = hvp!!(
            f, hess_col_j_old, backend, x, basis(backend, x, j), new_extras
        )
        # this allocates
        copyto!(hess_col_j_old, hess_col_j_new)
    end
    return hess
end
