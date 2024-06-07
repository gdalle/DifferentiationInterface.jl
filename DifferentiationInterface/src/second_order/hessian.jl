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

"""
    value_hessian_and_hessian(f, backend, x, [extras]) -> (y, grad, hess)

Compute the value, hessian vector and Hessian matrix of the function `f` at point `x`.
"""
function value_hessian_and_hessian end

"""
    value_hessian_and_hessian!(f, grad, hess, backend, x, [extras]) -> (y, grad, hess)

Compute the value, hessian vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.
"""
function value_hessian_and_hessian! end

## Preparation

"""
    HessianExtras

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianExtras <: Extras end

struct NoHessianExtras <: HessianExtras end

struct HVPGradientHessianExtras{E2<:HVPExtras,E1<:GradientExtras} <: HessianExtras
    hvp_extras::E2
    gradient_extras::E1
end

function prepare_hessian(f::F, backend::AbstractADType, x) where {F}
    v = basis(backend, x, first(CartesianIndices(x)))
    hvp_extras = prepare_hvp(f, backend, x, v)
    gradient_extras = prepare_gradient(f, maybe_inner(backend), x)
    return HVPGradientHessianExtras(hvp_extras, gradient_extras)
end

## One argument

function value_gradient_and_hessian(f::F, backend::AbstractADType, x) where {F}
    return value_gradient_and_hessian(f, backend, x, prepare_hessian(f, backend, x))
end

function value_gradient_and_hessian!(f::F, grad, hess, backend::AbstractADType, x) where {F}
    return value_gradient_and_hessian!(
        f, grad, hess, backend, x, prepare_hessian(f, backend, x)
    )
end

function hessian(f::F, backend::AbstractADType, x) where {F}
    return hessian(f, backend, x, prepare_hessian(f, backend, x))
end

function hessian!(f::F, hess, backend::AbstractADType, x) where {F}
    return hessian!(f, hess, backend, x, prepare_hessian(f, backend, x))
end

function hessian(
    f::F, backend::AbstractADType, x, extras::HVPGradientHessianExtras
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
    f::F, hess, backend::AbstractADType, x, extras::HVPGradientHessianExtras
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

function value_gradient_and_hessian(
    f::F, backend::AbstractADType, x, extras::HVPGradientHessianExtras
) where {F}
    y, grad = value_and_gradient(f, maybe_inner(backend), x, extras.gradient_extras)
    hess = hessian(f, backend, x, extras)
    return y, grad, hess
end

function value_gradient_and_hessian!(
    f::F, grad, hess, backend::AbstractADType, x, extras::HVPGradientHessianExtras
) where {F}
    y, _ = value_and_gradient!(f, grad, maybe_inner(backend), x, extras.gradient_extras)
    hessian!(f, hess, backend, x, extras)
    return y, grad, hess
end
