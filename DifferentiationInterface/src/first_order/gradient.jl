## Docstrings

"""
    prepare_gradient(f, backend, x) -> extras

Create an `extras` object that can be given to [`gradient`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_gradient end

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`.
"""
function value_and_gradient end

"""
    value_and_gradient!(f, grad, backend, x, [extras]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`, overwriting `grad`.
"""
function value_and_gradient! end

"""
    gradient(f, backend, x, [extras]) -> grad

Compute the gradient of the function `f` at point `x`.
"""
function gradient end

"""
    gradient!(f, grad, backend, x, [extras]) -> grad

Compute the gradient of the function `f` at point `x`, overwriting `grad`.
"""
function gradient! end

## Preparation

"""
    GradientExtras

Abstract type for additional information needed by [`gradient`](@ref) and its variants.
"""
abstract type GradientExtras <: Extras end

struct NoGradientExtras <: GradientExtras end

struct PullbackGradientExtras{E<:PullbackExtras} <: GradientExtras
    pullback_extras::E
end

function prepare_gradient(f::F, backend::AbstractADType, x) where {F}
    y = f(x)
    dy = one(y)
    pullback_extras = prepare_pullback(f, backend, x, dy)
    return PullbackGradientExtras(pullback_extras)
end

## One argument

function value_and_gradient(
    f::F, backend::AbstractADType, x, extras::GradientExtras=prepare_gradient(f, backend, x)
) where {F}
    return value_and_pullback(f, backend, x, one(eltype(x)), extras.pullback_extras)
end

function value_and_gradient!(
    f::F,
    grad,
    backend::AbstractADType,
    x,
    extras::GradientExtras=prepare_gradient(f, backend, x),
) where {F}
    return value_and_pullback!(f, grad, backend, x, one(eltype(x)), extras.pullback_extras)
end

function gradient(
    f::F, backend::AbstractADType, x, extras::GradientExtras=prepare_gradient(f, backend, x)
) where {F}
    return pullback(f, backend, x, one(eltype(x)), extras.pullback_extras)
end

function gradient!(
    f::F,
    grad,
    backend::AbstractADType,
    x,
    extras::GradientExtras=prepare_gradient(f, backend, x),
) where {F}
    return pullback!(f, grad, backend, x, one(eltype(x)), extras.pullback_extras)
end
