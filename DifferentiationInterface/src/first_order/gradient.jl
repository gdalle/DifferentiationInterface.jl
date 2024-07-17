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

$(document_preparation("gradient"))
"""
function value_and_gradient end

"""
    value_and_gradient!(f, grad, backend, x, [extras]) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`, overwriting `grad`.

$(document_preparation("gradient"))
"""
function value_and_gradient! end

"""
    gradient(f, backend, x, [extras]) -> grad

Compute the gradient of the function `f` at point `x`.

$(document_preparation("gradient"))
"""
function gradient end

"""
    gradient!(f, grad, backend, x, [extras]) -> grad

Compute the gradient of the function `f` at point `x`, overwriting `grad`.

$(document_preparation("gradient"))
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
    pullback_extras = prepare_pullback(f, backend, x, true)
    return PullbackGradientExtras(pullback_extras)
end

## One argument

### Without extras

function value_and_gradient(f::F, backend::AbstractADType, x) where {F}
    return value_and_gradient(f, backend, x, prepare_gradient(f, backend, x))
end

function value_and_gradient!(f::F, der, backend::AbstractADType, x) where {F}
    return value_and_gradient!(f, der, backend, x, prepare_gradient(f, backend, x))
end

function gradient(f::F, backend::AbstractADType, x) where {F}
    return gradient(f, backend, x, prepare_gradient(f, backend, x))
end

function gradient!(f::F, der, backend::AbstractADType, x) where {F}
    return gradient!(f, der, backend, x, prepare_gradient(f, backend, x))
end

### With extras

function value_and_gradient(
    f::F, backend::AbstractADType, x, extras::PullbackGradientExtras
) where {F}
    return value_and_pullback(f, backend, x, true, extras.pullback_extras)
end

function value_and_gradient!(
    f::F, grad, backend::AbstractADType, x, extras::PullbackGradientExtras
) where {F}
    return value_and_pullback!(f, grad, backend, x, true, extras.pullback_extras)
end

function gradient(
    f::F, backend::AbstractADType, x, extras::PullbackGradientExtras
) where {F}
    return pullback(f, backend, x, true, extras.pullback_extras)
end

function gradient!(
    f::F, grad, backend::AbstractADType, x, extras::PullbackGradientExtras
) where {F}
    return pullback!(f, grad, backend, x, true, extras.pullback_extras)
end
