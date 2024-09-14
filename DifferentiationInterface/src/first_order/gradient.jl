## Docstrings

"""
    prepare_gradient(f, backend, x) -> extras

Create an `extras` object that can be given to [`gradient`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_gradient end

"""
    value_and_gradient(f, [extras,] backend, x) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`.

$(document_preparation("gradient"))
"""
function value_and_gradient end

"""
    value_and_gradient!(f, grad, [extras,] backend, x) -> (y, grad)

Compute the value and the gradient of the function `f` at point `x`, overwriting `grad`.

$(document_preparation("gradient"))
"""
function value_and_gradient! end

"""
    gradient(f, [extras,] backend, x) -> grad

Compute the gradient of the function `f` at point `x`.

$(document_preparation("gradient"))
"""
function gradient end

"""
    gradient!(f, grad, [extras,] backend, x) -> grad

Compute the gradient of the function `f` at point `x`, overwriting `grad`.

$(document_preparation("gradient"))
"""
function gradient! end

## Preparation

struct PullbackGradientExtras{E<:PullbackExtras} <: GradientExtras
    pullback_extras::E
end

function prepare_gradient(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    pullback_extras = prepare_pullback(f, backend, x, Tangents(true), contexts...)
    return PullbackGradientExtras(pullback_extras)
end

## One argument

function value_and_gradient(
    f::F,
    extras::PullbackGradientExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, tx = value_and_pullback(
        f, extras.pullback_extras, backend, x, Tangents(true), contexts...
    )
    return y, only(tx)
end

function value_and_gradient!(
    f::F,
    grad,
    extras::PullbackGradientExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, _ = value_and_pullback!(
        f, Tangents(grad), extras.pullback_extras, backend, x, Tangents(true), contexts...
    )
    return y, grad
end

function gradient(
    f::F,
    extras::PullbackGradientExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    tx = pullback(f, extras.pullback_extras, backend, x, Tangents(true), contexts...)
    return only(tx)
end

function gradient!(
    f::F,
    grad,
    extras::PullbackGradientExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    pullback!(
        f, Tangents(grad), extras.pullback_extras, backend, x, Tangents(true), contexts...
    )
    return grad
end
