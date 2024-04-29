## Docstrings

"""
    prepare_gradient(f, backend, x) -> extras

Create an `extras` object subtyping [`GradientExtras`](@ref) that can be given to gradient operators.
"""
function prepare_gradient end

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient end

"""
    value_and_gradient!(f, grad, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient! end

"""
    gradient(f, backend, x, [extras]) -> grad
"""
function gradient end

"""
    gradient!(f, grad, backend, x, [extras]) -> grad
"""
function gradient! end

## Preparation

"""
    GradientExtras

Abstract type for additional information needed by gradient operators.
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
