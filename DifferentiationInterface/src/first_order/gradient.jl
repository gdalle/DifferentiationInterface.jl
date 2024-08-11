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

## Functors

"""
    Gradient

Functor computing the gradient of `f` with a fixed `backend`.

!!! warning
    This type is not part of the public API.

# Constructor

    Gradient(f, backend, extras=nothing)

If `extras` is provided, the gradient closure will skip preparation.

# Example

```jldoctest
using DifferentiationInterface
import Zygote

g = DifferentiationInterface.Gradient(x -> sum(abs2, x), AutoZygote())
g([2.0, 3.0])

# output

2-element Vector{Float64}:
 4.0
 6.0
```
"""
struct Gradient{F,B,E}
    f::F
    backend::B
    extras::E
end

Gradient(f, backend::AbstractADType) = Gradient(f, backend, nothing)

function (g::Gradient{F,B,Nothing})(x) where {F,B}
    @compat (; f, backend) = g
    return gradient(f, backend, x)
end

function (g::Gradient{F,B,<:GradientExtras})(x) where {F,B}
    @compat (; f, backend, extras) = g
    return gradient(f, backend, x, extras)
end
