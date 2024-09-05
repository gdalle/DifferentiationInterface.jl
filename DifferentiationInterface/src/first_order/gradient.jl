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

function prepare_gradient(f::F, backend::AbstractADType, x) where {F}
    pullback_extras = prepare_pullback(f, backend, x, SingleTangent(true))
    return PullbackGradientExtras(pullback_extras)
end

## One argument

function value_and_gradient(
    f::F, extras::PullbackGradientExtras, backend::AbstractADType, x
) where {F}
    y, tx = value_and_pullback(f, extras.pullback_extras, backend, x, SingleTangent(true))
    return y, only(tx)
end

function value_and_gradient!(
    f::F, grad, extras::PullbackGradientExtras, backend::AbstractADType, x
) where {F}
    y, _ = value_and_pullback!(
        f, SingleTangent(grad), extras.pullback_extras, backend, x, SingleTangent(true)
    )
    return y, grad
end

function gradient(
    f::F, extras::PullbackGradientExtras, backend::AbstractADType, x
) where {F}
    tx = pullback(f, extras.pullback_extras, backend, x, SingleTangent(true))
    return only(tx)
end

function gradient!(
    f::F, grad, extras::PullbackGradientExtras, backend::AbstractADType, x
) where {F}
    pullback!(
        f, SingleTangent(grad), extras.pullback_extras, backend, x, SingleTangent(true)
    )
    return grad
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
    return gradient(f, extras, backend, x)
end
