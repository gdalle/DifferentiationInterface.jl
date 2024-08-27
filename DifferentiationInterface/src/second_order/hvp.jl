## Docstrings

"""
    prepare_hvp(f, backend, x, dx) -> extras

Create an `extras` object that can be given to [`hvp`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp end

"""
    prepare_hvp_same_point(f, backend, x, dx) -> extras_same

Create an `extras_same` object that can be given to [`hvp`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp_same_point end

"""
    hvp(f, backend, x, dx, [extras]) -> dg

Compute the Hessian-vector product of `f` at point `x` with seed `dx`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp end

"""
    hvp!(f, dg, backend, x, dx, [extras]) -> dg

Compute the Hessian-vector product of `f` at point `x` with seed `dx`, overwriting `dg`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp! end

## Preparation

### Extras types

struct ForwardOverForwardHVPExtras{G<:Gradient,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::G
    outer_pushforward_extras::E
end

struct ForwardOverReverseHVPExtras{G<:Gradient,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::G
    outer_pushforward_extras::E
end

struct ReverseOverForwardHVPExtras{E<:GradientExtras} <: HVPExtras
    outer_gradient_extras::E
end

struct ReverseOverReverseHVPExtras{G<:Gradient,E<:PullbackExtras} <: HVPExtras
    inner_gradient::G
    outer_pullback_extras::E
end

### Different point

function prepare_hvp(f::F, backend::AbstractADType, x, dx) where {F}
    return prepare_hvp(f, SecondOrder(backend, backend), x, dx)
end

function prepare_hvp(f::F, backend::SecondOrder, x, dx) where {F}
    return _prepare_hvp_aux(f, backend, x, dx, hvp_mode(backend))
end

function _prepare_hvp_aux(f::F, backend::SecondOrder, x, dx, ::ForwardOverForward) where {F}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, dx)
    return ForwardOverForwardHVPExtras(inner_gradient, outer_pushforward_extras)
end

function _prepare_hvp_aux(f::F, backend::SecondOrder, x, dx, ::ForwardOverReverse) where {F}
    # pushforward of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, dx)
    return ForwardOverReverseHVPExtras(inner_gradient, outer_pushforward_extras)
end

function _prepare_hvp_aux(f::F, backend::SecondOrder, x, dx, ::ReverseOverForward) where {F}
    # gradient of pushforward
    # uses dx in the closure so it can't be stored
    inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), dx)
    outer_gradient_extras = prepare_gradient(inner_pushforward, outer(backend), x)
    return ReverseOverForwardHVPExtras(outer_gradient_extras)
end

function _prepare_hvp_aux(f::F, backend::SecondOrder, x, dx, ::ReverseOverReverse) where {F}
    # pullback of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pullback_extras = prepare_pullback(inner_gradient, outer(backend), x, dx)
    return ReverseOverReverseHVPExtras(inner_gradient, outer_pullback_extras)
end

### Same point

function prepare_hvp_same_point(
    f::F, backend::AbstractADType, x, dx, extras::HVPExtras
) where {F}
    return extras
end

function prepare_hvp_same_point(f::F, backend::AbstractADType, x, dx) where {F}
    extras = prepare_hvp(f, backend, x, dx)
    return prepare_hvp_same_point(f, backend, x, dx, extras)
end

## One argument

function hvp(f::F, backend::AbstractADType, x, dx, extras::HVPExtras) where {F}
    return hvp(f, SecondOrder(backend, backend), x, dx, extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ReverseOverForwardHVPExtras
) where {F}
    @compat (; outer_gradient_extras) = extras
    inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), dx)
    return gradient(inner_pushforward, outer(backend), x, outer_gradient_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback(inner_gradient, outer(backend), x, dx, outer_pullback_extras)
end

function hvp!(f::F, dg, backend::AbstractADType, x, dx, extras::HVPExtras) where {F}
    return hvp!(f, dg, SecondOrder(backend, backend), x, dx, extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, dg, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, dg, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ReverseOverForwardHVPExtras
) where {F}
    @compat (; outer_gradient_extras) = extras
    inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), dx)
    return gradient!(inner_pushforward, dg, outer(backend), x, outer_gradient_extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback!(inner_gradient, dg, outer(backend), x, dx, outer_pullback_extras)
end
