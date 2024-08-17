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

"""
    HVPExtras

Abstract type for additional information needed by [`hvp`](@ref) and its variants.
"""
abstract type HVPExtras <: Extras end

struct NoHVPExtras <: HVPExtras end

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

function prepare_hvp(f::F, backend::AbstractADType, x, tx::Tangents{1}) where {F}
    return prepare_hvp(f, SecondOrder(backend, backend), x, tx)
end

function prepare_hvp(f::F, backend::SecondOrder, x, tx::Tangents{1}) where {F}
    return prepare_hvp_aux(f, backend, x, tx, hvp_mode(backend))
end

function prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, ::ForwardOverForward
) where {F}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, tx)
    return ForwardOverForwardHVPExtras(inner_gradient, outer_pushforward_extras)
end

function prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, ::ForwardOverReverse
) where {F}
    # pushforward of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, tx)
    return ForwardOverReverseHVPExtras(inner_gradient, outer_pushforward_extras)
end

function prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, ::ReverseOverForward
) where {F}
    # gradient of pushforward
    # uses dx in the closure so it can't be stored
    inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), tx)
    outer_gradient_extras = prepare_gradient(inner_pushforward, outer(backend), x)
    return ReverseOverForwardHVPExtras(outer_gradient_extras)
end

function prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, ::ReverseOverReverse
) where {F}
    # pullback of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pullback_extras = prepare_pullback(inner_gradient, outer(backend), x, tx)
    return ReverseOverReverseHVPExtras(inner_gradient, outer_pullback_extras)
end

## One argument

function hvp(f::F, backend::AbstractADType, x, tx::Tangents{1}, extras::HVPExtras) where {F}
    return hvp(f, SecondOrder(backend, backend), x, tx, extras)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, extras::ReverseOverForwardHVPExtras
) where {F}
    @compat (; outer_gradient_extras) = extras
    inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), dx)
    dg = gradient(inner_pushforward, outer(backend), x, outer_gradient_extras)
    return Tangents(dg)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents{1}, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback(inner_gradient, outer(backend), x, tx, outer_pullback_extras)
end

function hvp!(
    f::F, tg::Tangents{1}, backend::AbstractADType, x, tx::Tangents{1}, extras::HVPExtras
) where {F}
    return hvp!(f, tg, SecondOrder(backend, backend), x, tx, extras)
end

function hvp!(
    f::F, tg, backend::SecondOrder, x, tx::Tangents{1}, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, tg, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp!(
    f::F,
    tg::Tangents{1},
    backend::SecondOrder,
    x,
    tx::Tangents{1},
    extras::ForwardOverReverseHVPExtras,
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, tg, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp!(
    f::F,
    tg::Tangents{1},
    backend::SecondOrder,
    x,
    tx::Tangents{1},
    extras::ReverseOverForwardHVPExtras,
) where {F}
    @compat (; outer_gradient_extras) = extras
    inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), dx)
    gradient!(inner_pushforward, only(tg), outer(backend), x, outer_gradient_extras)
    return tg
end

function hvp!(
    f::F,
    tg::Tangents{1},
    backend::SecondOrder,
    x,
    tx::Tangents{1},
    extras::ReverseOverReverseHVPExtras,
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback!(inner_gradient, tg, outer(backend), x, tx, outer_pullback_extras)
end
