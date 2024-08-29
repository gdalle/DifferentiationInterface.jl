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

struct ForwardOverForwardHVPExtras{G<:Gradient,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::G
    outer_pushforward_extras::E
end

struct ForwardOverReverseHVPExtras{G<:Gradient,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::G
    outer_pushforward_extras::E
end

struct ReverseOverForwardHVPExtras <: HVPExtras end

struct ReverseOverReverseHVPExtras{G<:Gradient,E<:PullbackExtras} <: HVPExtras
    inner_gradient::G
    outer_pullback_extras::E
end

function prepare_hvp(f::F, backend::AbstractADType, x, tx::Tangents) where {F}
    return prepare_hvp(f, SecondOrder(backend, backend), x, tx)
end

function prepare_hvp(f::F, backend::SecondOrder, x, tx::Tangents) where {F}
    return _prepare_hvp_aux(f, backend, x, tx, hvp_mode(backend))
end

function _prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents, ::ForwardOverForward
) where {F}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, tx)
    return ForwardOverForwardHVPExtras(inner_gradient, outer_pushforward_extras)
end

function _prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents, ::ForwardOverReverse
) where {F}
    # pushforward of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, tx)
    return ForwardOverReverseHVPExtras(inner_gradient, outer_pushforward_extras)
end

function _prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents, ::ReverseOverForward
) where {F}
    # gradient of pushforward
    # uses dx in the closure so it can't be prepared
    return ReverseOverForwardHVPExtras()
end

function _prepare_hvp_aux(
    f::F, backend::SecondOrder, x, tx::Tangents, ::ReverseOverReverse
) where {F}
    # pullback of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pullback_extras = prepare_pullback(inner_gradient, outer(backend), x, tx)
    return ReverseOverReverseHVPExtras(inner_gradient, outer_pullback_extras)
end

## One argument

function hvp(f::F, backend::AbstractADType, x, tx::Tangents, extras::HVPExtras) where {F}
    return hvp(f, SecondOrder(backend, backend), x, tx, extras)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents, ::ReverseOverForwardHVPExtras
) where {F}
    dgs = map(tx.d) do dx
        inner_pushforward = PushforwardFixedSeed(f, nested(inner(backend)), SingleTangent(dx))
        gradient(only ∘ inner_pushforward, outer(backend), x)
    end
    return Tangents(dgs)
end

function hvp(
    f::F, backend::SecondOrder, x, tx::Tangents, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback(inner_gradient, outer(backend), x, tx, outer_pullback_extras)
end

function hvp!(
    f::F, tg::Tangents, backend::AbstractADType, x, tx::Tangents, extras::HVPExtras
) where {F}
    return hvp!(f, tg, SecondOrder(backend, backend), x, tx, extras)
end

function hvp!(
    f::F,
    tg::Tangents,
    backend::SecondOrder,
    x,
    tx::Tangents,
    extras::ForwardOverForwardHVPExtras,
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, tg, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp!(
    f::F,
    tg::Tangents,
    backend::SecondOrder,
    x,
    tx::Tangents,
    extras::ForwardOverReverseHVPExtras,
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, tg, outer(backend), x, tx, outer_pushforward_extras)
end

function hvp!(
    f::F, tg::Tangents, backend::SecondOrder, x, tx::Tangents, ::ReverseOverForwardHVPExtras
) where {F}
    for b in eachindex(tx.d, tg.d)
        inner_pushforward = PushforwardFixedSeed(
            f, nested(inner(backend)), SingleTangent(tx.d[b])
        )
        gradient!(only ∘ inner_pushforward, tg.d[b], outer(backend), x)
    end
    return tg
end

function hvp!(
    f::F,
    tg::Tangents,
    backend::SecondOrder,
    x,
    tx::Tangents,
    extras::ReverseOverReverseHVPExtras,
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback!(inner_gradient, tg, outer(backend), x, tx, outer_pullback_extras)
end
