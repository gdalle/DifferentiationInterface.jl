## Docstrings

"""
    prepare_hvp(f, backend, x, tx) -> extras

Create an `extras` object that can be given to [`hvp`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp end

"""
    prepare_hvp_same_point(f, backend, x, tx) -> extras_same

Create an `extras_same` object that can be given to [`hvp`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp_same_point end

"""
    hvp(f, [extras,] backend, x, tx) -> tg

Compute the Hessian-vector product of `f` at point `x` with tangent `tx` of type [`Tangents`](@ref).

$(document_preparation("hvp"; same_point=true))
"""
function hvp end

"""
    hvp!(f, dg, [extras,] backend, x, tx) -> tg

Compute the Hessian-vector product of `f` at point `x` with tangent `tx` of type [`Tangents`](@ref), overwriting `tg`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp! end

## Preparation

struct ForwardOverForwardHVPExtras{G,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::G
    outer_pushforward_extras::E
end

struct ForwardOverReverseHVPExtras{G,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::G
    outer_pushforward_extras::E
end

struct ReverseOverForwardHVPExtras <: HVPExtras end

struct ReverseOverReverseHVPExtras{G,E<:PullbackExtras} <: HVPExtras
    inner_gradient::G
    outer_pullback_extras::E
end

function prepare_hvp(
    f::F, backend::AbstractADType, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_hvp_aux(hvp_mode(backend), f, backend, x, tx, contexts...)
end

function _prepare_hvp_aux(
    ::ForwardOverForward,
    f::F,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    function inner_gradient(_x, unannotated_contexts...)
        annotated_contexts = map.(typeof.(contexts), unannotated_contexts)
        return gradient(f, nested(maybe_inner(backend)), _x, annotated_contexts...)
    end
    outer_pushforward_extras = prepare_pushforward(
        inner_gradient, maybe_outer(backend), x, tx, contexts...
    )
    return ForwardOverForwardHVPExtras(inner_gradient, outer_pushforward_extras)
end

function _prepare_hvp_aux(
    ::ForwardOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    # pushforward of gradient
    function inner_gradient(_x, unannotated_contexts...)
        annotated_contexts = map.(typeof.(contexts), unannotated_contexts)
        return gradient(f, nested(maybe_inner(backend)), _x, annotated_contexts...)
    end
    outer_pushforward_extras = prepare_pushforward(
        inner_gradient, maybe_outer(backend), x, tx, contexts...
    )
    return ForwardOverReverseHVPExtras(inner_gradient, outer_pushforward_extras)
end

function _prepare_hvp_aux(
    ::ReverseOverForward,
    f::F,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    # gradient of pushforward
    # uses dx in the closure so it can't be prepared
    return ReverseOverForwardHVPExtras()
end

function _prepare_hvp_aux(
    ::ReverseOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    # pullback of gradient
    function inner_gradient(_x, unannotated_contexts...)
        annotated_contexts = map.(typeof.(contexts), unannotated_contexts)
        return gradient(f, nested(maybe_inner(backend)), _x, annotated_contexts...)
    end
    outer_pullback_extras = prepare_pullback(
        inner_gradient, maybe_outer(backend), x, tx, contexts...
    )
    return ReverseOverReverseHVPExtras(inner_gradient, outer_pullback_extras)
end

## One argument

function hvp(
    f::F,
    extras::ForwardOverForwardHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(
        inner_gradient, outer_pushforward_extras, maybe_outer(backend), x, tx, contexts...
    )
end

function hvp(
    f::F,
    extras::ForwardOverReverseHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(
        inner_gradient, outer_pushforward_extras, maybe_outer(backend), x, tx, contexts...
    )
end

function hvp(
    f::F,
    ::ReverseOverForwardHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    dgs = map(tx.d) do dx
        function inner_pushforward(_x)
            return only(
                pushforward(f, nested(maybe_inner(backend)), _x, Tangents(dx), contexts...),
            )
        end
        gradient(only ∘ inner_pushforward, maybe_outer(backend), x, contexts...)
    end
    return tg
end

function hvp(
    f::F,
    extras::ReverseOverReverseHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback(
        inner_gradient, outer_pullback_extras, maybe_outer(backend), x, tx, contexts...
    )
end

function hvp!(
    f::F,
    tg::Tangents,
    extras::ForwardOverForwardHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(
        inner_gradient,
        tg,
        outer_pushforward_extras,
        maybe_outer(backend),
        x,
        tx,
        contexts...,
    )
end

function hvp!(
    f::F,
    tg::Tangents,
    extras::ForwardOverReverseHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(
        inner_gradient,
        tg,
        outer_pushforward_extras,
        maybe_outer(backend),
        x,
        tx,
        contexts...,
    )
end

function hvp!(
    f::F,
    tg::Tangents,
    ::ReverseOverForwardHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    for b in eachindex(tx.d, tg.d)
        function inner_pushforward(x)
            return only(
                pushforward(
                    f, nested(maybe_inner(backend)), x, Tangents(tx.d[b]), contexts...
                ),
            )
        end
        gradient!(only ∘ inner_pushforward, tg.d[b], maybe_outer(backend), x, contexts...)
    end
    return tg
end

function hvp!(
    f::F,
    tg::Tangents,
    extras::ReverseOverReverseHVPExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback!(
        inner_gradient, tg, outer_pullback_extras, maybe_outer(backend), x, tx, contexts...
    )
end
