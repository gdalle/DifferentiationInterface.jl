## Docstrings

"""
    prepare_hvp(f, backend, x, tx, [contexts...]) -> prep

Create an `prep` object that can be given to [`hvp`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp end

"""
    prepare_hvp_same_point(f, backend, x, tx, [contexts...]) -> prep_same

Create an `prep_same` object that can be given to [`hvp`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp_same_point end

"""
    hvp(f, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with [`Tangents`](@ref) `tx`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp end

"""
    hvp!(f, tg, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with [`Tangents`](@ref) `tx`, overwriting `tg`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp! end

## Preparation

struct ForwardOverForwardHVPPrep{G,E<:PushforwardPrep} <: HVPPrep
    inner_gradient::G
    outer_pushforward_prep::E
end

struct ForwardOverReverseHVPPrep{G,E<:PushforwardPrep} <: HVPPrep
    inner_gradient::G
    outer_pushforward_prep::E
end

struct ReverseOverForwardHVPPrep <: HVPPrep end

struct ReverseOverReverseHVPPrep{G,E<:PullbackPrep} <: HVPPrep
    inner_gradient::G
    outer_pullback_prep::E
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
    rewrap = Rewrap(contexts...)
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    function inner_gradient(_x, unannotated_contexts...)
        annotated_contexts = rewrap(unannotated_contexts...)
        return gradient(f, nested(inner(backend)), _x, annotated_contexts...)
    end
    outer_pushforward_prep = prepare_pushforward(
        inner_gradient, outer(backend), x, tx, contexts...
    )
    return ForwardOverForwardHVPPrep(inner_gradient, outer_pushforward_prep)
end

function _prepare_hvp_aux(
    ::ForwardOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    # pushforward of gradient
    function inner_gradient(_x, unannotated_contexts...)
        annotated_contexts = rewrap(unannotated_contexts...)
        return gradient(f, nested(inner(backend)), _x, annotated_contexts...)
    end
    outer_pushforward_prep = prepare_pushforward(
        inner_gradient, outer(backend), x, tx, contexts...
    )
    return ForwardOverReverseHVPPrep(inner_gradient, outer_pushforward_prep)
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
    return ReverseOverForwardHVPPrep()
end

function _prepare_hvp_aux(
    ::ReverseOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    # pullback of gradient
    function inner_gradient(_x, unannotated_contexts...)
        annotated_contexts = rewrap(unannotated_contexts...)
        return gradient(f, nested(inner(backend)), _x, annotated_contexts...)
    end
    outer_pullback_prep = prepare_pullback(
        inner_gradient, outer(backend), x, tx, contexts...
    )
    return ReverseOverReverseHVPPrep(inner_gradient, outer_pullback_prep)
end

## One argument

function hvp(
    f::F,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward(
        inner_gradient, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp(
    f::F,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward(
        inner_gradient, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp(
    f::F,
    ::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    tg = map(tx) do dx
        function inner_pushforward(_x, unannotated_contexts...)
            annotated_contexts = rewrap(unannotated_contexts...)
            return only(
                pushforward(
                    f, nested(inner(backend)), _x, Tangents(dx), annotated_contexts...
                ),
            )
        end
        gradient(only ∘ inner_pushforward, outer(backend), x, contexts...)
    end
    return tg
end

function hvp(
    f::F,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pullback_prep) = prep
    return pullback(inner_gradient, outer_pullback_prep, outer(backend), x, tx, contexts...)
end

function hvp!(
    f::F,
    tg::Tangents,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward!(
        inner_gradient, tg, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp!(
    f::F,
    tg::Tangents,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward!(
        inner_gradient, tg, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp!(
    f::F,
    tg::Tangents,
    ::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    for b in eachindex(tx.d, tg.d)
        function inner_pushforward(_x, unannotated_contexts...)
            annotated_contexts = rewrap(unannotated_contexts...)
            return only(
                pushforward(
                    f, nested(inner(backend)), _x, Tangents(tx.d[b]), annotated_contexts...
                ),
            )
        end
        gradient!(only ∘ inner_pushforward, tg.d[b], outer(backend), x, contexts...)
    end
    return tg
end

function hvp!(
    f::F,
    tg::Tangents,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_gradient, outer_pullback_prep) = prep
    return pullback!(
        inner_gradient, tg, outer_pullback_prep, outer(backend), x, tx, contexts...
    )
end
