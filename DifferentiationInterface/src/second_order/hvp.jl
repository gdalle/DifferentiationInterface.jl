## Docstrings

"""
    prepare_hvp(f, backend, x, tx, [contexts...]) -> prep

Create a `prep` object that can be given to [`hvp`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp end

"""
    prepare!_hvp(f, backend, x, tx, [contexts...]) -> new_prep

Same behavior as [`prepare_hvp`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_hvp end

"""
    prepare_hvp_same_point(f, backend, x, tx, [contexts...]) -> prep_same

Create an `prep_same` object that can be given to [`hvp`](@ref) and its variants _if they are applied at the same point `x` and with the same `contexts`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp_same_point end

"""
    hvp(f, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp end

"""
    hvp!(f, tg, [prep,] backend, x, tx, [contexts...]) -> tg

Compute the Hessian-vector product of `f` at point `x` with a tuple of tangents `tx`, overwriting `tg`.

$(document_preparation("hvp"; same_point=true))
"""
function hvp! end

## Preparation

abstract type StandardHVPPrep <: HVPPrep end

struct ForwardOverForwardHVPPrep{G,E2<:PushforwardPrep,E1<:GradientPrep} <: StandardHVPPrep
    inner_gradient::G
    outer_pushforward_prep::E2
    gradient_prep::E1
end

struct ForwardOverReverseHVPPrep{G,E2<:PushforwardPrep,E1<:GradientPrep} <: StandardHVPPrep
    inner_gradient::G
    outer_pushforward_prep::E2
    gradient_prep::E1
end

struct ReverseOverForwardHVPPrep{P,E2<:GradientPrep,E1<:GradientPrep} <: StandardHVPPrep
    inner_pushforward::P
    outer_gradient_prep::E2
    gradient_prep::E1
end

struct ReverseOverReverseHVPPrep{G,E2<:PullbackPrep,E1<:GradientPrep} <: StandardHVPPrep
    inner_gradient::G
    outer_pullback_prep::E2
    gradient_prep::E1
end

function prepare_hvp(
    f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_hvp_aux(hvp_mode(backend), f, backend, x, tx, contexts...)
end

function _prepare_hvp_aux(
    ::ForwardOverForward,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
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
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    return ForwardOverForwardHVPPrep(inner_gradient, outer_pushforward_prep, gradient_prep)
end

function _prepare_hvp_aux(
    ::ForwardOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
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
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    return ForwardOverReverseHVPPrep(inner_gradient, outer_pushforward_prep, gradient_prep)
end

function _prepare_hvp_aux(
    ::ReverseOverForward,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    # gradient of pushforward
    function inner_pushforward(_x, _dx, unannotated_contexts...)
        annotated_contexts = rewrap(unannotated_contexts...)
        ty = pushforward(f, nested(inner(backend)), _x, (_dx,), annotated_contexts...)
        return only(ty)
    end
    outer_gradient_prep = prepare_gradient(
        inner_pushforward, outer(backend), x, contexts...
    )
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    return ReverseOverForwardHVPPrep(inner_pushforward, outer_gradient_prep, gradient_prep)
end

function _prepare_hvp_aux(
    ::ReverseOverReverse,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
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
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    return ReverseOverReverseHVPPrep(inner_gradient, outer_pullback_prep, gradient_prep)
end

## One argument

function hvp(
    f::F,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward(
        inner_gradient, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp(
    f::F,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward(
        inner_gradient, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp(
    f::F,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_pushforward, outer_gradient_prep) = prep
    tg = map(tx) do dx
        gradient(inner_pushforward, outer(backend), x, Constant(dx), contexts...)
    end
    return tg
end

function hvp(
    f::F,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient, outer_pullback_prep) = prep
    return pullback(inner_gradient, outer_pullback_prep, outer(backend), x, tx, contexts...)
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward!(
        inner_gradient, tg, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient, outer_pushforward_prep) = prep
    return pushforward!(
        inner_gradient, tg, outer_pushforward_prep, outer(backend), x, tx, contexts...
    )
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ReverseOverForwardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_pushforward, outer_gradient_prep) = prep
    for b in eachindex(tx, tg)
        gradient!(
            inner_pushforward,
            tg[b],
            outer_gradient_prep,
            outer(backend),
            x,
            Constant(tx[b]),
            contexts...,
        )
    end
    return tg
end

function hvp!(
    f::F,
    tg::NTuple,
    prep::ReverseOverReverseHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient, outer_pullback_prep) = prep
    return pullback!(
        inner_gradient, tg, outer_pullback_prep, outer(backend), x, tx, contexts...
    )
end

function gradient_and_hvp(
    f::F,
    prep::StandardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    tg = hvp(f, prep, backend, x, tx, contexts...)
    grad = gradient(f, prep.gradient_prep, inner(backend), x, contexts...)
    return grad, tg
end

function gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::StandardHVPPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    hvp!(f, tg, prep, backend, x, tx, contexts...)
    gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    return grad, tg
end
