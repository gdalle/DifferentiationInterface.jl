abstract type FromPrimitive <: AbstractADType end

function basis(fromprim::FromPrimitive, x::AbstractArray, i)
    return basis(fromprim.backend, x, i)
end

function multibasis(fromprim::FromPrimitive, x::AbstractArray, inds)
    return multibasis(fromprim.backend, x, inds)
end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
inplace_support(fromprim::FromPrimitive) = inplace_support(fromprim.backend)

function pick_batchsize(fromprim::FromPrimitive, dimension::Integer)
    return pick_batchsize(fromprim.backend, dimension)
end

## Forward

struct AutoForwardFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoForwardFromPrimitive) = ADTypes.ForwardMode()

struct FromPrimitivePushforwardPrep{E<:PushforwardPrep} <: PushforwardPrep
    pushforward_prep::E
end

function prepare_pushforward(
    f::F, fromprim::AutoForwardFromPrimitive, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    primitive_prep = prepare_pushforward(f, fromprim.backend, x, tx, contexts...)
    return FromPrimitivePushforwardPrep(primitive_prep)
end

function prepare_pushforward(
    f!::F,
    y,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    primitive_prep = prepare_pushforward(f!, y, fromprim.backend, x, tx, contexts...)
    return FromPrimitivePushforwardPrep(primitive_prep)
end

function value_and_pushforward(
    f::F,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(
        f, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward(
    f!::F,
    y,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(
        f!, y, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f::F,
    ty::Tangents,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(
        f, ty, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    prep::FromPrimitivePushforwardPrep,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(
        f!, y, ty, prep.pushforward_prep, fromprim.backend, x, tx, contexts...
    )
end

## Reverse

struct AutoReverseFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

struct FromPrimitivePullbackPrep{E<:PullbackPrep} <: PullbackPrep
    pullback_prep::E
end

function prepare_pullback(
    f::F, fromprim::AutoReverseFromPrimitive, x, ty::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    primitive_prep = prepare_pullback(f, fromprim.backend, x, ty, contexts...)
    return FromPrimitivePullbackPrep(primitive_prep)
end

function prepare_pullback(
    f!::F,
    y,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    primitive_prep = prepare_pullback(f!, y, fromprim.backend, x, ty, contexts...)
    return FromPrimitivePullbackPrep(primitive_prep)
end

function value_and_pullback(
    f::F,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(f, prep.pullback_prep, fromprim.backend, x, ty, contexts...)
end

function value_and_pullback(
    f!::F,
    y,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(
        f!, y, prep.pullback_prep, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f::F,
    tx::Tangents,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(
        f, tx, prep.pullback_prep, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f!::F,
    y,
    tx::Tangents,
    prep::FromPrimitivePullbackPrep,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(
        f!, y, tx, prep.pullback_prep, fromprim.backend, x, ty, contexts...
    )
end
