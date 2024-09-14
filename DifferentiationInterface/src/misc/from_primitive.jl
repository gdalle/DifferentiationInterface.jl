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

struct FromPrimitivePushforwardExtras{E<:PushforwardExtras} <: PushforwardExtras
    pushforward_extras::E
end

function prepare_pushforward(
    f::F, fromprim::AutoForwardFromPrimitive, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    primitive_extras = prepare_pushforward(f, fromprim.backend, x, tx, contexts...)
    return FromPrimitivePushforwardExtras(primitive_extras)
end

function prepare_pushforward(
    f!::F,
    y,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    primitive_extras = prepare_pushforward(f!, y, fromprim.backend, x, tx, contexts...)
    return FromPrimitivePushforwardExtras(primitive_extras)
end

function value_and_pushforward(
    f::F,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(
        f, extras.pushforward_extras, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward(
    f!::F,
    y,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(
        f!, y, extras.pushforward_extras, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f::F,
    ty::Tangents,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(
        f, ty, extras.pushforward_extras, fromprim.backend, x, tx, contexts...
    )
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(
        f!, y, ty, extras.pushforward_extras, fromprim.backend, x, tx, contexts...
    )
end

## Reverse

struct AutoReverseFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

struct FromPrimitivePullbackExtras{E<:PullbackExtras} <: PullbackExtras
    pullback_extras::E
end

function prepare_pullback(
    f::F, fromprim::AutoReverseFromPrimitive, x, ty::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    primitive_extras = prepare_pullback(f, fromprim.backend, x, ty, contexts...)
    return FromPrimitivePullbackExtras(primitive_extras)
end

function prepare_pullback(
    f!::F,
    y,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    primitive_extras = prepare_pullback(f!, y, fromprim.backend, x, ty, contexts...)
    return FromPrimitivePullbackExtras(primitive_extras)
end

function value_and_pullback(
    f::F,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(
        f, extras.pullback_extras, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback(
    f!::F,
    y,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(
        f!, y, extras.pullback_extras, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f::F,
    tx::Tangents,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(
        f, tx, extras.pullback_extras, fromprim.backend, x, ty, contexts...
    )
end

function value_and_pullback!(
    f!::F,
    y,
    tx::Tangents,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(
        f!, y, tx, extras.pullback_extras, fromprim.backend, x, ty, contexts...
    )
end
