abstract type FromPrimitive <: AbstractADType end

function basis(fromprim::FromPrimitive, x::AbstractArray, i)
    return basis(fromprim.backend, x, i)
end

function multibasis(fromprim::FromPrimitive, x::AbstractArray, inds)
    return multibasis(fromprim.backend, x, inds)
end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
twoarg_support(fromprim::FromPrimitive) = twoarg_support(fromprim.backend)

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

function prepare_pushforward(f, fromprim::AutoForwardFromPrimitive, x, tx::Tangents)
    return FromPrimitivePushforwardExtras(prepare_pushforward(f, fromprim.backend, x, tx))
end

function prepare_pushforward(f!, y, fromprim::AutoForwardFromPrimitive, x, tx::Tangents)
    return FromPrimitivePushforwardExtras(
        prepare_pushforward(f!, y, fromprim.backend, x, tx)
    )
end

function value_and_pushforward(
    f,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
)
    return value_and_pushforward(f, extras.pushforward_extras, fromprim.backend, x, tx)
end

function value_and_pushforward(
    f!,
    y,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
    extras::FromPrimitivePushforwardExtras,
)
    return value_and_pushforward(f!, y, extras.pushforward_extras, fromprim.backend, x, tx)
end

function value_and_pushforward!(
    f,
    ty::Tangents,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
)
    return value_and_pushforward!(f, ty, extras.pushforward_extras, fromprim.backend, x, tx)
end

function value_and_pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::FromPrimitivePushforwardExtras,
    fromprim::AutoForwardFromPrimitive,
    x,
    tx::Tangents,
)
    return value_and_pushforward!(
        f!, y, ty, extras.pushforward_extras, fromprim.backend, x, tx
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

function prepare_pullback(f, fromprim::AutoReverseFromPrimitive, x, ty::Tangents)
    return FromPrimitivePullbackExtras(prepare_pullback(f, fromprim.backend, x, ty))
end

function prepare_pullback(f!, y, fromprim::AutoReverseFromPrimitive, x, ty::Tangents)
    return FromPrimitivePullbackExtras(prepare_pullback(f!, y, fromprim.backend, x, ty))
end

function value_and_pullback(
    f,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
)
    return value_and_pullback(f, extras.pullback_extras, fromprim.backend, x, ty)
end

function value_and_pullback(
    f!,
    y,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
)
    return value_and_pullback(f!, y, extras.pullback_extras, fromprim.backend, x, ty)
end

function value_and_pullback!(
    f,
    tx::Tangents,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
)
    return value_and_pullback!(f, tx, extras.pullback_extras, fromprim.backend, x, ty)
end

function value_and_pullback!(
    f!,
    y,
    tx::Tangents,
    extras::FromPrimitivePullbackExtras,
    fromprim::AutoReverseFromPrimitive,
    x,
    ty::Tangents,
)
    return value_and_pullback!(f!, y, tx, extras.pullback_extras, fromprim.backend, x, ty)
end
