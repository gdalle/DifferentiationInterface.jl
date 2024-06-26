abstract type FromPrimitive <: AbstractADType end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
twoarg_support(fromprim::FromPrimitive) = twoarg_support(fromprim.backend)

## Forward

struct AutoForwardFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoForwardFromPrimitive) = ADTypes.ForwardMode()

struct FromPrimitivePushforwardExtras{E<:PushforwardExtras} <: PushforwardExtras
    pushforward_extras::E
end

### Standard

function prepare_pushforward(f, fromprim::AutoForwardFromPrimitive, x, dx)
    return FromPrimitivePushforwardExtras(prepare_pushforward(f, fromprim.backend, x, dx))
end

function prepare_pushforward(f!, y, fromprim::AutoForwardFromPrimitive, x, dx)
    return FromPrimitivePushforwardExtras(
        prepare_pushforward(f!, y, fromprim.backend, x, dx)
    )
end

function value_and_pushforward(
    f, fromprim::AutoForwardFromPrimitive, x, dx, extras::FromPrimitivePushforwardExtras
)
    return value_and_pushforward(f, fromprim.backend, x, dx, extras.pushforward_extras)
end

function value_and_pushforward(
    f!, y, fromprim::AutoForwardFromPrimitive, x, dx, extras::FromPrimitivePushforwardExtras
)
    return value_and_pushforward(f!, y, fromprim.backend, x, dx, extras.pushforward_extras)
end

function value_and_pushforward!(
    f, dy, fromprim::AutoForwardFromPrimitive, x, dx, extras::FromPrimitivePushforwardExtras
)
    return value_and_pushforward!(f, dy, fromprim.backend, x, dx, extras.pushforward_extras)
end

function value_and_pushforward!(
    f!,
    y,
    dy,
    fromprim::AutoForwardFromPrimitive,
    x,
    dx,
    extras::FromPrimitivePushforwardExtras,
)
    return value_and_pushforward!(
        f!, y, dy, fromprim.backend, x, dx, extras.pushforward_extras
    )
end

### Batched

function prepare_pushforward_batched(f, fromprim::AutoForwardFromPrimitive, x, dx::Batch)
    return FromPrimitivePushforwardExtras(
        prepare_pushforward_batched(f, fromprim.backend, x, dx)
    )
end

function prepare_pushforward_batched(
    f!, y, fromprim::AutoForwardFromPrimitive, x, dx::Batch
)
    return FromPrimitivePushforwardExtras(
        prepare_pushforward_batched(f!, y, fromprim.backend, x, dx)
    )
end

function pushforward_batched(
    f,
    fromprim::AutoForwardFromPrimitive,
    x,
    dx::Batch,
    extras::FromPrimitivePushforwardExtras,
)
    return pushforward_batched(f, fromprim.backend, x, dx, extras.pushforward_extras)
end

function pushforward_batched(
    f!,
    y,
    fromprim::AutoForwardFromPrimitive,
    x,
    dx::Batch,
    extras::FromPrimitivePushforwardExtras,
)
    return pushforward_batched(f!, y, fromprim.backend, x, dx, extras.pushforward_extras)
end

function pushforward_batched!(
    f,
    dy::Batch,
    fromprim::AutoForwardFromPrimitive,
    x,
    dx::Batch,
    extras::FromPrimitivePushforwardExtras,
)
    return pushforward_batched!(f, dy, fromprim.backend, x, dx, extras.pushforward_extras)
end

function pushforward_batched!(
    f!,
    y,
    dy::Batch,
    fromprim::AutoForwardFromPrimitive,
    x,
    dx::Batch,
    extras::FromPrimitivePushforwardExtras,
)
    return pushforward_batched!(
        f!, y, dy, fromprim.backend, x, dx, extras.pushforward_extras
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

### Standard

function prepare_pullback(f, fromprim::AutoReverseFromPrimitive, x, dy)
    return FromPrimitivePullbackExtras(prepare_pullback(f, fromprim.backend, x, dy))
end

function prepare_pullback(f!, y, fromprim::AutoReverseFromPrimitive, x, dy)
    return FromPrimitivePullbackExtras(prepare_pullback(f!, y, fromprim.backend, x, dy))
end

function value_and_pullback(
    f, fromprim::AutoReverseFromPrimitive, x, dy, extras::FromPrimitivePullbackExtras
)
    return value_and_pullback(f, fromprim.backend, x, dy, extras.pullback_extras)
end

function value_and_pullback(
    f!, y, fromprim::AutoReverseFromPrimitive, x, dy, extras::FromPrimitivePullbackExtras
)
    return value_and_pullback(f!, y, fromprim.backend, x, dy, extras.pullback_extras)
end

function value_and_pullback!(
    f, dx, fromprim::AutoReverseFromPrimitive, x, dy, extras::FromPrimitivePullbackExtras
)
    return value_and_pullback!(f, dx, fromprim.backend, x, dy, extras.pullback_extras)
end

function value_and_pullback!(
    f!,
    y,
    dx,
    fromprim::AutoReverseFromPrimitive,
    x,
    dy,
    extras::FromPrimitivePullbackExtras,
)
    return value_and_pullback!(f!, y, dx, fromprim.backend, x, dy, extras.pullback_extras)
end

### Batched

function prepare_pullback_batched(f, fromprim::AutoReverseFromPrimitive, x, dy::Batch)
    return FromPrimitivePullbackExtras(prepare_pullback_batched(f, fromprim.backend, x, dy))
end

function prepare_pullback_batched(f!, y, fromprim::AutoReverseFromPrimitive, x, dy::Batch)
    return FromPrimitivePullbackExtras(
        prepare_pullback_batched(f!, y, fromprim.backend, x, dy)
    )
end

function pullback_batched(
    f, fromprim::AutoReverseFromPrimitive, x, dy::Batch, extras::FromPrimitivePullbackExtras
)
    return pullback_batched(f, fromprim.backend, x, dy, extras.pullback_extras)
end

function pullback_batched(
    f!,
    y,
    fromprim::AutoReverseFromPrimitive,
    x,
    dy::Batch,
    extras::FromPrimitivePullbackExtras,
)
    return pullback_batched(f!, y, fromprim.backend, x, dy, extras.pullback_extras)
end

function pullback_batched!(
    f,
    dx::Batch,
    fromprim::AutoReverseFromPrimitive,
    x,
    dy::Batch,
    extras::FromPrimitivePullbackExtras,
)
    return pullback_batched!(f, dx, fromprim.backend, x, dy, extras.pullback_extras)
end

function pullback_batched!(
    f!,
    y,
    dx::Batch,
    fromprim::AutoReverseFromPrimitive,
    x,
    dy::Batch,
    extras::FromPrimitivePullbackExtras,
)
    return pullback_batched!(f!, y, dx, fromprim.backend, x, dy, extras.pullback_extras)
end
