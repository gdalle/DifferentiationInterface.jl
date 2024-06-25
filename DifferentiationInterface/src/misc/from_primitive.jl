abstract type FromPrimitive <: AbstractADType end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
twoarg_support(fromprim::FromPrimitive) = twoarg_support(fromprim.backend)

## Forward

struct AutoForwardFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoForwardFromPrimitive) = ADTypes.ForwardMode()

### Standard

function prepare_pushforward(f, fromprim::AutoForwardFromPrimitive, x, dx)
    return prepare_pushforward(f, fromprim.backend, x, dx)
end

function prepare_pushforward(f!, y, fromprim::AutoForwardFromPrimitive, x, dx)
    return prepare_pushforward(f!, y, fromprim.backend, x, dx)
end

function value_and_pushforward(
    f, fromprim::AutoForwardFromPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward(f, fromprim.backend, x, dx, extras)
end

function value_and_pushforward(
    f!, y, fromprim::AutoForwardFromPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward(f!, y, fromprim.backend, x, dx, extras)
end

function value_and_pushforward!(
    f, dy, fromprim::AutoForwardFromPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward!(f, dy, fromprim.backend, x, dx, extras)
end

function value_and_pushforward!(
    f!, y, dy, fromprim::AutoForwardFromPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward!(f!, y, dy, fromprim.backend, x, dx, extras)
end

### Batched

function prepare_pushforward_batched(
    f, fromprim::AutoForwardFromPrimitive, x, dx::Batch{B}
) where {B}
    return prepare_pushforward_batched(f, fromprim.backend, x, dx)
end

function prepare_pushforward_batched(
    f!, y, fromprim::AutoForwardFromPrimitive, x, dx::Batch{B}
) where {B}
    return prepare_pushforward_batched(f!, y, fromprim.backend, x, dx)
end

function pushforward_batched(
    f, fromprim::AutoForwardFromPrimitive, x, dx::Batch{B}, extras::PushforwardExtras
) where {B}
    return pushforward_batched(f, fromprim.backend, x, dx, extras)
end

function pushforward_batched(
    f!, y, fromprim::AutoForwardFromPrimitive, x, dx::Batch{B}, extras::PushforwardExtras
) where {B}
    return pushforward_batched(f!, y, fromprim.backend, x, dx, extras)
end

function pushforward_batched!(
    f,
    dy::Batch{B},
    fromprim::AutoForwardFromPrimitive,
    x,
    dx::Batch{B},
    extras::PushforwardExtras,
) where {B}
    return pushforward_batched!(f, dy, fromprim.backend, x, dx, extras)
end

function pushforward_batched!(
    f!,
    y,
    dy::Batch{B},
    fromprim::AutoForwardFromPrimitive,
    x,
    dx::Batch{B},
    extras::PushforwardExtras,
) where {B}
    return pushforward_batched!(f!, y, dy, fromprim.backend, x, dx, extras)
end

## Reverse

struct AutoReverseFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

### Standard

function prepare_pullback(f, fromprim::AutoReverseFromPrimitive, x, dy)
    return prepare_pullback(f, fromprim.backend, x, dy)
end

function prepare_pullback(f!, y, fromprim::AutoReverseFromPrimitive, x, dy)
    return prepare_pullback(f!, y, fromprim.backend, x, dy)
end

function value_and_pullback(
    f, fromprim::AutoReverseFromPrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback(f, fromprim.backend, x, dy, extras)
end

function value_and_pullback(
    f!, y, fromprim::AutoReverseFromPrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback(f!, y, fromprim.backend, x, dy, extras)
end

function value_and_pullback!(
    f, dx, fromprim::AutoReverseFromPrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback!(f, dx, fromprim.backend, x, dy, extras)
end

function value_and_pullback!(
    f!, y, dx, fromprim::AutoReverseFromPrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback!(f!, y, dx, fromprim.backend, x, dy, extras)
end

### Batched

function prepare_pullback_batched(
    f, fromprim::AutoReverseFromPrimitive, x, dy::Batch{B}
) where {B}
    return prepare_pullback_batched(f, fromprim.backend, x, dy)
end

function prepare_pullback_batched(
    f!, y, fromprim::AutoReverseFromPrimitive, x, dy::Batch{B}
) where {B}
    return prepare_pullback_batched(f!, y, fromprim.backend, x, dy)
end

function pullback_batched(
    f, fromprim::AutoReverseFromPrimitive, x, dy::Batch{B}, extras::PullbackExtras
) where {B}
    return pullback_batched(f, fromprim.backend, x, dy, extras)
end

function pullback_batched(
    f!, y, fromprim::AutoReverseFromPrimitive, x, dy::Batch{B}, extras::PullbackExtras
) where {B}
    return pullback_batched(f!, y, fromprim.backend, x, dy, extras)
end

function pullback_batched!(
    f,
    dx::Batch{B},
    fromprim::AutoReverseFromPrimitive,
    x,
    dy::Batch{B},
    extras::PullbackExtras,
) where {B}
    return pullback_batched!(f, dx, fromprim.backend, x, dy, extras)
end

function pullback_batched!(
    f!,
    y,
    dx::Batch{B},
    fromprim::AutoReverseFromPrimitive,
    x,
    dy::Batch{B},
    extras::PullbackExtras,
) where {B}
    return pullback_batched!(f!, y, dx, fromprim.backend, x, dy, extras)
end
