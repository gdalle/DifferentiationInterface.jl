abstract type FromPrimitive <: AbstractADType end

check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
twoarg_support(fromprim::FromPrimitive) = twoarg_support(fromprim.backend)

## Forward

struct AutoForwardFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoForwardFromPrimitive) = ADTypes.ForwardMode()

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

## Reverse

struct AutoReverseFromPrimitive{B} <: FromPrimitive
    backend::B
end

ADTypes.mode(::AutoReverseFromPrimitive) = ADTypes.ReverseMode()

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
