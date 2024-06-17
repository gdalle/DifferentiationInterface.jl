abstract type FromPrimitive <: AbstractADType end

ADTypes.mode(fromprim::FromPrimitive) = mode(fromprim.backend)
check_available(fromprim::FromPrimitive) = check_available(fromprim.backend)
twoarg_support(fromprim::FromPrimitive) = twoarg_support(fromprim.backend)

## Forward

struct AutoFromForwardPrimitive{B} <: FromPrimitive
    backend::B
end

function prepare_pushforward(f, fromprim::AutoFromForwardPrimitive, x, dx)
    return prepare_pushforward(f, fromprim.backend, x, dx)
end

function prepare_pushforward(f!, y, fromprim::AutoFromForwardPrimitive, x, dx)
    return prepare_pushforward(f!, y, fromprim.backend, x, dx)
end

function value_and_pushforward(
    f, fromprim::AutoFromForwardPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward(f, fromprim.backend, x, dx, extras)
end

function value_and_pushforward(
    f!, y, fromprim::AutoFromForwardPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward(f!, y, fromprim.backend, x, dx, extras)
end

function value_and_pushforward!(
    f, dy, fromprim::AutoFromForwardPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward!(f, dy, fromprim.backend, x, dx, extras)
end

function value_and_pushforward!(
    f!, y, dy, fromprim::AutoFromForwardPrimitive, x, dx, extras::PushforwardExtras
)
    return value_and_pushforward!(f!, y, dy, fromprim.backend, x, dx, extras)
end

## Reverse

struct AutoFromReversePrimitive{B} <: FromPrimitive
    backend::B
end

function prepare_pullback(f, fromprim::AutoFromReversePrimitive, x, dy)
    return prepare_pullback(f, fromprim.backend, x, dy)
end

function prepare_pullback(f!, y, fromprim::AutoFromReversePrimitive, x, dy)
    return prepare_pullback(f!, y, fromprim.backend, x, dy)
end

function value_and_pullback(
    f, fromprim::AutoFromReversePrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback(f, fromprim.backend, x, dy, extras)
end

function value_and_pullback(
    f!, y, fromprim::AutoFromReversePrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback(f!, y, fromprim.backend, x, dy, extras)
end

function value_and_pullback!(
    f, dx, fromprim::AutoFromReversePrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback!(f, dx, fromprim.backend, x, dy, extras)
end

function value_and_pullback!(
    f!, y, dx, fromprim::AutoFromReversePrimitive, x, dy, extras::PullbackExtras
)
    return value_and_pullback!(f!, y, dx, fromprim.backend, x, dy, extras)
end
