myzero(x::Number) = zero(x)
myzero(x::AbstractArray) = zero(x)

myzero!!(x::Number) = zero(x)
myzero!!(x::AbstractArray) = x .= zero(eltype(x))

## Forward

"""
    AutoZeroForward <: ADTypes.AbstractForwardMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: ADTypes.AbstractForwardMode end

DI.supports_mutation(::AutoZeroForward) = DI.MutationSupported()

DI.prepare_pushforward(f, ::AutoZeroForward, x) = NoPushforwardExtras()
DI.prepare_pushforward(f!, ::AutoZeroForward, y, x) = NoPushforwardExtras()

function DI.value_and_pushforward(f, ::AutoZeroForward, x, dx, ::NoPushforwardExtras)
    y = f(x)
    dy = myzero(y)
    return y, dy
end

function DI.value_and_pushforward!!(f, dy, ::AutoZeroForward, x, dx, ::NoPushforwardExtras)
    y = f(x)
    dy = myzero!!(dy)
    return y, dy
end

function DI.value_and_pushforward!!(
    f!, y, dy, ::AutoZeroForward, x, dx, ::NoPushforwardExtras
)
    f!(y, x)
    dy = myzero!!(dy)
    return y, dy
end

## Reverse

"""
    AutoZeroReverse <: ADTypes.AbstractReverseMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: ADTypes.AbstractReverseMode end

DI.supports_mutation(::AutoZeroReverse) = DI.MutationSupported()

DI.prepare_pullback(f, ::AutoZeroReverse, x) = NoPullbackExtras()
DI.prepare_pullback(f!, ::AutoZeroReverse, y, x) = NoPullbackExtras()

function DI.value_and_pullback(f, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    y = f(x)
    dx = myzero(x)
    return y, dx
end

function DI.value_and_pullback!!(f, dx, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    y = f(x)
    dx = myzero!!(dx)
    return y, dx
end

function DI.value_and_pullback!!(f!, y, dx, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    f!(y, x)
    dx = myzero!!(dx)
    return y, dx
end
