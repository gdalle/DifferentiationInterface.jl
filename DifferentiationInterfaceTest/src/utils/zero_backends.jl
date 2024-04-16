zero!(x::AbstractArray) = x .= zero(eltype(x))

## Forward

"""
    AutoZeroForward <: ADTypes.AbstractForwardMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: ADTypes.AbstractForwardMode end

DI.check_available(::AutoZeroForward) = true
DI.mutation_support(::AutoZeroForward) = DI.MutationSupported()

DI.prepare_pushforward(f, ::AutoZeroForward, x) = NoPushforwardExtras()
DI.prepare_pushforward(f!, y, ::AutoZeroForward, x) = NoPushforwardExtras()

function DI.value_and_pushforward(f, ::AutoZeroForward, x, dx, ::NoPushforwardExtras)
    y = f(x)
    dy = zero(y)
    return y, dy
end

function DI.value_and_pushforward(f!, y, ::AutoZeroForward, x, dx, ::NoPushforwardExtras)
    f!(y, x)
    dy = zero(y)
    return y, dy
end

function DI.value_and_pushforward!(f, dy, ::AutoZeroForward, x, dx, ::NoPushforwardExtras)
    y = f(x)
    zero!(dy)
    return y, dy
end

function DI.value_and_pushforward!(
    f!, y, dy, ::AutoZeroForward, x, dx, ::NoPushforwardExtras
)
    f!(y, x)
    zero!(dy)
    return y, dy
end

## Reverse

"""
    AutoZeroReverse <: ADTypes.AbstractReverseMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: ADTypes.AbstractReverseMode end

DI.check_available(::AutoZeroReverse) = true
DI.mutation_support(::AutoZeroReverse) = DI.MutationSupported()

DI.prepare_pullback(f, ::AutoZeroReverse, x) = NoPullbackExtras()
DI.prepare_pullback(f!, y, ::AutoZeroReverse, x) = NoPullbackExtras()

function DI.value_and_pullback(f, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    y = f(x)
    dx = zero(x)
    return y, dx
end

function DI.value_and_pullback(f!, y, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    f!(y, x)
    dx = zero(x)
    return y, dx
end

function DI.value_and_pullback!(f, dx, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    y = f(x)
    zero!(dx)
    return y, dx
end

function DI.value_and_pullback!(f!, y, dx, ::AutoZeroReverse, x, dy, ::NoPullbackExtras)
    f!(y, x)
    zero!(dx)
    return y, dx
end
