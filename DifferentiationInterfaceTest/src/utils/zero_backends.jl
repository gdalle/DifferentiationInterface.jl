zero!(x::AbstractArray) = x .= zero(eltype(x))

## Forward

"""
    AutoZeroForward <: ADTypes.AbstractADType

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: AbstractADType end

ADTypes.mode(::AutoZeroForward) = ForwardMode()
DI.check_available(::AutoZeroForward) = true
DI.twoarg_support(::AutoZeroForward) = DI.TwoArgSupported()

DI.prepare_pushforward(f, ::AutoZeroForward, x, dx) = NoPushforwardExtras()
DI.prepare_pushforward(f!, y, ::AutoZeroForward, x, dx) = NoPushforwardExtras()

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
    AutoZeroReverse <: ADTypes.AbstractADType

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: AbstractADType end

ADTypes.mode(::AutoZeroReverse) = ReverseMode()
DI.check_available(::AutoZeroReverse) = true
DI.twoarg_support(::AutoZeroReverse) = DI.TwoArgSupported()

DI.prepare_pullback(f, ::AutoZeroReverse, x, dy) = NoPullbackExtras()
DI.prepare_pullback(f!, y, ::AutoZeroReverse, x, dy) = NoPullbackExtras()

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
