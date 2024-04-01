myzero(x::Number) = zero(x)
myzero(x::AbstractArray) = zero(x)

myzero!!(x::Number) = zero(x)
myzero!!(x::AbstractArray) = x .= zero(eltype(x))

"""
    AutoZeroForward <: ADTypes.AbstractForwardMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: ADTypes.AbstractForwardMode end

DI.supports_mutation(::AutoZeroForward) = DI.MutationSupported()

"""
    AutoZeroReverse <: ADTypes.AbstractReverseMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: ADTypes.AbstractReverseMode end

DI.supports_mutation(::AutoZeroReverse) = DI.MutationSupported()

## Primitives

function DI.value_and_pushforward!!(f, dy, ::AutoZeroForward, x, dx, ::Nothing)
    y = f(x)
    dy = myzero!!(dy)
    return y, dy
end

function DI.value_and_pushforward!!(f!, y, dy, ::AutoZeroForward, x, dx, ::Nothing)
    f!(y, x)
    dy = myzero!!(dy)
    return y, dy
end

function DI.value_and_pushforward(f, ::AutoZeroForward, x, dx, ::Nothing)
    y = f(x)
    dy = myzero(y)
    return y, dy
end

function DI.value_and_pullback!!(f, dx, ::AutoZeroReverse, x, dy, ::Nothing)
    y = f(x)
    dx = myzero!!(dx)
    return y, dx
end

function DI.value_and_pullback!!(f!, y, dx, ::AutoZeroReverse, x, dy, ::Nothing)
    f!(y, x)
    dx = myzero!!(dx)
    return y, dx
end

function DI.value_and_pullback(f, ::AutoZeroReverse, x, dy, ::Nothing)
    y = f(x)
    dx = myzero(x)
    return y, dx
end
