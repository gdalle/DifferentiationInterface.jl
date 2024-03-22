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

function DI.value_and_pushforward!(f::F, dy, ::AutoZeroForward, x, dx, ::Nothing) where {F}
    y = f(x)
    dy = myzero!(dy)
    return y, dy
end

function DI.value_and_pushforward!(
    f!::F, y, dy, ::AutoZeroForward, x, dx, ::Nothing
) where {F}
    f!(y, x)
    dy = myzero!(dy)
    return y, dy
end

function DI.value_and_pushforward(f::F, ::AutoZeroForward, x, dx, ::Nothing) where {F}
    y = f(x)
    dy = myzero(y)
    return y, dy
end

function DI.value_and_pullback!(f::F, dx, ::AutoZeroReverse, x, dy, ::Nothing) where {F}
    y = f(x)
    dx = myzero!(dx)
    return y, dx
end

function DI.value_and_pullback!(f!::F, y, dx, ::AutoZeroReverse, x, dy, ::Nothing) where {F}
    f!(y, x)
    dx = myzero!(dx)
    return y, dx
end

function DI.value_and_pullback(f::F, ::AutoZeroReverse, x, dy, ::Nothing) where {F}
    y = f(x)
    dx = myzero(x)
    return y, dx
end
