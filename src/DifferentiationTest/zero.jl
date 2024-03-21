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

function DI.value_and_pushforward!(
    dy, ::AutoZeroForward, f::F, x, dx, extras::Nothing
) where {F}
    y = f(x)
    dy = zero!(dy)
    return y, dy
end

function DI.value_and_pushforward(::AutoZeroForward, f::F, x, dx, extras::Nothing) where {F}
    y = f(x)
    dy = zero(y)
    return y, dy
end

function DI.value_and_pushforward!(
    y::AbstractArray, dy, ::AutoZeroForward, f!::F, x, dx, extras::Nothing
) where {F}
    f!(y, x)
    dy = zero!(dy)
    return y, dy
end

function DI.value_and_pullback!(
    dx, ::AutoZeroReverse, f::F, x, dy, extras::Nothing
) where {F}
    y = f(x)
    dx = zero!(dx)
    return y, dx
end

function DI.value_and_pullback(::AutoZeroReverse, f::F, x, dy, extras::Nothing) where {F}
    y = f(x)
    dx = zero(x)
    return y, dx
end

function DI.value_and_pullback!(
    y::AbstractArray, dx, ::AutoZeroReverse, f!::F, x, dy, extras::Nothing
) where {F}
    f!(y, x)
    dx = zero!(dx)
    return y, dx
end
