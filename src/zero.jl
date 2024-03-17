
"""
    AutoZeroForward <: ADTypes.AbstractForwardMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: ADTypes.AbstractForwardMode end

"""
    AutoZeroReverse <: ADTypes.AbstractReverseMode

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: ADTypes.AbstractReverseMode end

function value_and_pushforward!(dy, ::AutoZeroForward, f, x, dx, extras::Nothing)
    return f(x), zero!(dy)
end

function value_and_pullback!(dx, ::AutoZeroReverse, f, x, dy, extras::Nothing)
    return f(x), zero!(dx)
end

function value_and_pushforward!(
    y::AbstractArray, dy, ::AutoZeroForward, f!, x, dx, extras::Nothing
)
    f!(y, x)
    return y, zero!(dy)
end

function value_and_pullback!(
    y::AbstractArray, dx, ::AutoZeroReverse, f!, x, dy, extras::Nothing
)
    f!(y, x)
    return y, zero!(dx)
end
