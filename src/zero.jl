
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

function value_and_pushforward!(
    dy, ::AutoZeroForward, f::F, x, dx, extras::Nothing
) where {F}
    return f(x), zero!(dy)
end

function value_and_pullback!(dx, ::AutoZeroReverse, f::F, x, dy, extras::Nothing) where {F}
    return f(x), zero!(dx)
end

function value_and_pushforward!(
    y::AbstractArray, dy, ::AutoZeroForward, f!::F, x, dx, extras::Nothing
) where {F}
    f!(y, x)
    return y, zero!(dy)
end

function value_and_pullback!(
    y::AbstractArray, dx, ::AutoZeroReverse, f!::F, x, dy, extras::Nothing
) where {F}
    f!(y, x)
    return y, zero!(dx)
end
