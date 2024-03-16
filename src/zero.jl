
"""
    AutoZeroForward <: ADTypes.AbstractForwardMode

Trivial backend that sets all derivatives to zero. Used in testing and benchmarking.
"""
struct AutoZeroForward <: AbstractForwardMode end

"""
    AutoZeroReverse <: ADTypes.AbstractReverseMode

Trivial backend that sets all derivatives to zero. Used in testing and benchmarking.
"""
struct AutoZeroReverse <: AbstractReverseMode end

function value_and_pushforward!(
    dy::Union{Number,AbstractArray}, ::AutoZeroForward, f, x, dx, extras::Nothing=nothing
)
    return f(x), zero!(dy)
end

function value_and_pullback!(
    dx::Union{Number,AbstractArray}, ::AutoZeroReverse, f, x, dy, extras::Nothing=nothing
)
    return f(x), zero!(dx)
end

function value_and_pushforward!(
    y::AbstractArray,
    dy::Union{Number,AbstractArray},
    ::AutoZeroForward,
    f!,
    x,
    dx,
    extras::Nothing=nothing,
)
    f!(y, x)
    return y, zero!(dy)
end

function value_and_pullback!(
    y::AbstractArray,
    dx::Union{Number,AbstractArray},
    ::AutoZeroReverse,
    f!,
    x,
    dy,
    extras::Nothing=nothing,
)
    f!(y, x)
    return y, zero!(dx)
end
