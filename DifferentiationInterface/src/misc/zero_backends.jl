struct ReturnZero{T}
    template::T
end

(rz::ReturnZero)(i) = zero(rz.template)

_zero!(x::AbstractArray) = x .= zero(eltype(x))

## Forward

"""
    AutoZeroForward <: ADTypes.AbstractADType

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroForward <: AbstractADType end

ADTypes.mode(::AutoZeroForward) = ForwardMode()
check_available(::AutoZeroForward) = true
inplace_support(::AutoZeroForward) = InPlaceSupported()

function prepare_pushforward(
    f::F, ::AutoZeroForward, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    return NoPushforwardPrep()
end

function prepare_pushforward(
    f!::F, y, ::AutoZeroForward, x, tx::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    return NoPushforwardPrep()
end

function value_and_pushforward(
    f::F,
    ::NoPushforwardPrep,
    ::AutoZeroForward,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    y = f(x, map(unwrap, contexts)...)
    ty = map(ReturnZero(y), tx)
    return y, ty
end

function value_and_pushforward(
    f!::F,
    y,
    ::NoPushforwardPrep,
    ::AutoZeroForward,
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f!(y, x, map(unwrap, contexts)...)
    ty = map(ReturnZero(y), tx)
    return y, ty
end

function value_and_pushforward!(
    f::F,
    ty::Tangents,
    ::NoPushforwardPrep,
    ::AutoZeroForward,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    for b in eachindex(ty.d)
        _zero!(ty.d[b])
    end
    return y, ty
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    ::NoPushforwardPrep,
    ::AutoZeroForward,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    f!(y, x, map(unwrap, contexts)...)
    for b in eachindex(ty.d)
        _zero!(ty.d[b])
    end
    return y, ty
end

## Reverse

"""
    AutoZeroReverse <: ADTypes.AbstractADType

Trivial backend that sets all derivatives to zero.
Used in testing and benchmarking.
"""
struct AutoZeroReverse <: AbstractADType end

ADTypes.mode(::AutoZeroReverse) = ReverseMode()
check_available(::AutoZeroReverse) = true
inplace_support(::AutoZeroReverse) = InPlaceSupported()

function prepare_pullback(
    f::F, ::AutoZeroReverse, x, ty::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    return NoPullbackPrep()
end

function prepare_pullback(
    f!::F, y, ::AutoZeroReverse, x, ty::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    return NoPullbackPrep()
end

function value_and_pullback(
    f::F,
    ::NoPullbackPrep,
    ::AutoZeroReverse,
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    y = f(x, map(unwrap, contexts)...)
    dxs = ntuple(ReturnZero(x), Val(B))
    return y, Tangents(dxs...)
end

function value_and_pullback(
    f!::F,
    y,
    ::NoPullbackPrep,
    ::AutoZeroReverse,
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f!(y, x, map(unwrap, contexts)...)
    dxs = ntuple(ReturnZero(x), Val(B))
    return y, Tangents(dxs...)
end

function value_and_pullback!(
    f::F,
    tx::Tangents,
    ::NoPullbackPrep,
    ::AutoZeroReverse,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    for b in eachindex(tx.d)
        _zero!(tx.d[b])
    end
    return y, tx
end

function value_and_pullback!(
    f!::F,
    y,
    tx::Tangents,
    ::NoPullbackPrep,
    ::AutoZeroReverse,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    f!(y, x, map(unwrap, contexts)...)
    for b in eachindex(tx.d)
        _zero!(tx.d[b])
    end
    return y, tx
end
