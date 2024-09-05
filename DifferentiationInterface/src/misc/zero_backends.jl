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

prepare_pushforward(f, ::AutoZeroForward, x, tx::Tangents) = NoPushforwardExtras()
prepare_pushforward(f!, y, ::AutoZeroForward, x, tx::Tangents) = NoPushforwardExtras()

function value_and_pushforward(
    f, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents{B}
) where {B}
    y = f(x)
    dys = ntuple(ReturnZero(y), Val(B))
    return y, Tangents(dys)
end

function value_and_pushforward(
    f!, y, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents{B}
) where {B}
    f!(y, x)
    dys = ntuple(ReturnZero(y), Val(B))
    return y, Tangents(dys)
end

function value_and_pushforward!(
    f, ty::Tangents, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents
)
    y = f(x)
    for b in eachindex(ty.d)
        _zero!(ty.d[b])
    end
    return y, ty
end

function value_and_pushforward!(
    f!, y, ty::Tangents, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents
)
    f!(y, x)
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

prepare_pullback(f, ::AutoZeroReverse, x, ty::Tangents) = NoPullbackExtras()
prepare_pullback(f!, y, ::AutoZeroReverse, x, ty::Tangents) = NoPullbackExtras()

function value_and_pullback(
    f, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents{B}
) where {B}
    y = f(x)
    dxs = ntuple(ReturnZero(x), Val(B))
    return y, Tangents(dxs)
end

function value_and_pullback(
    f!, y, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents{B}
) where {B}
    f!(y, x)
    dxs = ntuple(ReturnZero(x), Val(B))
    return y, Tangents(dxs)
end

function value_and_pullback!(
    f, tx::Tangents, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents
)
    y = f(x)
    for b in eachindex(tx.d)
        _zero!(tx.d[b])
    end
    return y, tx
end

function value_and_pullback!(
    f!, y, tx::Tangents, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents
)
    f!(y, x)
    for b in eachindex(tx.d)
        _zero!(tx.d[b])
    end
    return y, tx
end
