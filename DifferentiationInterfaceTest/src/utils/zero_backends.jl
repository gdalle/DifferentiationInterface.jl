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

DI.prepare_pushforward(f, ::AutoZeroForward, x, tx::Tangents) = NoPushforwardExtras()
DI.prepare_pushforward(f!, y, ::AutoZeroForward, x, tx::Tangents) = NoPushforwardExtras()

function DI.value_and_pushforward(
    f, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents{B}
) where {B}
    y = f(x)
    dys = ntuple(Returns(zero(y)), Val(B))
    return y, Tangents(dys)
end

function DI.value_and_pushforward(
    f!, y, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents{B}
) where {B}
    f!(y, x)
    dys = ntuple(Returns(zero(y)), Val(B))
    return y, Tangents(dys)
end

function DI.value_and_pushforward!(
    f, ty::Tangents, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents
)
    y = f(x)
    for b in eachindex(ty.d)
        zero!(ty.d[b])
    end
    return y, ty
end

function DI.value_and_pushforward!(
    f!, y, ty::Tangents, ::NoPushforwardExtras, ::AutoZeroForward, x, tx::Tangents
)
    f!(y, x)
    for b in eachindex(ty.d)
        zero!(ty.d[b])
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
DI.check_available(::AutoZeroReverse) = true
DI.twoarg_support(::AutoZeroReverse) = DI.TwoArgSupported()

DI.prepare_pullback(f, ::AutoZeroReverse, x, ty::Tangents) = NoPullbackExtras()
DI.prepare_pullback(f!, y, ::AutoZeroReverse, x, ty::Tangents) = NoPullbackExtras()

function DI.value_and_pullback(
    f, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents{B}
) where {B}
    y = f(x)
    dxs = ntuple(Returns(zero(x)), Val(B))
    return y, Tangents(dxs)
end

function DI.value_and_pullback(
    f!, y, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents{B}
) where {B}
    f!(y, x)
    dxs = ntuple(Returns(zero(x)), Val(B))
    return y, Tangents(dxs)
end

function DI.value_and_pullback!(
    f, tx::Tangents, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents
)
    y = f(x)
    for b in eachindex(tx.d)
        zero!(tx.d[b])
    end
    return y, tx
end

function DI.value_and_pullback!(
    f!, y, tx::Tangents, ::NoPullbackExtras, ::AutoZeroReverse, x, ty::Tangents
)
    f!(y, x)
    for b in eachindex(tx.d)
        zero!(tx.d[b])
    end
    return y, tx
end
