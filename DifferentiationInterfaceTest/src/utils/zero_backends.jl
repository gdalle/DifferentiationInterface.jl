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
    f, ::AutoZeroForward, x, tx::Tangents{B}, ::NoPushforwardExtras
) where {B}
    y = f(x)
    dy = ntuple(_ -> zero(y), Val(B))
    return y, Tangents(dy...)
end

function DI.value_and_pushforward(
    f!, y, ::AutoZeroForward, x, tx::Tangents{B}, ::NoPushforwardExtras
) where {B}
    f!(y, x)
    dy = ntuple(_ -> zero(y), Val(B))
    return y, Tangents(dy...)
end

function DI.value_and_pushforward!(
    f, ty::Tangents, ::AutoZeroForward, x, tx::Tangents, ::NoPushforwardExtras
)
    y = f(x)
    for b in eachindex(ty.d)
        zero!(ty.d[b])
    end
    return y, ty
end

function DI.value_and_pushforward!(
    f!, y, ty::Tangents, ::AutoZeroForward, x, tx::Tangents, ::NoPushforwardExtras
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
    f, ::AutoZeroReverse, x, ty::Tangents{B}, ::NoPullbackExtras
) where {B}
    y = f(x)
    dx = ntuple(_ -> zero(x), Val(B))
    return y, Tangents(dx...)
end

function DI.value_and_pullback(
    f!, y, ::AutoZeroReverse, x, ty::Tangents{B}, ::NoPullbackExtras
) where {B}
    f!(y, x)
    dx = ntuple(_ -> zero(x), Val(B))
    return y, Tangents(dx...)
end

function DI.value_and_pullback!(
    f, tx::Tangents, ::AutoZeroReverse, x, ty::Tangents, ::NoPullbackExtras
)
    y = f(x)
    for b in eachindex(tx.d)
        zero!(tx.d[b])
    end
    return y, tx
end

function DI.value_and_pullback!(
    f!, y, tx::Tangents, ::AutoZeroReverse, x, ty::Tangents, ::NoPullbackExtras
)
    f!(y, x)
    for b in eachindex(tx.d)
        zero!(tx.d[b])
    end
    return y, tx
end
