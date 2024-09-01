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
twoarg_support(::AutoZeroForward) = TwoArgSupported()

prepare_pushforward(f, ::AutoZeroForward, x, tx::Tangents) = NoPushforwardExtras()
prepare_pushforward(f!, y, ::AutoZeroForward, x, tx::Tangents) = NoPushforwardExtras()

function value_and_pushforward(
    f, ::AutoZeroForward, x, tx::Tangents{B}, ::NoPushforwardExtras
) where {B}
    y = f(x)
    dys = ntuple(ReturnZero(y), Val(B))
    return y, Tangents(dys)
end

function value_and_pushforward(
    f!, y, ::AutoZeroForward, x, tx::Tangents{B}, ::NoPushforwardExtras
) where {B}
    f!(y, x)
    dys = ntuple(ReturnZero(y), Val(B))
    return y, Tangents(dys)
end

function value_and_pushforward!(
    f, ty::Tangents, ::AutoZeroForward, x, tx::Tangents, ::NoPushforwardExtras
)
    error()
    y = f(x)
    for b in eachindex(ty.d)
        _zero!(ty.d[b])
    end
    return y, ty
end

function value_and_pushforward!(
    f!, y, ty::Tangents, ::AutoZeroForward, x, tx::Tangents, ::NoPushforwardExtras
)
    error()
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
twoarg_support(::AutoZeroReverse) = TwoArgSupported()

prepare_pullback(f, ::AutoZeroReverse, x, ty::Tangents) = NoPullbackExtras()
prepare_pullback(f!, y, ::AutoZeroReverse, x, ty::Tangents) = NoPullbackExtras()

function value_and_pullback(
    f, ::AutoZeroReverse, x, ty::Tangents{B}, ::NoPullbackExtras
) where {B}
    y = f(x)
    dxs = ntuple(ReturnZero(x), Val(B))
    return y, Tangents(dxs)
end

function value_and_pullback(
    f!, y, ::AutoZeroReverse, x, ty::Tangents{B}, ::NoPullbackExtras
) where {B}
    f!(y, x)
    dxs = ntuple(ReturnZero(x), Val(B))
    return y, Tangents(dxs)
end

function value_and_pullback!(
    f, tx::Tangents, ::AutoZeroReverse, x, ty::Tangents, ::NoPullbackExtras
)
    y = f(x)
    for b in eachindex(tx.d)
        _zero!(tx.d[b])
    end
    return y, tx
end

function value_and_pullback!(
    f!, y, tx::Tangents, ::AutoZeroReverse, x, ty::Tangents, ::NoPullbackExtras
)
    f!(y, x)
    for b in eachindex(tx.d)
        _zero!(tx.d[b])
    end
    return y, tx
end
