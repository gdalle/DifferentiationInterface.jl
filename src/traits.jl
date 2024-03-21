const AbstractMode = AbstractADType

"""
    mode(backend)

Return the AD mode of a backend in a statically predictable way.

The return value is a `Type` object chosen among:

- `ADTypes.AbstractForwardMode`
- `ADTypes.AbstractFiniteDifferencesMode`
- `ADTypes.AbstractReverseMode`
- `ADTypes.AbstractSymbolicDifferentiationMode`

This function exists because there are backends (like Enzyme) that can support both forward and reverse mode, which means their ADTypes.jl object does not subtype either.
"""
mode(::AbstractForwardMode) = AbstractForwardMode
mode(::AbstractFiniteDifferencesMode) = AbstractFiniteDifferencesMode
mode(::AbstractReverseMode) = AbstractReverseMode
mode(::AbstractSymbolicDifferentiationMode) = AbstractSymbolicDifferentiationMode
mode(backend::SecondOrder) = mode(inner(backend)), mode(outer(backend))

## Mutation

abstract type MutationBehavior end

"""
    MutationSupported

Trait identifying backends that support mutating functions `f!(y, x)`.
"""
struct MutationSupported <: MutationBehavior end

"""
    MutationNotSupported

Trait identifying backends that do not support mutating functions `f!(y, x)`.
"""
struct MutationNotSupported <: MutationBehavior end

"""
    supports_mutation(backend)

Return [`MutationSupported`](@ref) or [`MutationNotSupported`](@ref) in a statically predictable way.
"""
supports_mutation(::AbstractADType) = MutationSupported()

## Pushforward

abstract type PushforwardBehavior end

"""
    PushforwardSupported

Trait identifying backends that support efficient pushforwards.
"""
struct PushforwardSupported <: PushforwardBehavior end

"""
    PushforwardNotSupported

Trait identifying backends that do not support efficient pushforwards.
"""
struct PushforwardNotSupported <: PushforwardBehavior end

"""
    supports_pushforward(backend)

Return [`PushforwardSupported`](@ref) or [`PushforwardNotSupported`](@ref) in a statically predictable way.
"""
supports_pushforward(backend::AbstractADType) = supports_pushforward(mode(backend))
supports_pushforward(::Type{AbstractForwardMode}) = PushforwardSupported()
supports_pushforward(::Type{AbstractFiniteDifferencesMode}) = PushforwardSupported()
supports_pushforward(::Type{AbstractReverseMode}) = PushforwardNotSupported()
supports_pushforward(::Type{AbstractSymbolicDifferentiationMode}) = PushforwardSupported()

## Pullback

abstract type PullbackBehavior end

"""
    PullbackSupported

Trait identifying backends that support efficient pullbacks.
"""
struct PullbackSupported <: PullbackBehavior end

"""
    PullbackNotSupported

Trait identifying backends that do not support efficient pullbacks.
"""
struct PullbackNotSupported <: PullbackBehavior end

"""
    supports_pullback(backend)

Return [`PullbackSupported`](@ref) or [`PullbackNotSupported`](@ref) in a statically predictable way.
"""
supports_pullback(backend::AbstractADType) = supports_pullback(mode(backend))
supports_pullback(::Type{<:AbstractForwardMode}) = PullbackNotSupported()
supports_pullback(::Type{AbstractFiniteDifferencesMode}) = PullbackNotSupported()
supports_pullback(::Type{AbstractReverseMode}) = PullbackSupported()
supports_pullback(::Type{AbstractSymbolicDifferentiationMode}) = PullbackSupported()

## Hessian-vector product

abstract type HVPBehavior end

"""
    HVPSupported

Trait identifying backends that support efficient HVPs.
"""
struct HVPSupported <: HVPBehavior end

"""
    HVPNotSupported

Trait identifying backends that do not support efficient HVPs.
"""
struct HVPNotSupported <: HVPBehavior end

"""
    supports_hvp(backend)

Return [`HVPSupported`](@ref) or [`HVPNotSupported`](@ref) in a statically predictable way.
"""
supports_hvp(backend::AbstractADType) = supports_hvp(SecondOrder(backend, backend))

function supports_hvp(backend::SecondOrder)
    return supports_hvp(mode(inner(backend)), mode(outer(backend)))
end

function supports_hvp(::Type{<:AbstractMode}, ::Type{<:AbstractMode})
    return HVPNotSupported()
end

function supports_hvp(::Type{AbstractReverseMode}, ::Type{AbstractForwardMode})
    return HVPSupported()
end

function supports_hvp(::Type{AbstractReverseMode}, ::Type{AbstractReverseMode})
    return HVPSupported()
end

function supports_hvp(::Type{AbstractForwardMode}, ::Type{AbstractReverseMode})
    return HVPSupported()
end

## Conversions

Base.Bool(::MutationSupported) = true
Base.Bool(::MutationNotSupported) = false

Base.Bool(::PushforwardSupported) = true
Base.Bool(::PushforwardNotSupported) = false

Base.Bool(::PullbackSupported) = true
Base.Bool(::PullbackNotSupported) = false

Base.Bool(::HVPSupported) = true
Base.Bool(::HVPNotSupported) = false
