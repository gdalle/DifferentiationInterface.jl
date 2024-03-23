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

## HVP

abstract type HVPMode end

"""
    ForwardOverReverse

Traits identifying second-order backends that compute HVPs in forward over reverse mode.
"""
struct ForwardOverReverse end

"""
    ReverseOverForward

Traits identifying second-order backends that compute HVPs in reverse over forward mode.
"""
struct ReverseOverForward end

"""
    ReverseOverReverse

Traits identifying second-order backends that compute HVPs in reverse over reverse mode.
"""
struct ReverseOverReverse end

"""
    ForwardOverForward

Traits identifying second-order backends that compute HVPs in forward over forward mode (inefficient).
"""
struct ForwardOverForward end

hvp_mode(::AbstractADType) = error("HVP mode undefined for first order backend")

function hvp_mode(ba::SecondOrder)
    if Bool(supports_pushforward(outer(ba))) && Bool(supports_pullback(inner(ba)))
        return ForwardOverReverse()
    elseif Bool(supports_pullback(outer(ba))) && Bool(supports_pushforward(inner(ba)))
        return ReverseOverForward()
    elseif Bool(supports_pullback(outer(ba))) && Bool(supports_pullback(inner(ba)))
        return ReverseOverReverse()
    elseif Bool(supports_pushforward(outer(ba))) && Bool(supports_pushforward(inner(ba)))
        return ForwardOverForward()
    else
        error("HVP mode unknown")
    end
end

## Conversions

Base.Bool(::MutationSupported) = true
Base.Bool(::MutationNotSupported) = false

Base.Bool(::PushforwardSupported) = true
Base.Bool(::PushforwardNotSupported) = false

Base.Bool(::PullbackSupported) = true
Base.Bool(::PullbackNotSupported) = false
