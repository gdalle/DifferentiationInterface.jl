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

Trait identifying backends that support two-argument functions `f!(y, x)`.
"""
struct MutationSupported <: MutationBehavior end

"""
    MutationNotSupported

Trait identifying backends that do not support two-argument functions `f!(y, x)`.
"""
struct MutationNotSupported <: MutationBehavior end

"""
    mutation_support(backend)

Return [`MutationSupported`](@ref) or [`MutationNotSupported`](@ref) in a statically predictable way.
"""
mutation_support(::AbstractADType) = MutationSupported()

## Pushforward

abstract type PushforwardPerformance end

"""
    PushforwardFast

Trait identifying backends that support efficient pushforwards.
"""
struct PushforwardFast <: PushforwardPerformance end

"""
    PushforwardSlow

Trait identifying backends that do not support efficient pushforwards.
"""
struct PushforwardSlow <: PushforwardPerformance end

"""
    pushforward_performance(backend)

Return [`PushforwardFast`](@ref) or [`PushforwardSlow`](@ref) in a statically predictable way.
"""
pushforward_performance(backend::AbstractADType) = pushforward_performance(mode(backend))
pushforward_performance(::Type{AbstractForwardMode}) = PushforwardFast()
pushforward_performance(::Type{AbstractFiniteDifferencesMode}) = PushforwardFast()
pushforward_performance(::Type{AbstractReverseMode}) = PushforwardSlow()
pushforward_performance(::Type{AbstractSymbolicDifferentiationMode}) = PushforwardFast()

## Pullback

abstract type PullbackPerformance end

"""
    PullbackFast

Trait identifying backends that support efficient pullbacks.
"""
struct PullbackFast <: PullbackPerformance end

"""
    PullbackSlow

Trait identifying backends that do not support efficient pullbacks.
"""
struct PullbackSlow <: PullbackPerformance end

"""
    pullback_performance(backend)

Return [`PullbackFast`](@ref) or [`PullbackSlow`](@ref) in a statically predictable way.
"""
pullback_performance(backend::AbstractADType) = pullback_performance(mode(backend))
pullback_performance(::Type{<:AbstractForwardMode}) = PullbackSlow()
pullback_performance(::Type{AbstractFiniteDifferencesMode}) = PullbackSlow()
pullback_performance(::Type{AbstractReverseMode}) = PullbackFast()
pullback_performance(::Type{AbstractSymbolicDifferentiationMode}) = PullbackFast()

## HVP

abstract type HVPMode end

"""
    ForwardOverReverse

Traits identifying second-order backends that compute HVPs in forward over reverse mode.
"""
struct ForwardOverReverse <: HVPMode end

"""
    ReverseOverForward

Traits identifying second-order backends that compute HVPs in reverse over forward mode.
"""
struct ReverseOverForward <: HVPMode end

"""
    ReverseOverReverse

Traits identifying second-order backends that compute HVPs in reverse over reverse mode.
"""
struct ReverseOverReverse <: HVPMode end

"""
    ForwardOverForward

Traits identifying second-order backends that compute HVPs in forward over forward mode (inefficient).
"""
struct ForwardOverForward <: HVPMode end

hvp_mode(::AbstractADType) = error("HVP mode undefined for first order backend")

function hvp_mode(ba::SecondOrder)
    if Bool(pushforward_performance(outer(ba))) && Bool(pullback_performance(inner(ba)))
        return ForwardOverReverse()
    elseif Bool(pullback_performance(outer(ba))) && Bool(pushforward_performance(inner(ba)))
        return ReverseOverForward()
    elseif Bool(pullback_performance(outer(ba))) && Bool(pullback_performance(inner(ba)))
        return ReverseOverReverse()
    elseif Bool(pushforward_performance(outer(ba))) &&
        Bool(pushforward_performance(inner(ba)))
        return ForwardOverForward()
    else
        error("HVP mode unknown")
    end
end

## Conversions

Base.Bool(::MutationSupported) = true
Base.Bool(::MutationNotSupported) = false

Base.Bool(::PushforwardFast) = true
Base.Bool(::PushforwardSlow) = false

Base.Bool(::PullbackFast) = true
Base.Bool(::PullbackSlow) = false
