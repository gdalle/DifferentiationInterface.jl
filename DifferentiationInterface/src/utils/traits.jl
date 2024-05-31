## Mutation

abstract type MutationBehavior end

"""
    TwoArgSupported

Trait identifying backends that support two-argument functions `f!(y, x)`.
"""
struct TwoArgSupported <: MutationBehavior end

"""
    TwoArgNotSupported

Trait identifying backends that do not support two-argument functions `f!(y, x)`.
"""
struct TwoArgNotSupported <: MutationBehavior end

"""
    twoarg_support(backend)

Return [`TwoArgSupported`](@ref) or [`TwoArgNotSupported`](@ref) in a statically predictable way.
"""
twoarg_support(::AbstractADType) = TwoArgSupported()

function twoarg_support(backend::SecondOrder)
    if Bool(twoarg_support(inner(backend))) && Bool(twoarg_support(outer(backend)))
        return TwoArgSupported()
    else
        return TwoArgNotSupported()
    end
end

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
pushforward_performance(::ForwardMode) = PushforwardFast()
pushforward_performance(::ForwardOrReverseMode) = PushforwardFast()
pushforward_performance(::ReverseMode) = PushforwardSlow()
pushforward_performance(::SymbolicMode) = PushforwardFast()

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
pullback_performance(::ForwardMode) = PullbackSlow()
pullback_performance(::ForwardOrReverseMode) = PullbackFast()
pullback_performance(::ReverseMode) = PullbackFast()
pullback_performance(::SymbolicMode) = PullbackFast()

## HVP

abstract type HVPMode end

"""
    ForwardOverForward

Trait identifying second-order backends that compute HVPs in forward over forward mode (inefficient).
"""
struct ForwardOverForward <: HVPMode end

"""
    ForwardOverReverse

Trait identifying second-order backends that compute HVPs in forward over reverse mode.
"""
struct ForwardOverReverse <: HVPMode end

"""
    ReverseOverForward

Trait identifying second-order backends that compute HVPs in reverse over forward mode.
"""
struct ReverseOverForward <: HVPMode end

"""
    ReverseOverReverse

Trait identifying second-order backends that compute HVPs in reverse over reverse mode.
"""
struct ReverseOverReverse <: HVPMode end

"""
    hvp_mode(backend)

Return, by order of preference and depending on `backend`:

1. [`ForwardOverReverse`](@ref)
2. [`ReverseOverForward`](@ref)
3. [`ReverseOverReverse`](@ref)
4. [`ForwardOverForward`](@ref)
"""
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

Base.Bool(::TwoArgSupported) = true
Base.Bool(::TwoArgNotSupported) = false

Base.Bool(::PushforwardFast) = true
Base.Bool(::PushforwardSlow) = false

Base.Bool(::PullbackFast) = true
Base.Bool(::PullbackSlow) = false
