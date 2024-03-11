abstract type AbstractImplem end

"""
    CustomImplem

Trait specifying that the custom utilities from the backend should be used as much as possible.
Used for internal dispatch only.
"""
struct CustomImplem <: AbstractImplem end

"""
    FallbackImplem

Trait specifying that the fallback utilities from DifferentiationInterface.jl should be used as much as possible, until they call a pushforward or pullback.
Used for internal dispatch only.
"""
struct FallbackImplem <: AbstractImplem end
