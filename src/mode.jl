abstract type AbstractMode end

"""
    ForwardMode

Trait identifying forward mode first-order AD backends.
"""
struct ForwardMode <: AbstractMode end

"""
    ReverseMode

Trait identifying reverse mode first-order AD backends.
"""
struct ReverseMode <: AbstractMode end

"""
    mode(backend)

Return the AD mode of a backend in a statically predictable way.

This function must be overloaded for backends that do not inherit from `ADTypes.AbstractForwardMode` or `ADTypes.AbstractReverseMode` (e.g. because they support both forward and reverse).

We classify `ADTypes.AbstractFiniteDifferencesMode` as forward mode.
"""
function mode end

mode(::AbstractForwardMode) = ForwardMode()
mode(::AbstractFiniteDifferencesMode) = ForwardMode()
mode(::AbstractReverseMode) = ReverseMode()
