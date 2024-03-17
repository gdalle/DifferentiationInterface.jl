abstract type AbstractMode end

"""
    ForwardMode

Trait identifying forward mode (and finite differences) first-order AD backends.
"""
struct ForwardMode <: AbstractMode end

"""
    ReverseMode

Trait identifying reverse mode first-order AD backends.
"""
struct ReverseMode <: AbstractMode end

"""
    SymbolicMode

Trait identifying symbolic first-order AD backends.
Their fallback structure is different from the rest.
"""
struct SymbolicMode <: AbstractMode end

"""
    mode(backend)

Return the AD mode of a backend in a statically predictable way.

This function must be overloaded for backends that support both forward and reverse.

We classify finite differences as a forward mode.
"""
function mode end

mode(::ADTypes.AbstractForwardMode) = ForwardMode()
mode(::ADTypes.AbstractFiniteDifferencesMode) = ForwardMode()
mode(::ADTypes.AbstractReverseMode) = ReverseMode()
mode(::ADTypes.AbstractSymbolicDifferentiationMode) = SymbolicMode()
