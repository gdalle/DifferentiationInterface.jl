## Traits and access

"""
    autodiff_mode(backend)

Return `ForwardMode()` or `ReverseMode()` in a statically predictable way.

This function must be overloaded for backends that do not inherit from `ADTypes.AbstractForwardMode` or `ADTypes.AbstractReverseMode` (e.g. because they support both forward and reverse).

We classify `ADTypes.AbstractFiniteDifferencesMode` as forward mode.
"""
autodiff_mode(::AbstractForwardMode) = ForwardMode()
autodiff_mode(::AbstractFiniteDifferencesMode) = ForwardMode()
autodiff_mode(::AbstractReverseMode) = ReverseMode()
