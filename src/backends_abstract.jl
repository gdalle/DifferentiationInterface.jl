
"""
$(TYPEDEF)

Abstract type for the choice of an AD package.

The boolean `custom` parameter determines the implementation of special cases (derivative, multiderivative, gradient and jacobian):
- `custom = false`: always use generic fallbacks defined from the pushforwards and pullbacks
- `custom = true`: use backend-specific routines if they exist and generic fallbacks otherwise.

Every backend type `T{custom}` has a convenience constructor that looks more or less like this:

```julia
T(args...; custom::Bool=true) = T{custom}(args...)
```
"""
abstract type AbstractBackend{custom} end

"""
$(TYPEDEF)

Abstract subtype of [`AbstractBackend`](@ref) gathering forward mode AD packages.
"""
abstract type AbstractForwardBackend{custom} <: AbstractBackend{custom} end

"""
$(TYPEDEF)

Abstract subtype of [`AbstractBackend`](@ref) gathering reverse mode AD packages.
"""
abstract type AbstractReverseBackend{custom} <: AbstractBackend{custom} end
