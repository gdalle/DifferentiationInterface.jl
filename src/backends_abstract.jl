
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

is_custom(::AbstractBackend{custom}) where {custom} = custom

"""
    handles_input_type(backend, ::Type{X})

Check if `backend` can differentiate functions with input type `X`.
"""
handles_input_type(::AbstractBackend, ::Type{<:Number}) = true
handles_input_type(::AbstractBackend, ::Type{<:AbstractArray}) = true

"""
    handles_output_type(backend, ::Type{Y})

Check if `backend` can differentiate functions with output type `Y`.
"""
handles_output_type(::AbstractBackend, ::Type{<:Number}) = true
handles_output_type(::AbstractBackend, ::Type{<:AbstractArray}) = true

"""
    handles_types(backend, ::Type{X}, ::Type{Y})

Check if `backend` can differentiate functions with input type `X` and output type `Y`.
"""
function handles_types(backend::AbstractBackend, ::Type{X}, ::Type{Y}) where {X,Y}
    return handles_input_type(backend, X) && handles_output_type(backend, Y)
end

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

"""
    ad_mode(backend)

Return either `:forward` or `:reverse` depending on the backend.
"""
ad_mode(::AbstractForwardBackend) = :forward
ad_mode(::AbstractReverseBackend) = :reverse
