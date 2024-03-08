
"""
    AbstractBackend

Abstract type pointing to the AD package chosen by the user, which is called a "backend".

## Custom

When we say that a backend is "custom", it describes how the utilities (derivative, multiderivative, gradient and jacobian) are implemented:

- Custom backends use specific routines defined in their package whenever they exist
- Non-custom backends use fallbacks defined in DifferentiationInterface.jl, which end up calling the pushforward or pullback
"""
abstract type AbstractBackend{custom} end

"""
    is_custom(backend)

Return a boolean `custom` that describes how utilities (derivative, multiderivative, gradient and jacobian) are implemented.
"""
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
    AbstractForwardBackend <: AbstractBackend

Abstract subtype of [`AbstractBackend`](@ref) for forward mode AD packages.
"""
abstract type AbstractForwardBackend{custom} <: AbstractBackend{custom} end

"""
    AbstractReverseBackend <: AbstractBackend

Abstract subtype of [`AbstractBackend`](@ref) for reverse mode AD packages.
"""
abstract type AbstractReverseBackend{custom} <: AbstractBackend{custom} end

"""
    autodiff_mode(backend)

Return either `:forward` or `:reverse` depending on the mode of `backend`.
"""
autodiff_mode(::AbstractForwardBackend) = :forward
autodiff_mode(::AbstractReverseBackend) = :reverse
